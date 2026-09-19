from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import multiprocessing
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from torch import nn

from Model.policy import ClashRLModel, NUM_ACTIONS, CARDS_COUNT

from clasher.rl.selfplay_env import LOW_ELIXIR_PENALTY, SelfPlayBattleEnv
from .accel import (
    AMP_CHOICES,
    autocast,
    configure_torch,
    default_num_workers,
    describe_device,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_torch_device,
    worker_process_init,
)
from ..path import DECKS_FILE

@dataclass
class Transition:
    player_id: int
    board: np.ndarray
    entities: np.ndarray
    entity_mask: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    next_value: float
    rollout_id: int
    hidden_state: np.ndarray


@dataclass(frozen=True)
class RolloutWorkerTask:
    model_state_dict: Dict[str, torch.Tensor]
    rollout_steps: int
    decision_interval_ticks: int
    max_ticks: int
    decks_path: str
    mirror_match: bool
    quiet_engine: bool
    seed: int
    num_think_steps: int
    low_elixir_penalty: float


_WORKER_MODEL: Optional[ClashRLModel] = None
_WORKER_ENV: Optional[SelfPlayBattleEnv] = None
_WORKER_CONFIG: Optional[Tuple[Any, ...]] = None
_WORKER_THREADS_SET: bool = False


@contextmanager
def maybe_silence_stdio(enabled: bool):
    if not enabled:
        yield
        return

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield

def model_inputs(obs_list, device):
    board = torch.as_tensor(np.stack([obs.board for obs in obs_list]), dtype=torch.float32, device=device)
    entities = torch.as_tensor(np.stack([obs.entities for obs in obs_list]), dtype=torch.float32, device=device)
    entity_mask = torch.as_tensor(np.stack([obs.entity_mask for obs in obs_list]), dtype=torch.bool, device=device)
    global_features = torch.as_tensor(np.stack([obs.global_features for obs in obs_list]), dtype=torch.float32, device=device)
    return board, entities, entity_mask, global_features

def collect_rollout(env: SelfPlayBattleEnv, model : ClashRLModel, device: torch.device, rollout_steps: int, quiet_engine: bool) -> List[Transition]:
    model.eval()
    transitions: List[Transition] = []
    hidden_state = model.init_hidden_state(2, device)

    if env.battle is None:
        with maybe_silence_stdio(quiet_engine):
            env.reset()

    for _ in range(rollout_steps):
        step_data: Dict[int, Dict[str, object]] = {}
        actions: Dict[int, int] = {}
        obs0 = env.get_observation(0)
        obs1 = env.get_observation(1)
        mask0 = env.get_action_mask(0)
        mask1 = env.get_action_mask(1)

        board_t, entities_t, entity_mask_t, global_features_t = model_inputs([obs0, obs1], device)
        mask_t = torch.as_tensor(np.stack([mask0, mask1]), dtype=torch.bool, device=device)

        hidden_before = hidden_state.clone()

        with torch.inference_mode():
            logits, values, hidden_state = model(global_features_t, board_t, entities_t, entity_mask_t, hidden_state)

        masked_logits = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
        value_t = values.squeeze(-1)

        action_arr = action_t.detach().cpu().numpy()
        log_prob_arr = log_prob_t.detach().cpu().numpy()
        value_arr = value_t.detach().cpu().numpy()

        actions[0] = int(action_arr[0])
        actions[1] = int(action_arr[1])
        step_data[0] = {
            "board": obs0.board,
            "entities": obs0.entities,
            "entity_mask": obs0.entity_mask,
            "global_features": obs0.global_features,
            "mask": mask0,
            "action": actions[0],
            "old_log_prob": float(log_prob_arr[0]),
            "value": float(value_arr[0]),
        }

        step_data[1] = {
            "board": obs1.board,
            "entities": obs1.entities,
            "entity_mask": obs1.entity_mask,
            "global_features": obs1.global_features,
            "mask": mask1,
            "action": actions[1],
            "old_log_prob": float(log_prob_arr[1]),
            "value": float(value_arr[1]),
        }

        with maybe_silence_stdio(quiet_engine):
            rewards, done, _ = env.step(actions)

        for player_id in (0, 1):
            pdata = step_data[player_id]
            transitions.append(
                Transition(
                    player_id=player_id,
                    board=pdata["board"],
                    entities=pdata["entities"],
                    entity_mask=pdata["entity_mask"],
                    global_features=pdata["global_features"],
                    action_mask=pdata["mask"],
                    action=pdata["action"],
                    old_log_prob=pdata["old_log_prob"],
                    value=pdata["value"],
                    reward=float(rewards[player_id]),
                    done=done,
                    next_value=0.0,
                    rollout_id=0,
                    hidden_state=hidden_before[0, player_id].cpu().numpy().copy()
                )
            )

        if done:
            with maybe_silence_stdio(quiet_engine):
                env.reset()
            hidden_state = model.init_hidden_state(2, device)

    _fill_next_values(transitions, env=env, model=model, device=device, hidden_state=hidden_state)
    return transitions


def split_rollout_steps(total_steps: int, num_workers: int) -> List[int]:
    workers = max(1, num_workers)
    base = total_steps // workers
    remainder = total_steps % workers
    chunks: List[int] = []
    for idx in range(workers):
        chunk = base + (1 if idx < remainder else 0)
        if chunk > 0:
            chunks.append(chunk)
    return chunks


def _collect_rollout_worker(task: RolloutWorkerTask) -> List[Transition]:
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_CONFIG, _WORKER_THREADS_SET
    # Rollout workers are CPU-only: env stepping and action masking are
    # CPU-bound and batch-1 inference is faster on a CPU core than on a GPU
    # (see accel.py). This also keeps GPU memory free for the learner.
    torch_device = torch.device("cpu")
    if not _WORKER_THREADS_SET:
        # Avoid CPU thread oversubscription with many actor processes.
        torch.set_num_threads(1)
        _WORKER_THREADS_SET = True
    config = (task.decision_interval_ticks, task.max_ticks, task.decks_path, task.mirror_match, task.num_think_steps, task.low_elixir_penalty)
    
    if _WORKER_MODEL is None or _WORKER_ENV is None or _WORKER_CONFIG != config:
        torch.manual_seed(task.seed)
        np.random.seed(task.seed)

        _WORKER_MODEL = ClashRLModel(num_think_steps=task.num_think_steps).to(torch_device)

        _WORKER_ENV = SelfPlayBattleEnv(
            decision_interval_ticks=task.decision_interval_ticks,
            max_ticks=task.max_ticks,
            decks_path=task.decks_path,
            seed=task.seed,
            mirror_match=task.mirror_match,
            canonical_perspective=True,
            low_elixir_penalty=task.low_elixir_penalty,
        )
        with maybe_silence_stdio(task.quiet_engine):
            _WORKER_ENV.reset()
        _WORKER_CONFIG = config

    model = _WORKER_MODEL
    env = _WORKER_ENV
    assert model is not None
    assert env is not None
    model.load_state_dict(task.model_state_dict, strict=False)
    model.eval()
    return collect_rollout(env=env, model=model, device=torch_device, rollout_steps=task.rollout_steps, quiet_engine=task.quiet_engine)


def collect_rollout_parallel(executor: ProcessPoolExecutor, model: ClashRLModel, rollout_steps: int, num_workers: int, decision_interval_ticks: int,
    max_ticks: int, decks_path: str, mirror_match: bool, quiet_engine: bool, seed: int, worker_retries: int, num_think_steps: int, low_elixir_penalty: float) -> List[Transition]:
    
    chunks = split_rollout_steps(rollout_steps, num_workers)
    if not chunks:
        return []

    # Copy to CPU tensors for process transport.
    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tasks = [
        RolloutWorkerTask(
            model_state_dict=model_state_dict,
            rollout_steps=chunk_steps,
            decision_interval_ticks=decision_interval_ticks,
            max_ticks=max_ticks,
            decks_path=decks_path,
            mirror_match=mirror_match,
            quiet_engine=quiet_engine,
            seed=seed + (worker_idx + 1) * 1009,
            num_think_steps=num_think_steps,
            low_elixir_penalty=low_elixir_penalty,
        )
        for worker_idx, chunk_steps in enumerate(chunks)
    ]

    attempts: Dict[int, int] = {idx: 0 for idx in range(len(tasks))}
    pending: List[tuple[int, RolloutWorkerTask]] = list(enumerate(tasks))
    parts_by_idx: Dict[int, List[Transition]] = {}

    while pending:
        submitted = {
            executor.submit(_collect_rollout_worker, task): (idx, task)
            for idx, task in pending
        }
        pending = []

        for future in as_completed(submitted):
            idx, task = submitted[future]
            try:
                parts_by_idx[idx] = future.result()
            except BrokenProcessPool:
                raise
            except Exception as exc:
                attempts[idx] += 1
                print(
                    f"worker_failure idx={idx} attempt={attempts[idx]} "
                    f"error={type(exc).__name__}: {exc}"
                )
                if attempts[idx] > worker_retries:
                    raise RuntimeError(f"worker {idx} failed after {worker_retries + 1} attempts") from exc
                pending.append((idx, task))

    transitions: List[Transition] = []
    for idx in range(len(tasks)):
        part = parts_by_idx[idx]
        for transition in part:
            transition.rollout_id = idx
        transitions.extend(part)
        
    return transitions


def _fill_next_values(transitions: List[Transition], env: SelfPlayBattleEnv, model: ClashRLModel, device: torch.device, hidden_state: torch.Tensor) -> None:
    idx_by_player: Dict[int, List[int]] = {0: [], 1: []}
    for idx, tr in enumerate(transitions):
        idx_by_player[tr.player_id].append(idx)

    bootstrap_value: Dict[int, float] = {0: 0.0, 1: 0.0}
    if env.battle is not None and not env.battle.game_over:
        next_obs0 = env.get_observation(0)
        next_obs1 = env.get_observation(1)

        with torch.no_grad():
            next_board_t, next_entities_t, next_entity_mask_t, next_global_features_t = model_inputs([next_obs0, next_obs1], device)
            
            _, value, _ = model(next_global_features_t, next_board_t, next_entities_t, next_entity_mask_t, hidden_state)

            values = value.squeeze(-1).detach().cpu().numpy()

            bootstrap_value[0] = float(values[0])
            bootstrap_value[1] = float(values[1])

    for player_id in (0, 1):
        indices = idx_by_player[player_id]
        for offset, idx in enumerate(indices):
            tr = transitions[idx]
            if tr.done:
                tr.next_value = 0.0
            elif offset + 1 < len(indices):
                tr.next_value = transitions[indices[offset + 1]].value
            else:
                tr.next_value = bootstrap_value[player_id]


def compute_gae(transitions: List[Transition], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    sequences: Dict[Tuple[int, int], List[int]] = {}

    for idx, transition in enumerate(transitions):
        key = (transition.rollout_id, transition.player_id)
        sequences.setdefault(key, []).append(idx)

    for indices in sequences.values():
        gae = 0.0

        for idx in reversed(indices):
            tr = transitions[idx]
            non_terminal = 0.0 if tr.done else 1.0
            delta = tr.reward + gamma * non_terminal * tr.next_value - tr.value
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[idx] = gae
            returns[idx] = gae + tr.value

    return advantages, returns


def ppo_update(model: ClashRLModel, optimizer: torch.optim.Optimizer, transitions: List[Transition], advantages: np.ndarray, returns: np.ndarray, 
    clip_ratio: float, value_coef: float, entropy_coef: float, epochs: int, batch_size: int, sequence_length: int, device: torch.device,
    amp_dtype=None, scaler: Optional["torch.amp.GradScaler"] = None) -> Dict[str, float]:
    
    model.train()

    if not transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    sequences: Dict[Tuple[int, int], List[int]] = {}

    for idx, transition in enumerate(transitions):
        key = (transition.rollout_id, transition.player_id)
        sequences.setdefault(key, []).append(idx)

    chunks: List[List[int]] = []

    for indices in sequences.values():
        for start in range(0, len(indices), sequence_length):
            chunks.append(indices[start:start + sequence_length])

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    advantages_t = ((advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8))

    sequence_batch_size = max(1, batch_size // sequence_length)

    # Stack every transition once and move it to the device once; each
    # mini-batch is then a gather on the device instead of a Python loop
    # copying transitions into fresh arrays.
    def stacked(values, dtype) -> torch.Tensor:
        return torch.from_numpy(np.stack(values).astype(dtype, copy=False)).to(device)

    all_boards = stacked([t.board for t in transitions], np.float32)
    all_entities = stacked([t.entities for t in transitions], np.float32)
    all_entity_masks = stacked([t.entity_mask for t in transitions], np.bool_)
    all_global_features = stacked([t.global_features for t in transitions], np.float32)
    all_action_masks = stacked([t.action_mask for t in transitions], np.bool_)
    all_actions = stacked([t.action for t in transitions], np.int64)
    all_old_log_probs = stacked([t.old_log_prob for t in transitions], np.float32)
    all_dones = stacked([t.done for t in transitions], np.bool_)
    all_hidden = stacked([t.hidden_state for t in transitions], np.float32)

    stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    steps = 0

    for _ in range(epochs):
        order = torch.randperm(len(chunks)).tolist()

        for start in range(0, len(order), sequence_batch_size):
            batch_chunks = [chunks[idx] for idx in order[start:start + sequence_batch_size]]
            batch_size_actual = len(batch_chunks)
            max_len = max(len(chunk) for chunk in batch_chunks)

            index_np = np.zeros((batch_size_actual, max_len), dtype=np.int64)
            valid_np = np.zeros((batch_size_actual, max_len), dtype=bool)
            for batch_idx, chunk in enumerate(batch_chunks):
                index_np[batch_idx, :len(chunk)] = chunk
                index_np[batch_idx, len(chunk):] = chunk[-1]  # padding, masked out below
                valid_np[batch_idx, :len(chunk)] = True
            index = torch.from_numpy(index_np).to(device)
            valid_mask = torch.from_numpy(valid_np).to(device)

            boards = all_boards[index]
            entities = all_entities[index]
            entity_masks = all_entity_masks[index]
            global_features = all_global_features[index]
            action_masks = all_action_masks[index] | ~valid_mask.unsqueeze(-1)
            actions = all_actions[index]
            old_log_probs = all_old_log_probs[index]
            advantages_batch = advantages_t[index]
            returns_batch = returns_t[index]
            done_mask = all_dones[index] | ~valid_mask
            hidden_state = all_hidden[index[:, 0]].unsqueeze(0)

            with autocast(device, amp_dtype):
                logits, values = model.forward_sequence(global_features=global_features, board=boards, entities=entities, entity_mask=entity_masks, done_mask=done_mask, hidden_state=hidden_state,)

            # Loss maths in fp32: -1e9 masking and log-softmax overflow in fp16.
            logits = logits.float()
            values = values.float()
            masked_logits = logits.masked_fill(~action_masks, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(actions)
            # Entropy of the placement given that a card is played, not of the
            # flat distribution: no-op is 1 action vs ~1000 legal placements, so
            # flat entropy is maximised by never no-opping and the bonus itself
            # taught the policy to spam. This keeps exploring what/where to
            # play without rewarding how often.
            place_mask = action_masks[..., :-1]
            place_dist = torch.distributions.Categorical(logits=masked_logits[..., :-1])
            entropy = place_dist.entropy() * place_mask.any(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped = ratio * advantages_batch
            clipped = (torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_batch)

            valid_advantages = torch.min(unclipped, clipped)[valid_mask]
            policy_loss = -valid_advantages.mean()

            values = values.squeeze(-1)

            valid_values = values[valid_mask]
            valid_returns = returns_batch[valid_mask]
            value_loss = 0.5 * torch.mean((valid_returns - valid_values) ** 2)

            entropy_loss = entropy[valid_mask].mean()

            loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy_loss)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            stats["loss"] += float(loss.item())
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy_loss.item())
            steps += 1

    if steps > 0:
        for key in stats:
            stats[key] /= steps

    return stats


def resolve_entropy_coef(update: int, start_update: int, entropy_coef_start: float, entropy_coef_end: float, entropy_coef_anneal_updates: int) -> float:
    """Linearly anneal entropy_coef from entropy_coef_start down to
    entropy_coef_end over entropy_coef_anneal_updates updates, counted from
    start_update (i.e. from whenever THIS run began, not from update 0 -
    so resuming a run picks up the anneal schedule fresh from wherever you
    are now, which is exactly what you want for "recover a collapsed
    policy": a fresh burst of high entropy_coef starting right now).
    After the anneal window, holds at entropy_coef_end.
    """
    if entropy_coef_anneal_updates <= 0:
        return entropy_coef_end

    progress = (update - start_update) / float(entropy_coef_anneal_updates)
    progress = max(0.0, min(1.0, progress))
    return entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CV-equivalent self-play policy")
    parser.add_argument("--decks-path", type=str, default=str(DECKS_FILE))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--decision-interval", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=9090)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor. Bumped from an earlier 0.995 default - at "
        "0.995 the effective horizon (~1/(1-gamma) decisions) is roughly "
        "200 decisions, which is shorter than most full matches at this "
        "decision-interval/max-ticks setting, so reward for a payoff late "
        "in a match (e.g. baiting then punishing) barely propagates back "
        "to the decision to hold elixir earlier. 0.999 gives a ~1000 "
        "decision horizon, closer to covering a full match.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Used directly when --entropy-coef-anneal-updates is 0 (no annealing). "
        "When annealing is enabled, this is IGNORED in favor of "
        "--entropy-coef-start/--entropy-coef-end.",
    )
    parser.add_argument(
        "--entropy-coef-start",
        type=float,
        default=None,
        help="Entropy coefficient at the start of THIS run (i.e. right now, "
        "including on resume). Set high (e.g. 0.1-0.2) to force exploration "
        "back open on an already-collapsed/converged policy. Defaults to "
        "--entropy-coef if not set.",
    )
    parser.add_argument(
        "--entropy-coef-end",
        type=float,
        default=None,
        help="Entropy coefficient after the anneal window finishes - the "
        "normal steady-state value. Defaults to --entropy-coef if not set.",
    )
    parser.add_argument(
        "--entropy-coef-anneal-updates",
        type=int,
        default=0,
        help="Number of updates (counted from wherever THIS run starts, so "
        "resuming restarts the schedule) over which entropy_coef linearly "
        "anneals from --entropy-coef-start to --entropy-coef-end. 0 disables "
        "annealing entirely and just uses --entropy-coef as a flat value.",
    )
    parser.add_argument(
        "--low-elixir-penalty",
        type=float,
        default=LOW_ELIXIR_PENALTY,
        help="Max per-decision penalty for sitting below 2 elixir (scaled by "
        "how far below; 0 disables). Counters card spamming.",
    )
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--num-think-steps",
        type=int,
        default=0,
        help="Number of ThinkingBlock refinement passes the model runs "
        "before its policy/value heads, per decision. 0 (default) disables "
        "thinking entirely - byte-identical behavior to the pre-thinking "
        "architecture. The block's weights are always present in the model "
        "regardless of this value, so a checkpoint saved with "
        "--num-think-steps 0 can later be resumed with a higher value (the "
        "block starts untrained and needs some fine-tuning), and a "
        "checkpoint trained WITH thinking can be deployed with a lower "
        "value (or 0) for a speed/quality tradeoff - see "
        "ClashRLModel.forward's num_think_steps override for the "
        "no-retrain-needed version of that at inference time.",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/selfplay")
    parser.add_argument("--mirror-match", action="store_true")
    parser.add_argument("--quiet-engine", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Rollout worker processes (CPU). 0 = auto: all cores but one.",
    )
    parser.add_argument(
        "--amp",
        choices=AMP_CHOICES,
        default="auto",
        help="Mixed precision for the PPO update on GPU. auto = bf16 (else fp16) "
        "on NVIDIA, fp32 on AMD/ROCm (half precision crashes ROCm 7.14 on RDNA4; "
        "force it with bf16/fp16 at your own risk).",
    )
    parser.add_argument("--worker-retries", type=int, default=2)
    parser.add_argument("--max-update-restarts", type=int, default=5)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates = sorted(checkpoint_dir.glob("policy_update_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def main() -> None:
    args = parse_args()
    if args.resume_latest and args.resume_from:
        raise ValueError("Use only one of --resume-latest or --resume-from")

    entropy_coef_start = (
        args.entropy_coef_start if args.entropy_coef_start is not None else args.entropy_coef
    )
    entropy_coef_end = (
        args.entropy_coef_end if args.entropy_coef_end is not None else args.entropy_coef
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_torch_device(args.device)
    configure_torch(device)
    amp_dtype = resolve_amp_dtype(device, args.amp)
    if amp_dtype is not None and device.type == "cuda" and torch.version.hip:
        print("warning: half precision on ROCm can crash the GPU (hipBLASLt memory fault); use --amp off if it does")
    scaler = make_grad_scaler(device, amp_dtype)
    if args.num_workers <= 0:
        args.num_workers = default_num_workers()
    rollout_device = torch.device("cpu")
    print(f"device={device} [{describe_device(device)}] amp={amp_dtype} rollout_device=cpu num_workers={args.num_workers}")

    env = SelfPlayBattleEnv(
        decision_interval_ticks=args.decision_interval,
        max_ticks=args.max_ticks,
        decks_path=args.decks_path,
        seed=args.seed,
        mirror_match=args.mirror_match,
        canonical_perspective=True,
        low_elixir_penalty=args.low_elixir_penalty,
    )

    with maybe_silence_stdio(args.quiet_engine):
        env.reset()

    model = ClashRLModel(num_think_steps=args.num_think_steps).to(device)
    # CPU copy used for single-process rollouts (synced after every update).
    rollout_model = ClashRLModel(num_think_steps=args.num_think_steps).to(rollout_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_update = 1

    resume_checkpoint: Optional[Path] = None
    if args.resume_from:
        resume_checkpoint = Path(args.resume_from)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
    elif args.resume_latest:
        resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=device)
        # strict=False: lets an older checkpoint (saved before ThinkingBlock
        # existed, or with a different num_think_steps setup) load into the
        # current model class. Missing keys (e.g. think_block.* the first
        # time you turn thinking on) come up freshly-initialized; unexpected
        # keys (e.g. loading a "thinking" checkpoint back into a version of
        # this file that removed it) are just dropped. Both are logged so a
        # real architecture mismatch doesn't silently pass.
        load_result = model.load_state_dict(state["model_state_dict"], strict=False)
        if load_result.missing_keys:
            print(f"resume_missing_keys={load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"resume_unexpected_keys={load_result.unexpected_keys}")
        if "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except ValueError as exc:
                print(f"resume_optimizer_reset reason={exc}")
            # The saved state carries the old lr; keep the Adam moments but
            # honor --learning-rate for this run.
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        saved_update = int(state.get("update", 0))
        start_update = saved_update + 1
        print(f"resumed_from={resume_checkpoint} saved_update={saved_update} start_update={start_update}")

    if args.entropy_coef_anneal_updates > 0:
        print(
            f"entropy_coef_anneal: {entropy_coef_start:.4f} -> {entropy_coef_end:.4f} "
            f"over {args.entropy_coef_anneal_updates} updates, starting at update={start_update}"
        )
    else:
        print(f"entropy_coef: {entropy_coef_end:.4f} (flat, no annealing)")

    print(f"num_think_steps={args.num_think_steps}")
    print(f"gamma={args.gamma} gae_lambda={args.gae_lambda}")

    if args.num_workers > 1:
        print(f"rollout_workers={args.num_workers}")

    # "spawn": workers must not inherit the parent's initialised GPU context
    # (forking after CUDA/HIP init is unsafe, especially on ROCm).
    mp_context = multiprocessing.get_context("spawn")
    executor_ctx = ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context, initializer=worker_process_init) if args.num_workers > 1 else None

    try:
        if start_update > args.updates:
            print(
                f"nothing_to_do start_update={start_update} is greater than target updates={args.updates}"
            )
            return
        for update in range(start_update, args.updates + 1):
            current_entropy_coef = resolve_entropy_coef(
                update=update,
                start_update=start_update,
                entropy_coef_start=entropy_coef_start,
                entropy_coef_end=entropy_coef_end,
                entropy_coef_anneal_updates=args.entropy_coef_anneal_updates,
            )

            restarts = 0
            while True:
                try:
                    rollout_start = time.perf_counter()
                    if args.num_workers > 1:
                        assert executor_ctx is not None
                        transitions = collect_rollout_parallel(
                            executor=executor_ctx,
                            model=model,
                            rollout_steps=args.rollout_steps,
                            num_workers=args.num_workers,
                            decision_interval_ticks=args.decision_interval,
                            max_ticks=args.max_ticks,
                            decks_path=args.decks_path,
                            mirror_match=args.mirror_match,
                            quiet_engine=args.quiet_engine,
                            seed=args.seed + update * 100_003,
                            worker_retries=args.worker_retries,
                            num_think_steps=args.num_think_steps,
                            low_elixir_penalty=args.low_elixir_penalty,
                        )
                    else:
                        rollout_model.load_state_dict(model.state_dict())
                        transitions = collect_rollout(
                            env=env,
                            model=rollout_model,
                            device=rollout_device,
                            rollout_steps=args.rollout_steps,
                            quiet_engine=args.quiet_engine,
                        )
                    rollout_elapsed = time.perf_counter() - rollout_start
                    break
                except BrokenProcessPool as exc:
                    restarts += 1
                    print(f"pool_restart update={update:04d} attempt={restarts} error={exc}")
                    if executor_ctx is not None:
                        executor_ctx.shutdown(wait=False, cancel_futures=True)
                    if restarts > args.max_update_restarts:
                        raise RuntimeError(
                            f"Exceeded max_update_restarts={args.max_update_restarts} at update={update}"
                        ) from exc
                    executor_ctx = ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context, initializer=worker_process_init)
                except Exception as exc:
                    restarts += 1
                    print(
                        f"rollout_restart update={update:04d} attempt={restarts} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    if restarts > args.max_update_restarts:
                        raise RuntimeError(
                            f"Exceeded max_update_restarts={args.max_update_restarts} at update={update}"
                        ) from exc

            update_start = time.perf_counter()
            advantages, returns = compute_gae(
                transitions=transitions,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            stats = ppo_update(
                model=model,
                optimizer=optimizer,
                transitions=transitions,
                advantages=advantages,
                returns=returns,
                clip_ratio=args.clip_ratio,
                value_coef=args.value_coef,
                entropy_coef=current_entropy_coef,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                device=device,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            update_elapsed = time.perf_counter() - update_start

            mean_reward = float(np.mean([t.reward for t in transitions]))
            reward_std = float(np.std([t.reward for t in transitions]))
            decisions_per_sec = (len(transitions) / 2.0) / max(1e-6, rollout_elapsed)
            effective_decisions_per_sec = (len(transitions) / 2.0) / max(
                1e-6, rollout_elapsed + update_elapsed
            )
            approx_games_per_min = (
                decisions_per_sec / (args.max_ticks / args.decision_interval) * 60.0
            )
            effective_games_per_min = (
                effective_decisions_per_sec / (args.max_ticks / args.decision_interval) * 60.0
            )
            print(
                f"update={update:04d} "
                f"mean_reward={mean_reward:+.4f} "
                f"reward_std={reward_std:.4f} "
                f"entropy_coef={current_entropy_coef:.4f} "
                f"loss={stats['loss']:.4f} "
                f"policy={stats['policy_loss']:.4f} "
                f"value={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} "
                f"rollout_s={rollout_elapsed:.2f} "
                f"update_s={update_elapsed:.2f} "
                f"dps={decisions_per_sec:.1f} "
                f"gpm~={approx_games_per_min:.1f} "
                f"eff_dps={effective_decisions_per_sec:.1f} "
                f"eff_gpm~={effective_games_per_min:.1f}"
            )

            if update % args.save_every == 0 or update == args.updates:
                ckpt_path = checkpoint_dir / f"policy_update_{update:04d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "update": update,
                        "spec_version": 2,  # observation/action view frame (clasher/rl/common.py)
                        "board_channels": 3,
                        "board_height": 32,
                        "board_width": 18,
                        "n_max": 40,
                        "entity_features": 6,
                        "global_value_count": 13,
                        "num_hand_slots": 4,
                        "num_actions": NUM_ACTIONS,
                        "cards_count": CARDS_COUNT,
                        "num_think_steps": args.num_think_steps,
                    },
                    ckpt_path,
                )
                print(f"saved_checkpoint={ckpt_path}")
    finally:
        if executor_ctx is not None:
            executor_ctx.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
