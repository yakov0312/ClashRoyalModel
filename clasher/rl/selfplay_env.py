from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Optional

import numpy as np

from clasher.battle import BattleState

from .action_space import DiscreteTileActionSpace
from .deck_pool import apply_deck_to_player, load_deck_pool, sample_decks
from .obs_cv import ObservationBuilder
from shared.cardRegistry import CardDatabase
from ..path import DECKS_FILE, CARDS_FILE


@dataclass
class StepInfo:
    action_success: Dict[int, bool]
    ticks_advanced: int


# --- Rewards ------------------------------------------------------------
CROWN_REWARD = 0.25
INVALID_ACTION_PENALTY = 0.01

# Discourage dumping all elixir immediately.
LOW_ELIXIR_THRESHOLD = 2.0
LOW_ELIXIR_PENALTY = 0.01

# Effective elixir shaping.
# Effective elixir = banked elixir + elixir cost of living board units.
ELIXIR_ADVANTAGE_REWARD = 0.001


class SelfPlayBattleEnv:
    """Two-player self-play environment over the battle simulator."""

    def __init__(
        self,
        decision_interval_ticks: int = 8,
        max_ticks: int = 9090,
        decks_path: str | Path = DECKS_FILE,
        seed: Optional[int] = None,
        mirror_match: bool = False,
        canonical_perspective: bool = True,
        low_elixir_penalty: float = LOW_ELIXIR_PENALTY,
    ) -> None:
        self.decision_interval_ticks = decision_interval_ticks
        self.low_elixir_penalty = low_elixir_penalty
        self.max_ticks = max_ticks
        self.mirror_match = mirror_match
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.decks = load_deck_pool(decks_path)
        self.cards = CardDatabase(CARDS_FILE)
        self.obs_builder = ObservationBuilder(
            self.cards,
            canonical_perspective=canonical_perspective,
        )
        self.action_space = DiscreteTileActionSpace(
            canonical_perspective=canonical_perspective
        )

        self.battle: Optional[BattleState] = None
        self._prev_damage_dealt = {0: 0.0, 1: 0.0}
        self._prev_crowns = {0: 0, 1: 0}
        self._prev_effective_elixir_advantage = 0.0

    def _sample_and_apply_decks(self) -> None:
        assert self.battle is not None
        deck0, deck1 = sample_decks(
            self.decks,
            rng=self.rng,
            mirror_match=self.mirror_match,
        )
        apply_deck_to_player(self.battle.players[0], deck0, rng=self.rng)
        apply_deck_to_player(self.battle.players[1], deck1, rng=self.rng)

    def _reset_reward_trackers(self) -> None:
        self._prev_damage_dealt = {0: 0.0, 1: 0.0}
        self._prev_crowns = {0: 0, 1: 0}
        self._prev_effective_elixir_advantage = 0.0

    def reset(self) -> None:
        self.battle = BattleState()
        self._sample_and_apply_decks()
        self._reset_reward_trackers()

    def get_observation(self, player_id: int):
        assert self.battle is not None
        return self.obs_builder.build(self.battle, player_id)

    def get_action_mask(self, player_id: int) -> np.ndarray:
        assert self.battle is not None
        return self.action_space.legal_action_mask(self.battle, player_id)

    def _get_board_elixir(self, player_id: int) -> float:
        assert self.battle is not None

        total = 0.0

        for entity in self.battle.entities:
            if getattr(entity, "owner", getattr(entity, "player_id", None)) != player_id:
                continue

            if getattr(entity, "dead", False):
                continue

            card_id = getattr(entity, "card_id", None)
            if card_id is None:
                continue

            card = self.cards.get_card(card_id)
            if card is None:
                continue

            cost = getattr(card, "elixir", getattr(card, "elixir_cost", 0))
            total += float(cost)

        return total

    def _get_effective_elixir(self, player_id: int) -> float:
        assert self.battle is not None

        banked_elixir = float(self.battle.players[player_id].elixir)
        board_elixir = self._get_board_elixir(player_id)

        return banked_elixir + board_elixir

    def _get_elixir_advantage(self, player_id: int) -> float:
        return (
            self._get_effective_elixir(player_id)
            - self._get_effective_elixir(1 - player_id)
        )

    def _compute_rewards(self) -> Dict[int, float]:
        assert self.battle is not None

        damage_fraction = {}

        for player_id in (0, 1):
            damage_now = self.battle._tower_damage_dealt_by_player(player_id)
            enemy_start_hp = float(
                self.battle._starting_total_tower_hp.get(
                    1 - player_id,
                    1.0,
                )
            )

            damage_fraction[player_id] = (
                damage_now - self._prev_damage_dealt[player_id]
            ) / max(1.0, enemy_start_hp)

            self._prev_damage_dealt[player_id] = damage_now

        crowns_gained = {}

        for player_id in (0, 1):
            crowns_now = self.battle.crowns(player_id)
            crowns_gained[player_id] = (
                crowns_now - self._prev_crowns[player_id]
            )
            self._prev_crowns[player_id] = crowns_now

        current_elixir_advantage = self._get_elixir_advantage(0)
        elixir_advantage_delta = (
            current_elixir_advantage
            - self._prev_effective_elixir_advantage
        )
        self._prev_effective_elixir_advantage = current_elixir_advantage

        rewards = {}

        for player_id in (0, 1):
            opponent_id = 1 - player_id

            rewards[player_id] = (
                damage_fraction[player_id]
                - damage_fraction[opponent_id]
                + CROWN_REWARD
                * (
                    crowns_gained[player_id]
                    - crowns_gained[opponent_id]
                )
            )

            if player_id == 0:
                rewards[player_id] += (
                    ELIXIR_ADVANTAGE_REWARD
                    * elixir_advantage_delta
                )
            else:
                rewards[player_id] -= (
                    ELIXIR_ADVANTAGE_REWARD
                    * elixir_advantage_delta
                )

            elixir = float(self.battle.players[player_id].elixir)

            if elixir < LOW_ELIXIR_THRESHOLD:
                deficit = (
                    LOW_ELIXIR_THRESHOLD - elixir
                ) / LOW_ELIXIR_THRESHOLD

                rewards[player_id] -= (
                    self.low_elixir_penalty * deficit
                )

        return rewards

    def step(
        self,
        actions: Dict[int, int],
    ) -> tuple[Dict[int, float], bool, StepInfo]:
        assert self.battle is not None

        action_success: Dict[int, bool] = {}
        order = [0, 1]
        self.rng.shuffle(order)

        for player_id in order:
            action_id = actions.get(
                player_id,
                self.action_space.no_op_action,
            )

            success = self.action_space.apply_action(
                self.battle,
                player_id,
                action_id,
            )

            action_success[player_id] = success

        ticks = 0

        while (
            ticks < self.decision_interval_ticks
            and not self.battle.game_over
            and self.battle.tick < self.max_ticks
        ):
            self.battle.step()
            ticks += 1

        done = (
            self.battle.game_over
            or self.battle.tick >= self.max_ticks
        )

        rewards = self._compute_rewards()

        for player_id in (0, 1):
            attempted = actions.get(
                player_id,
                self.action_space.no_op_action,
            )

            if (
                attempted != self.action_space.no_op_action
                and not action_success.get(player_id, True)
            ):
                rewards[player_id] -= INVALID_ACTION_PENALTY

        if done and self.battle.winner is not None:
            rewards[self.battle.winner] += 1.0
            rewards[1 - self.battle.winner] -= 1.0

        return (
            rewards,
            done,
            StepInfo(
                action_success=action_success,
                ticks_advanced=ticks,
            ),
        )
