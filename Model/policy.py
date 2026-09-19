import os

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from Model.groups import CARD_GROUPS, CARD_GROUP_COUNT

HAND_SIZE = 4
CARD_SLOTS = HAND_SIZE + 1

GLOBAL_VALUE_COUNT = 13

BOARD_CHANNELS = 3
BOARD_WIDTH = 18
BOARD_HEIGHT = 32

CARDS_COUNT = 236

N_MAX = 40
ENTITY_FEATURES = 6

CARD_EMBED_DIM = 32
GROUP_EMBED_DIM = 16
FEATURE_DIM = 128
SHARED_DIM = 256

NUM_ACTIONS = HAND_SIZE * BOARD_WIDTH * BOARD_HEIGHT + 1

DEFAULT_NUM_THINK_STEPS = 0

# Fused attention kernels (CUDA flash / mem-efficient) with MATH as the
# fallback. On ROCm the fused kernels are opt-in (experimental), so only use
# them when the user has enabled them.
if getattr(torch.version, "hip", None) and not os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"):
    _SDPA_BACKENDS = [SDPBackend.MATH]
else:
    _SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class ThinkingBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ClashRLModel(nn.Module):
    def __init__(self, num_think_steps: int = DEFAULT_NUM_THINK_STEPS):
        super().__init__()
        self.num_think_steps = num_think_steps

        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_VALUE_COUNT, FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
            nn.ReLU(),
        )

        self.class_embedding = nn.Embedding(
            CARDS_COUNT + 1,
            CARD_EMBED_DIM,
        )

        self.group_embedding = nn.Embedding(
            CARD_GROUP_COUNT,
            GROUP_EMBED_DIM,
        )

        self.group_projection = nn.Linear(
            CARD_EMBED_DIM + GROUP_EMBED_DIM,
            CARD_EMBED_DIM,
        )

        groupMask = torch.zeros(
            CARDS_COUNT + 1,
            CARD_GROUP_COUNT,
            dtype=torch.float32,
        )

        for cardId, groups in CARD_GROUPS.items():
            if not 0 <= cardId <= CARDS_COUNT:
                continue

            for groupId in groups:
                if 0 <= groupId < CARD_GROUP_COUNT:
                    groupMask[cardId, groupId] = 1.0

        self.register_buffer(
            "card_group_mask",
            groupMask,
            persistent=False,
        )

        self.card_encoder = nn.Sequential(
            nn.Linear(CARD_SLOTS * CARD_EMBED_DIM, FEATURE_DIM),
            nn.ReLU(),
        )

        self.global_combiner = nn.Sequential(
            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),
            nn.ReLU(),
        )

        self.board_encoder = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                64 * (BOARD_HEIGHT // 4) * (BOARD_WIDTH // 4),
                FEATURE_DIM,
            ),
            nn.ReLU(),
        )

        self.entity_numeric_encoder = nn.Sequential(
            nn.Linear(ENTITY_FEATURES - 2, CARD_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(CARD_EMBED_DIM, CARD_EMBED_DIM),
            nn.ReLU(),
        )

        self.entity_projection = nn.Sequential(
            nn.Linear(CARD_EMBED_DIM * 2, FEATURE_DIM),
            nn.ReLU(),
        )

        self.entity_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=FEATURE_DIM,
                nhead=4,
                # No dropout: rollouts run in eval mode and PPO recomputes
                # log-probs in train mode; dropout would make the PPO ratio
                # noisy from the first step. (No parameters change, so older
                # checkpoints still load.)
                dropout=0.0,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.shared = nn.Sequential(
            nn.Linear(FEATURE_DIM * 3, SHARED_DIM),
            nn.ReLU(),
            nn.Linear(SHARED_DIM, SHARED_DIM),
            nn.ReLU(),
        )

        self.recurrent = nn.GRU(
            SHARED_DIM,
            SHARED_DIM,
            num_layers=1,
            batch_first=True,
        )

        self.think_block = ThinkingBlock(SHARED_DIM)

        self.policy_head = nn.Linear(
            SHARED_DIM,
            NUM_ACTIONS,
        )

        self.value_head = nn.Linear(
            SHARED_DIM,
            1,
        )

    def init_hidden_state(self, batch_size, device):
        return torch.zeros(
            1,
            batch_size,
            SHARED_DIM,
            device=device,
        )

    def _think(self, combined, num_think_steps=None):
        steps = (
            self.num_think_steps
            if num_think_steps is None
            else num_think_steps
        )

        for _ in range(steps):
            combined = self.think_block(combined)

        return combined

    def _get_class_features(self, class_ids):
        class_features = self.class_embedding(class_ids)

        group_mask = self.card_group_mask[class_ids]
        group_features = group_mask @ self.group_embedding.weight

        group_count = group_mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

        group_features = group_features / group_count

        return self.group_projection(
            torch.cat(
                [
                    class_features,
                    group_features,
                ],
                dim=-1,
            )
        )

    def _encode(
        self,
        global_features,
        board,
        entities,
        entity_mask,
    ):
        no_entities = ~entity_mask.any(dim=1)

        if no_entities.any():
            entities = entities.clone()
            entity_mask = entity_mask.clone()
            entities[no_entities, 0, :] = 0
            entity_mask[no_entities, 0] = True

        global_values = global_features[:, :GLOBAL_VALUE_COUNT]
        card_ids = global_features[:, GLOBAL_VALUE_COUNT:].long()

        valid_card_ids = (
            (card_ids >= 0) &
            (card_ids <= CARDS_COUNT)
        )

        card_ids = torch.where(
            valid_card_ids,
            card_ids,
            torch.zeros_like(card_ids),
        )

        global_features = self.global_encoder(
            global_values
        )

        card_features = self._get_class_features(card_ids)

        card_features = card_features.flatten(
            start_dim=1
        )

        card_features = self.card_encoder(
            card_features
        )

        global_features = torch.cat(
            [
                global_features,
                card_features,
            ],
            dim=-1,
        )

        global_features = self.global_combiner(
            global_features
        )

        board_features = self.board_encoder(board)

        class_ids = entities[:, :, 0].long()

        valid_class_ids = (
            (class_ids >= 0) &
            (class_ids <= CARDS_COUNT)
        )

        class_ids = torch.where(
            valid_class_ids,
            class_ids,
            torch.zeros_like(class_ids),
        )

        class_features = self._get_class_features(class_ids)

        numeric_features = entities[:, :, 1:5]

        numeric_features = self.entity_numeric_encoder(
            numeric_features
        )

        entity_features = torch.cat(
            [
                class_features,
                numeric_features,
            ],
            dim=-1,
        )

        entity_features = self.entity_projection(
            entity_features
        )

        entity_features = entity_features.float()

        with sdpa_kernel(_SDPA_BACKENDS):
            entity_features = self.entity_attention(
                entity_features,
                src_key_padding_mask=~entity_mask,
            )

        mask = entity_mask.unsqueeze(-1)
        entity_features = entity_features * mask

        entity_sum = entity_features.sum(dim=1)
        entity_count = mask.sum(dim=1)

        entity_features = (
            entity_sum /
            entity_count.clamp(min=1)
        )

        combined = torch.cat(
            [
                global_features,
                board_features,
                entity_features,
            ],
            dim=-1,
        )

        combined = self.shared(combined)

        return combined

    def forward(
        self,
        global_features,
        board,
        entities,
        entity_mask,
        hidden_state=None,
        num_think_steps=None,
    ):
        combined = self._encode(
            global_features,
            board,
            entities,
            entity_mask,
        )

        if hidden_state is None:
            hidden_state = self.init_hidden_state(
                combined.size(0),
                combined.device,
            )

        combined, new_hidden_state = self.recurrent(
            combined.unsqueeze(1),
            hidden_state,
        )

        combined = combined.squeeze(1)

        combined = self._think(
            combined,
            num_think_steps,
        )

        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)

        return policy_logits, value, new_hidden_state

    def forward_sequence(
        self,
        global_features,
        board,
        entities,
        entity_mask,
        done_mask,
        hidden_state=None,
        num_think_steps=None,
    ):
        B, T = (
            entity_mask.shape[0],
            entity_mask.shape[1],
        )

        def flat(x):
            return x.reshape(
                B * T,
                *x.shape[2:],
            )

        combined = self._encode(
            flat(global_features),
            flat(board),
            flat(entities),
            flat(entity_mask),
        )

        combined = combined.view(
            B,
            T,
            SHARED_DIM,
        )

        if hidden_state is None:
            hidden_state = self.init_hidden_state(
                B,
                combined.device,
            )

        # Run the GRU over whole segments between episode ends: the hidden
        # state of rows whose episode ended at step t is zeroed after t, which
        # is exactly what a step-by-step loop does, with far fewer launches.
        reset_steps = done_mask.any(dim=0).nonzero().flatten().tolist()
        boundaries = sorted(set(t for t in reset_steps if t < T - 1)) + [T - 1]

        outputs = []
        start = 0
        for end in boundaries:
            segment_out, hidden_state = self.recurrent(
                combined[:, start:end + 1],
                hidden_state,
            )
            outputs.append(segment_out)

            if end < T - 1:
                reset = done_mask[:, end].view(1, B, 1)
                hidden_state = hidden_state * (~reset)

            start = end + 1

        combined = torch.cat(
            outputs,
            dim=1,
        )

        combined = self._think(
            combined,
            num_think_steps,
        )

        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)

        return policy_logits, value
