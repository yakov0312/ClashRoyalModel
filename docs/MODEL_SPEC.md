# Skynet CR Model — Observation & Action Spec

`SPEC_VERSION = 1`

Status: draft v1 (single-frame, no velocity/history yet).
This model is trained from scratch. It is *inspired by* an existing open-source
simulator's RL setup but does not depend on it and does not reuse its weights
or vocab.

**Extensibility rule:** new fields are always **appended** to the end of
`global_features` (never inserted in the middle, never reordered). This keeps
every existing index valid as the spec grows, so old code and old checkpoints
don't silently break. `entities` and `board` are considered stable/closed for
now — planned future fields (e.g. velocity) go through the same append rule
if/when they're added. Every implementation (tensor builder, model, any other
AI reading this doc) must reference fields by the named constant / table
index below, never a hardcoded magic number, so appends stay cheap.

Bump `SPEC_VERSION` and add a line to the Changelog whenever a field is
added.

Scope note: identity of every on-board unit (troop, building, tower) is
tracked individually via an **entity list**, not via the spatial grid. The
spatial grid (`board`) is intentionally identity-free to avoid a
multi-tile object (e.g. a building spanning 2x2 tiles) being miscounted as
multiple objects when the grid is later pooled/summed.

---

## Constants

| Name | Value | Notes |
|---|---|---|
| `BOARD_WIDTH` | 18 | tile columns (x-axis), standard CR arena |
| `BOARD_HEIGHT` | 32 | tile rows (y-axis) |
| `VOCAB_SIZE` | `V` (TBD) | number of distinct card classes (troops + buildings + towers unified). Placeholder until YOLO class list vs. full card list is finalized. `class_id = 0` is reserved for "empty/pad". |
| `N_MAX` | 40 (tunable) | max simultaneous entities the model can represent in one frame. Must be >= worst-case on-screen unit count (e.g. Skeleton Army + Goblin Gang double-elixir chaos). |
| `NUM_HAND_SLOTS` | 4 | visible hand size |
| `NUM_ACTIONS` | `NUM_HAND_SLOTS * BOARD_WIDTH * BOARD_HEIGHT + 1` | one action per (hand_slot, tile), plus 1 no-op |

---

## Inputs

### 1. `board` — spatial occupancy (identity-free)

- **Shape:** `(3, BOARD_HEIGHT, BOARD_WIDTH)` = `(3, 32, 18)`
- **dtype:** `float32`
- **Layout:** channel-first `(C, H, W)`

| Channel | Name | Values | Notes |
|---|---|---|---|
| 0 | `unwalkable` | `{0.0, 1.0}` | blocked tiles + river (excluding bridge tiles, which are walkable). Static per match, computed once. |
| 1 | `own_occupied` | `{0.0, 1.0}` | 1 on every tile covered by an own unit's true footprint (troops: 1 tile; buildings/towers: their real multi-tile footprint). No identity, no count — just occupancy. |
| 2 | `enemy_occupied` | `{0.0, 1.0}` | same as above, enemy side |

**Why occupancy-only:** a 2x2 building correctly marks 4 tiles as occupied
(that's true — you can't walk there), but it is still exactly **one** entity.
Counting/identity lives only in the entity list below, never in this grid.

---

### 2. `entities` — per-object identity, HP, position (fixed-size, padded)

- **Shape:** `(N_MAX, F)` = `(40, 6)`
- **dtype:** `float32` (class_id and owner are integer-valued but stored as float32 for a uniform tensor; cast to `long` before embedding lookup)
- **Row = one real-world object.** Multiple objects may share a tile (e.g. two
  Larries stacked, or a unit standing on a building's footprint tile) — this
  is fine and expected, since each still gets its own row.
- **Column order (fixed, do not reorder):**

| Index | Field | dtype (logical) | Range | Notes |
|---|---|---|---|---|
| 0 | `class_id` | int | `0..VOCAB_SIZE` | 0 = padding row (ignored). 1..V = card id, shared vocab for troops/buildings/towers. |
| 1 | `owner` | int | `{0, 1}` | 0 = own, 1 = enemy. Meaningless if `present == 0`. |
| 2 | `hp_pct` | float | `[0.0, 1.0]` | current HP / max HP for this specific object |
| 3 | `tile_x` | float | `[0.0, 1.0]` | anchor position (object center), normalized by `BOARD_WIDTH` |
| 4 | `tile_y` | float | `[0.0, 1.0]` | anchor position (object center), normalized by `BOARD_HEIGHT` |
| 5 | `present` | int | `{0, 1}` | 1 = real detection, 0 = padding. Always check this before trusting any other field in the row. |

- **Padding rule:** rows beyond the number of real detected entities are
  zero-filled (`class_id=0, owner=0, hp_pct=0, tile_x=0, tile_y=0, present=0`).
- **Overflow rule:** if detected entity count exceeds `N_MAX`, drop
  lowest-confidence detections first (define this at the YOLO→tensor stage,
  not silently in the model).

### 2b. `entity_mask` — convenience mask, redundant with column 5 above

- **Shape:** `(N_MAX,)`
- **dtype:** `bool`
- **Values:** `True` where `entities[:, 5] == 1` (real), `False` for padding.
- Provided as a separate tensor because attention/set-encoder layers
  typically want a plain boolean mask, not a slice of the feature tensor.

---

### 3. `global_features` — flat, non-spatial match state

- **Shape:** `(G,)` — see field list below for exact `G`
- **dtype:** `float32`
- **Fixed field order:**

| Index | Field | Range | Notes |
|---|---|---|---|
| 0 | `time_norm` | `[0.0, 1.0]` | match time / max match duration |
| 1 | `double_elixir` | `{0.0, 1.0}` | flag |
| 2 | `triple_elixir` | `{0.0, 1.0}` | flag |
| 3 | `overtime` | `{0.0, 1.0}` | flag |
| 4 | `own_elixir_pct` | `[0.0, 1.0]` | own elixir / max elixir |
| 5 | `own_crowns` | `[0.0, 1.0]` | own crowns / 3 |
| 6 | `enemy_crowns` | `[0.0, 1.0]` | enemy crowns / 3 |
| 7 | `own_king_hp_pct` | `[0.0, 1.0]` | |
| 8 | `own_left_hp_pct` | `[0.0, 1.0]` | 0 if tower destroyed |
| 9 | `own_right_hp_pct` | `[0.0, 1.0]` | 0 if tower destroyed |
| 10 | `enemy_king_hp_pct` | `[0.0, 1.0]` | |
| 11 | `enemy_left_hp_pct` | `[0.0, 1.0]` | |
| 12 | `enemy_right_hp_pct` | `[0.0, 1.0]` | |
| 13..13+4*V-1 | `hand_onehot` | `{0.0, 1.0}` | 4 hand slots × one-hot over `VOCAB_SIZE` (same vocab as `entities.class_id`) |

`G_v1 = 13 + NUM_HAND_SLOTS * VOCAB_SIZE`

**Planned append (v2, not yet implemented — do not build tensor_builder
support for this until `EnemyCycleTracker` exists):**

| Index | Field | Range | Notes |
|---|---|---|---|
| `G_v1` .. `G_v1+4*V-1` | `enemy_hand_onehot` | `{0.0, 1.0}` | same layout as `hand_onehot`, but for the inferred enemy hand |
| `G_v1+4*V` | `enemy_hand_known` | `{0.0, 1.0}` | 1 once the enemy's 8-card deck cycle has been fully observed and is being tracked with certainty; 0 (with `enemy_hand_onehot` all-zero) while still unresolved, e.g. early in the match. Model must be able to tell "unknown" apart from "confidently empty" — never fake a guess here. |

`G_v2 = G_v1 + NUM_HAND_SLOTS * VOCAB_SIZE + 1`

When this ships: bump `SPEC_VERSION` to 2, log it below, and update `G` to
`G_v2` everywhere `global_features` shape is referenced.

---

## Outputs

| Name | Shape | dtype | Notes |
|---|---|---|---|
| `policy_logits` | `(NUM_ACTIONS,)` | `float32` | raw logits, one per `(hand_slot, tile)` pair + 1 no-op (last index). Must be masked with legal-action mask (`-1e9` on illegal entries) before sampling/softmax — mask is supplied externally, not produced by the model. |
| `value` | scalar | `float32` | state-value estimate (critic head) |

---

## Explicitly out of scope for v1 (flag for v2)

- **No temporal/velocity info.** A single frame cannot distinguish an
  advancing Hog Rider from a retreating one. v2 should either (a) stack the
  last 2-3 frames' `entities`/`board` tensors, or (b) add a `velocity_x`,
  `velocity_y` field to each entity row computed by matching detections
  across consecutive frames.
- **`VOCAB_SIZE` and the exact card list are not finalized** — must be fixed
  before training (YOLO detector's trained class list vs. full card list).
- **`N_MAX` is a guess (40)** — should be validated against real worst-case
  on-screen unit counts before locking in.

---

## Changelog

- **v1** — initial spec: `board` (3ch occupancy), `entities` (N_MAX x 6,
  identity/hp/position), `entity_mask`, `global_features` (match state +
  own hand). `policy_logits` + `value` outputs.
- **v2 (planned)** — append `enemy_hand_onehot` + `enemy_hand_known` to
  `global_features`. Requires `EnemyCycleTracker` (deck-cycle inference from
  observed enemy card plays) to actually populate real data.
