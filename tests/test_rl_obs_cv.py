import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.rl.obs_cv import ObservationBuilder


def _prepare_hand(battle: BattleState, player_id: int, cards: list[str]) -> None:
    player = battle.players[player_id]
    player.elixir = 10.0
    player.hand = list(cards[:4])
    player.deck = list(cards[:8] if len(cards) >= 8 else cards + cards)
    player.cycle_queue = deque(player.deck[4:])


def test_cv_observation_shapes_and_visible_hud_only():
    battle = BattleState()
    _prepare_hand(
        battle,
        0,
        ["Knight", "Archers", "Giant", "Minions", "Musketeer", "Fireball", "Zap", "Cannon"],
    )

    builder = ObservationBuilder(card_vocab=["Knight", "Archers", "Giant", "Minions", "Fireball"])
    obs = builder.build(battle, player_id=0)

    assert obs.board.shape == (3, 32, 18)
    assert obs.entities.shape == (40, 6)
    assert obs.entity_mask.shape == (40,)
    assert obs.global_features.shape == (18,)
    assert np.isfinite(obs.board).all()
    assert np.isfinite(obs.entities).all()
    assert np.isfinite(obs.global_features).all()

    own_elixir_norm = battle.players[0].elixir / battle.players[0].max_elixir
    enemy_elixir_norm = battle.players[1].elixir / battle.players[1].max_elixir

    assert np.isclose(obs.global_features[4], own_elixir_norm)
    assert not np.isclose(obs.global_features[4], enemy_elixir_norm)


def test_enemy_hand_changes_do_not_change_cv_observation():
    battle = BattleState()
    _prepare_hand(
        battle,
        0,
        ["Knight", "Archers", "Giant", "Minions", "Musketeer", "Fireball", "Zap", "Cannon"],
    )
    _prepare_hand(
        battle,
        1,
        ["HogRider", "Earthquake", "TheLog", "Firecracker", "Skeletons", "Cannon", "IceSpirit", "Knight"],
    )

    builder = ObservationBuilder(
        card_vocab=[
            "Knight",
            "Archers",
            "Giant",
            "Minions",
            "Musketeer",
            "Fireball",
            "Zap",
            "Cannon",
            "HogRider",
            "Earthquake",
            "TheLog",
            "Firecracker",
            "Skeletons",
            "IceSpirit",
        ]
    )

    obs_before = builder.build(battle, player_id=0)

    battle.players[1].hand = ["Golem", "Lightning", "NightWitch", "Tornado"]
    battle.players[1].cycle_queue = deque(["BabyDragon", "BarbarianBarrel", "Lumberjack", "Bowler"])
    obs_after = builder.build(battle, player_id=0)

    np.testing.assert_allclose(obs_before.board, obs_after.board)
    np.testing.assert_allclose(obs_before.entities, obs_after.entities)
    np.testing.assert_allclose(obs_before.global_features, obs_after.global_features)


def test_entity_planes_reflect_spawned_units():
    battle = BattleState()
    _prepare_hand(
        battle,
        0,
        ["Knight", "Archers", "Giant", "Minions", "Musketeer", "Fireball", "Zap", "Cannon"],
    )
    _prepare_hand(
        battle,
        1,
        ["Knight", "Archers", "Giant", "Minions", "Musketeer", "Fireball", "Zap", "Cannon"],
    )

    assert battle.deploy_card(0, "Knight", Position(9.0, 10.0))
    assert battle.deploy_card(1, "Knight", Position(9.0, 22.0))

    builder = ObservationBuilder(card_vocab=["Knight", "Archers", "Giant", "Minions", "Fireball"])
    obs = builder.build(battle, player_id=0)

    assert float(obs.board[1].sum()) > 0.0
    assert float(obs.board[2].sum()) > 0.0
    assert int(obs.entity_mask.sum()) >= 2
