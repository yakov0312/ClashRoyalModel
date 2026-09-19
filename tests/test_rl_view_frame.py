from collections import deque

import numpy as np

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building
from clasher.rl.action_space import DiscreteTileActionSpace
from clasher.rl.common import BOARD_HEIGHT, BOARD_WIDTH, NUM_TILES, world_to_view
from clasher.rl.obs_cv import (
    GLOBAL_ENEMY_CROWNS,
    GLOBAL_ENEMY_LEFT_HP,
    GLOBAL_HAND_START,
    GLOBAL_OWN_CROWNS,
    ObservationBuilder,
)
from clasher.rl.selfplay_env import SelfPlayBattleEnv
from shared.observationSpec import staticUnwalkableMask, towerHpFeature


def _mirror(position: Position) -> Position:
    """Same spot as seen from the other player's seat (180-degree rotation)."""
    return Position(BOARD_WIDTH - position.x, BOARD_HEIGHT - position.y)


def _hand(battle: BattleState, player_id: int, cards) -> None:
    player = battle.players[player_id]
    player.elixir = 7.0
    player.hand = list(cards)
    player.deck = list(cards)
    player.cycle_queue = deque()


def test_mirrored_situations_give_identical_observations():
    battle = BattleState()
    cards = ["Knight", "HogRider", "Musketeer", "Fireball"]
    _hand(battle, 0, cards)
    _hand(battle, 1, cards)
    for name, pos in (("Knight", Position(3.5, 10.5)), ("Musketeer", Position(12.2, 5.7)), ("Giant", Position(15.5, 20.5))):
        stats = battle.card_loader.get_card(name)
        battle._spawn_troop(pos, 0, stats)
        battle._spawn_troop(_mirror(pos), 1, stats)
    builder = ObservationBuilder()
    obs0 = builder.build(battle, 0)
    obs1 = builder.build(battle, 1)
    assert np.array_equal(obs0.board, obs1.board)
    assert np.array_equal(obs0.entities, obs1.entities)
    assert np.array_equal(obs0.global_features, obs1.global_features)


def test_view_frame_puts_own_side_at_the_bottom():
    battle = BattleState()
    builder = ObservationBuilder()
    for player_id in (0, 1):
        own_king = battle.arena.BLUE_KING_TOWER if player_id == 0 else battle.arena.RED_KING_TOWER
        vx, vy = world_to_view(own_king.x, own_king.y, player_id)
        assert (vx, vy) == (9.0, 29.0)
    terrain = builder.build(battle, 0).board[0]
    assert terrain[27:31, 7:11].all()   # own King, rows 27-30
    assert terrain[1:5, 7:11].all()     # enemy King, rows 1-4


def test_terrain_matches_simulator_arena():
    """The shared layout (used by perception too) must equal the real sim terrain."""
    battle = BattleState()
    arena = battle.arena
    expected = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for wy in range(BOARD_HEIGHT):
        for wx in range(BOARD_WIDTH):
            center = Position(wx + 0.5, wy + 0.5)
            if arena.is_blocked_tile(wx, wy) or not arena.is_walkable(center) or arena.is_tower_tile(center, battle):
                expected[BOARD_HEIGHT - 1 - wy, wx] = 1.0  # player 0 view
    assert np.array_equal(staticUnwalkableMask(), expected)
    assert np.array_equal(ObservationBuilder().build(battle, 0).board[0], expected)


def test_action_masks_are_identical_for_both_seats():
    battle = BattleState()
    cards = ["Knight", "Cannon", "Fireball", "Log"]
    _hand(battle, 0, cards)
    _hand(battle, 1, cards)
    space = DiscreteTileActionSpace()
    assert np.array_equal(space.legal_action_mask(battle, 0), space.legal_action_mask(battle, 1))
    # Own back row (view row 31) behind the King is playable, the enemy half is not.
    assert space.legal_action_mask(battle, 0)[0 * NUM_TILES + 31 * BOARD_WIDTH + 9]
    assert not space.legal_action_mask(battle, 0)[0 * NUM_TILES + 5 * BOARD_WIDTH + 9]


def test_tower_hp_and_crowns_are_observed():
    battle = BattleState()
    red_left = next(e for e in battle.entities.values()
                    if isinstance(e, Building) and e.player_id == 1 and e.position.x == 3.5 and e.card_stats.name == "PrincessTower")
    red_left.take_damage(red_left.hitpoints)
    battle.step()
    assert battle.crowns(0) == 1 and battle.crowns(1) == 0
    builder = ObservationBuilder()
    blue_view = builder.build(battle, 0).global_features
    red_view = builder.build(battle, 1).global_features
    assert blue_view[GLOBAL_OWN_CROWNS] == 1 / 3 and blue_view[GLOBAL_ENEMY_CROWNS] == 0
    assert red_view[GLOBAL_ENEMY_CROWNS] == 1 / 3
    # Red's world-left tower is on blue's screen-left: its HP feature is 0 now.
    assert blue_view[GLOBAL_ENEMY_LEFT_HP] == 0.0
    assert blue_view[GLOBAL_ENEMY_LEFT_HP + 1] == towerHpFeature(3052)
    assert blue_view[GLOBAL_HAND_START] > 0


def test_player_with_more_crowns_wins_at_time_up():
    battle = BattleState()
    red_left = next(e for e in battle.entities.values()
                    if isinstance(e, Building) and e.player_id == 1 and e.card_stats.name == "PrincessTower")
    red_left.take_damage(red_left.hitpoints)
    battle.time = battle.sudden_death_start_time
    battle.step()
    assert battle.game_over and battle.winner == 0


def test_rewards_favour_the_player_who_destroys_a_tower():
    env = SelfPlayBattleEnv(seed=0)
    env.reset()
    red_left = next(e for e in env.battle.entities.values()
                    if isinstance(e, Building) and e.player_id == 1 and e.card_stats.name == "PrincessTower")
    red_left.take_damage(red_left.hitpoints)
    no_op = env.action_space.no_op_action
    rewards, _, _ = env.step({0: no_op, 1: no_op})
    assert rewards[0] > 0.25 and rewards[1] < -0.25
    assert abs(rewards[0] + rewards[1]) < 1e-9
