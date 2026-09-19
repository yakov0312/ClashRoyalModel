from collections import deque

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building, Troop
from clasher.towers import KING_TOWER, PRINCESS_TOWER, load_tower_stats


def _towers(battle: BattleState, name: str):
    return [e for e in battle.entities.values() if isinstance(e, Building) and e.card_stats.name == name]


def _king(battle: BattleState, player_id: int) -> Building:
    return next(t for t in _towers(battle, KING_TOWER) if t.player_id == player_id)


def _hand(battle: BattleState, player_id: int, *cards: str) -> None:
    player = battle.players[player_id]
    player.elixir = 20.0
    player.hand = list(cards)
    player.deck = list(cards)
    player.cycle_queue = deque()


def test_crown_towers_use_towers_json_stats():
    battle = BattleState()
    data = load_tower_stats()
    princesses = _towers(battle, PRINCESS_TOWER)
    kings = _towers(battle, KING_TOWER)
    assert len(princesses) == 4 and len(kings) == 2
    for name, towers in ((PRINCESS_TOWER, princesses), (KING_TOWER, kings)):
        for tower in towers:
            assert tower.max_hitpoints == data[name]["hp"]
            assert tower.damage == data[name]["damage"]
            assert tower.range == data[name]["range"]
            assert abs(tower.get_attack_interval_seconds() - data[name]["hitSpeed"]) < 1e-9
    assert battle.players[0].left_tower_hp == data[PRINCESS_TOWER]["hp"]
    assert battle.players[0].king_tower_hp == data[KING_TOWER]["hp"]


def test_princess_tower_fires_every_hit_speed():
    battle = BattleState()
    golem = battle.card_loader.get_card("Golem")
    battle._spawn_troop(Position(3.5, 12.5), 1, golem)
    target = next(e for e in battle.entities.values() if isinstance(e, Troop) and e.card_stats.name == "Golem")
    target.deploy_delay_remaining = 99.0  # stand still inside tower range
    hits = []
    last = target.hitpoints
    for _ in range(200):
        battle.step()
        if target.hitpoints < last:
            hits.append(battle.time)
            last = target.hitpoints
    intervals = [b - a for a, b in zip(hits, hits[1:])]
    assert len(intervals) >= 5
    assert all(abs(i - 0.8) < 0.05 for i in intervals)


def test_king_tower_leaves_back_row_free():
    battle = BattleState()
    arena = battle.arena
    # King footprint is rows 1-4 (y 1.0..5.0), so row 0 behind it is open.
    assert arena.is_tower_tile(Position(9.5, 1.5), battle)
    assert arena.is_tower_tile(Position(9.5, 4.5), battle)
    assert not arena.is_tower_tile(Position(9.5, 0.5), battle)
    assert not arena.is_tower_tile(Position(9.5, 5.5), battle)
    assert not arena.is_tower_tile(Position(9.5, 31.5), battle)


def test_can_deploy_behind_own_king_and_walk_out():
    battle = BattleState()
    _hand(battle, 0, "Giant")
    assert battle.deploy_card(0, "Giant", Position(9.5, 0.5))
    giant = next(e for e in battle.entities.values() if isinstance(e, Troop) and e.card_stats.name == "Giant")
    for _ in range(300):
        battle.step()
    assert giant.position.y > 5.0  # walked around the King Tower

    _hand(battle, 1, "Knight")
    assert battle.deploy_card(1, "Knight", Position(8.5, 31.5))
    # The blocked back corners stay blocked.
    _hand(battle, 1, "Knight")
    assert not battle.deploy_card(1, "Knight", Position(2.5, 31.5))


def test_king_tower_activation_rules():
    # 1) Direct damage wakes the King.
    battle = BattleState()
    king = _king(battle, 1)
    assert king._tower_active is False
    king.take_damage(1)
    assert king._tower_active is True

    # 2) Losing a Princess Tower wakes the King.
    battle = BattleState()
    princess = next(t for t in _towers(battle, PRINCESS_TOWER) if t.player_id == 1)
    princess.take_damage(princess.hitpoints)
    battle.step()
    assert _king(battle, 1)._tower_active is True

    # Nothing else wakes it: an enemy troop near the King, or dragged toward
    # it by Tornado, leaves it asleep.
    battle = BattleState()
    knight = battle.card_loader.get_card("Knight")
    battle._spawn_troop(Position(9.5, 8.5), 1, knight)
    _hand(battle, 0, "Tornado")
    assert battle.deploy_card(0, "Tornado", Position(9.5, 9.0))
    for _ in range(90):
        battle.step()
    assert _king(battle, 0)._tower_active is False
