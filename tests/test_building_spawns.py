from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building, TimedExplosive, Troop


def test_tombstone_skeletons_walk_out_and_fight():
    battle = BattleState()
    tomb = battle._spawn_entity(Building, Position(9.5, 9.5), 0, battle.card_loader.get_card("Tombstone"))
    battle._spawn_troop(Position(9.5, 17.5), 1, battle.card_loader.get_card("Knight"))
    radius = tomb.card_stats.collision_radius
    seen_outside = False
    for _ in range(260):
        battle.step()
        skeletons = [e for e in battle.entities.values()
                     if isinstance(e, Troop) and e.player_id == 0 and e.card_stats.name == "Larry"]
        if any(s.position.distance_to(tomb.position) > radius + 1.0 for s in skeletons):
            seen_outside = True
            break
    assert tomb.is_alive
    assert seen_outside, "Tombstone skeletons never left the tombstone"


def test_unit_inside_a_building_walks_out():
    battle = BattleState()
    cannon = battle._spawn_entity(Building, Position(9.5, 9.5), 0, battle.card_loader.get_card("Cannon"))
    knight = battle._spawn_single_troop(Position(9.6, 9.6), 0, battle.card_loader.get_card("Knight"))
    for _ in range(90):
        battle.step()
    assert knight.position.distance_to(cannon.position) > cannon.card_stats.collision_radius


def test_giant_skeleton_and_balloon_leave_timed_bombs():
    battle = BattleState()
    for name, x in (("GiantSkeleton", 5.5), ("Balloon", 12.5)):
        unit = battle._spawn_single_troop(Position(x, 10.5), 0, battle.card_loader.get_card(name))
        unit.take_damage(unit.hitpoints)
    battle.step()
    bombs = [e for e in battle.entities.values() if isinstance(e, TimedExplosive)]
    assert len(bombs) == 2
    assert all(b.explosion_timer > 1.0 for b in bombs)
