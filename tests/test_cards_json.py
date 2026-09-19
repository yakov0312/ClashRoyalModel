import json
import re

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building
from clasher.path import CARDS_FILE
from clasher.spells import SPELL_REGISTRY
from shared.cardRegistry import SPEED_TIER_TILES_PER_MIN, CardDatabase

CAMEL_CASE_KEY = re.compile(r"^[a-z][a-zA-Z0-9]*$")


def _raw_cards():
    with open(CARDS_FILE) as f:
        return json.load(f)


def test_card_ids_are_unique_and_contiguous():
    ids = [entry["id"] for entry in _raw_cards().values()]
    assert all(isinstance(card_id, int) for card_id in ids)
    assert sorted(ids) == list(range(1, len(ids) + 1))


def test_card_keys_are_camel_case_and_entries_are_typed():
    for name, entry in _raw_cards().items():
        bad = [key for key in entry if not CAMEL_CASE_KEY.match(key)]
        assert not bad, f"{name}: non camelCase keys {bad}"
        assert entry.get("type") in {"troop", "air", "building", "spell", "projectile"}, name


def test_simulated_units_use_cards_json_stats():
    db = CardDatabase()
    battle = BattleState()
    checked = 0
    for stats in db.allStats():
        if stats.variant != "base" or stats.is_spell or stats.hitpoints is None or stats.name.endswith("Hidden"):
            continue
        card = battle.card_loader.get_card(stats.name)
        assert card is not None, stats.name
        entity = battle._spawn_entity(Building, Position(9.0, 10.0), 0, card) if stats.is_building \
            else battle._spawn_single_troop(Position(9.0, 10.0), 0, card)
        assert entity.max_hitpoints == stats.hitpoints, stats.name
        if stats.damage is not None and stats.number("maxDamage") is None:
            assert entity.damage == stats.damage, stats.name
        if stats.hit_speed:
            assert abs(entity.get_attack_interval_seconds() - stats.hit_speed) < 1e-6, stats.name
        if stats.range:
            assert abs(entity.range - stats.range) < 1e-6, stats.name
        if stats.speed_tier and not stats.is_building:
            assert entity.speed == SPEED_TIER_TILES_PER_MIN[stats.speed_tier], stats.name
        checked += 1
    assert checked >= 100


def test_every_playable_card_deploys():
    db = CardDatabase()
    for stats in db.allStats():
        if stats.variant != "base" or not stats.is_playable or stats.name == "Mirror":
            continue
        battle = BattleState()
        player = battle.players[0]
        player.elixir = 10
        player.hand = [stats.name]
        player.deck = [stats.name]
        player.cycle_queue.clear()
        spell = SPELL_REGISTRY.get(stats.name)
        # Most spells go anywhere; rolling ones (Log, Barbarian Barrel) follow troop rules.
        free_spell = spell is not None and type(spell).__name__ != "RollingProjectileSpell"
        position = Position(9.5, 16.5) if free_spell else Position(9.5, 10.5)
        assert battle.deploy_card(0, stats.name, position), stats.name
        for _ in range(30):
            battle.step()
