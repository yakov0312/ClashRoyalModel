"""Build simulator card entries with Cards.json as the source of numeric stats.

``CardBehaviors.json`` (keyed by Cards.json names, generated once from an old
level-1 data dump by ``scripts/data/build_card_behaviors.py``) is only used as
a *behaviour template*: projectile speeds/splash radii, collision radii, sight
ranges, target flags, buff percentages, nested death-spawn structure, etc.
Every number that Cards.json provides overwrites the template value, and
entries are flagged ``_prescaled`` so no level multiplier is applied on top
(Cards.json values are already tournament-level).

Cards without a behaviour template are synthesised from Cards.json alone.

Evolutions/Heroes: ``build_variant_entry`` is the extension point. Right now
only ``base`` variants are built; an Evo/Hero builder can be registered in
``VARIANT_BUILDERS`` later and receives the already-built base entry plus the
variant's ``CardStats`` (Cards.json already stores full stat lines for them).
"""

from __future__ import annotations

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..path import CARD_BEHAVIORS_FILE, CARD_SPAWNS_FILE, CARDS_FILE, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.cardRegistry import (  # noqa: E402  (path set up above)
    VARIANT_BASE,
    CardDatabase,
    CardStats,
)

# Fallbacks used ONLY for properties Cards.json does not describe and no
# behaviour template exists for. Tile/ms units as in the behaviour templates.
DEFAULT_SIGHT_RANGE = 5500
DEFAULT_COLLISION_RADIUS = 500
DEFAULT_BUILDING_COLLISION_RADIUS = 1000
DEFAULT_DEPLOY_TIME_MS = 1000
DEFAULT_PROJECTILE_SPEED = 600
RANGED_THRESHOLD_TILES = 2.0  # attack range above which a synthesised unit fires projectiles

CHARACTER_TYPES = {"troop", "air", "building"}


@lru_cache(maxsize=None)
def get_card_database(cards_file: str = str(CARDS_FILE), spawns_file: str = str(CARD_SPAWNS_FILE)) -> CardDatabase:
    return CardDatabase(cards_file, spawns_file if Path(spawns_file).exists() else None)


@lru_cache(maxsize=None)
def load_behaviors(path: str = str(CARD_BEHAVIORS_FILE)) -> Dict[str, dict]:
    """Behaviour templates keyed by Cards.json name (treat as read-only)."""
    with open(path, "r") as f:
        return json.load(f)


def _ms(seconds: Optional[float]) -> Optional[int]:
    return None if seconds is None else int(round(seconds * 1000))


def _milli_tiles(tiles: Optional[float]) -> Optional[int]:
    return None if tiles is None else int(round(tiles * 1000))


# ---------------------------------------------------------------------------
# Stat application
# ---------------------------------------------------------------------------

def first_number(stats: CardStats, *keys: str) -> Optional[float]:
    for key in keys:
        value = stats.number(key)
        if value is not None:
            return value
    return None


def _is_bomb(char: dict) -> bool:
    return isinstance(char, dict) and char.get("deathDamage") is not None and not char.get("hitpoints")


def apply_unit_stats(char: dict, stats: CardStats) -> dict:
    """Overwrite a behaviour-template character dict with Cards.json numbers (in place)."""
    hp = stats.hitpoints
    if hp is not None:
        char["hitpoints"] = hp

    # Decorative main projectile (Princess): the real damaging shot is the
    # "custom first" projectile, which carries the splash radius.
    main_projectile = char.get("projectileData")
    custom_first = char.get("customFirstProjectileData")
    if isinstance(main_projectile, dict) and isinstance(custom_first, dict) and "damage" not in main_projectile:
        char["projectileData"] = custom_first

    damage = stats.damage
    projectile = char.get("projectileData") if isinstance(char.get("projectileData"), dict) else None
    if damage is not None:
        char["damage"] = damage
        if projectile is not None:
            projectile["damage"] = damage
        custom_first = char.get("customFirstProjectileData")
        if isinstance(custom_first, dict):
            custom_first["damage"] = damage

    if stats.hit_speed is not None:
        char["hitSpeed"] = _ms(stats.hit_speed)
    if stats.range is not None:
        char["range"] = _milli_tiles(stats.range)
        # Units never "see" less far than they can shoot.
        if char.get("sightRange") is not None and char["sightRange"] < char["range"]:
            char["sightRange"] = char["range"]
    if stats.speed_tiles_per_min is not None:
        char["speed"] = stats.speed_tiles_per_min
    if stats.lifetime is not None:
        char["lifeTime"] = _ms(stats.lifetime)
    if stats.deploy_time is not None:
        char["deployTime"] = _ms(stats.deploy_time)

    projectile_range = stats.number("projectileRange")
    if projectile_range is not None and projectile is not None:
        projectile["projectileRange"] = _milli_tiles(projectile_range)

    shield = stats.number("shieldHealth")
    if shield is not None:
        char["shieldHitpoints"] = shield

    charge = stats.number("chargeDamage")
    if charge is not None:
        char["damageSpecial"] = charge

    dash_damage = stats.number("dashDamage")
    if dash_damage is not None:
        char["dashDamage"] = dash_damage
    dash_range = stats.range_pair("dashRange")
    if dash_range is not None:
        char["dashMinRange"], char["dashMaxRange"] = _milli_tiles(dash_range[0]), _milli_tiles(dash_range[1])

    max_damage = stats.number("maxDamage")
    if max_damage is not None:
        base = damage if damage is not None else char.get("damage")
        old_stage2 = char.get("variableDamage2")
        old_stage3 = char.get("variableDamage3")
        if old_stage2 is not None and old_stage3:
            # Keep the template's stage-2 position between stage 1 and max.
            old_base = float(char.get("_template_damage", old_stage2) or 0)
            span = float(old_stage3) - old_base
            frac = (float(old_stage2) - old_base) / span if span > 0 else 0.5
            char["variableDamage2"] = round(base + (max_damage - base) * frac)
        else:
            char["variableDamage2"] = round((base + max_damage) / 2)
        char["variableDamage3"] = max_damage

    death_damage = stats.death_damage
    death_child = char.get("deathSpawnCharacterData")
    if death_damage is not None:
        if char.get("deathDamage") is not None or not _is_bomb(death_child):
            char["deathDamage"] = death_damage
        else:
            death_child["deathDamage"] = death_damage
            death_child["_prescaled"] = True

    death_radius = stats.number("deathDamageRadius")
    if death_radius is not None:
        target = death_child if _is_bomb(death_child) else char
        target["deathDamageRadius"] = _milli_tiles(death_radius)

    spawn_on_death = stats.number("spawnOnDeath")
    if spawn_on_death is not None:
        if _is_bomb(death_child) and isinstance(death_child.get("deathSpawnCharacterData"), dict):
            death_child["deathSpawnCount"] = int(spawn_on_death)
        else:
            char["deathSpawnCount"] = int(spawn_on_death)

    spawn_speed = stats.number("spawnSpeed")
    if spawn_speed is not None:
        char["spawnPauseTime"] = _ms(spawn_speed)

    # Status durations (stun / freeze / slow) carried by the attack.
    status_duration = first_number(stats, "stunDuration", "slowdownDuration", "duration")
    if status_duration is not None:
        if projectile is not None and isinstance(projectile.get("targetBuffData"), dict):
            projectile["buffTime"] = _ms(status_duration)
        if isinstance(char.get("buffOnDamageData"), dict):
            char["buffOnDamageTime"] = _ms(status_duration)

    char["_prescaled"] = True
    char["_cardsJson"] = stats.name
    return char


def patch_nested_units(node: Any, db: CardDatabase, owner: Optional[str], skip_root: bool = True) -> None:
    """Patch nested sub-unit character dicts (death spawns, periodic spawns...)."""

    def walk(value: Any, is_root: bool) -> None:
        if isinstance(value, dict):
            name = value.get("name")
            if (
                not (is_root and skip_root)
                and isinstance(name, str)
                and name != owner
                and db.contains(name)
                and ("hitpoints" in value or "damage" in value)
            ):
                stats = db.stats(name)
                if stats is not None and stats.type in CHARACTER_TYPES:
                    apply_unit_stats(value, stats)
            for child in value.values():
                walk(child, False)
        elif isinstance(value, list):
            for child in value:
                walk(child, False)

    walk(node, True)


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

def _tid_type(stats: CardStats) -> str:
    if stats.is_spell:
        return "TID_CARD_TYPE_SPELL"
    if stats.is_building:
        return "TID_CARD_TYPE_BUILDING"
    return "TID_CARD_TYPE_CHARACTER"


def synthesize_character(stats: CardStats) -> dict:
    """Character dict for a unit Cards.json describes but no behaviour template covers."""
    is_building = stats.is_building
    rng = stats.range if stats.range is not None else None
    char: Dict[str, Any] = {
        "name": stats.name,
        "rarity": "Common",
        "sightRange": max(DEFAULT_SIGHT_RANGE, _milli_tiles(rng) or 0),
        "deployTime": DEFAULT_DEPLOY_TIME_MS,
        "collisionRadius": DEFAULT_BUILDING_COLLISION_RADIUS if is_building else DEFAULT_COLLISION_RADIUS,
        "tidTarget": "TID_TARGETS_AIR_AND_GROUND" if (rng or 0) >= RANGED_THRESHOLD_TILES else "TID_TARGETS_GROUND",
        "attacksGround": True,
    }
    if stats.damage is not None and rng is not None and rng >= RANGED_THRESHOLD_TILES:
        char["projectileData"] = {
            "name": f"{stats.name}Projectile",
            "speed": DEFAULT_PROJECTILE_SPEED,
            "damage": stats.damage,
            "tidTarget": "TID_TARGETS_AIR_AND_GROUND",
        }
    if not is_building and stats.speed_tiles_per_min is None:
        char["speed"] = 60.0
    return apply_unit_stats(char, stats)


def _base_entry(stats: CardStats, template: Optional[dict]) -> dict:
    entry = copy.deepcopy(template) if template else {}
    entry.pop("evolvedSpellsData", None)
    entry["name"] = stats.name
    entry["id"] = stats.id
    entry["manaCost"] = stats.elixir if stats.elixir is not None else 0
    entry.setdefault("rarity", "Common")
    entry["tidType"] = _tid_type(stats)
    entry["cardType"] = stats.type
    entry["statsSource"] = "Cards.json"
    entry["statsPrescaled"] = True
    entry["variant"] = stats.variant
    return entry


def _patch_structure(name: str, char: dict, stats: CardStats, behaviors: Dict[str, dict], db: CardDatabase) -> None:
    """Behaviour wiring for cards whose behaviour template is missing or outdated.

    Only *structure* is added here (which unit spawns, which projectile is
    thrown); all numbers still come from Cards.json.
    """
    if name == "Furnace":
        # Troop version: ranged attack + periodically spawns Fire Spirits.
        char["spawnCharacterData"] = {"name": "FireSpirit"}
        char.setdefault("spawnNumber", 1)
    elif name == "GoblinDrill":
        char["spawnCharacterData"] = {"name": "Goblin"}
        char.setdefault("spawnNumber", 1)
        char["deathSpawnCharacterData"] = {"name": "Goblin"}
        char.pop("tidTarget", None)
        char.pop("projectileData", None)
    elif name == "SuspiciousBush":
        char["deathSpawnCharacterData"] = {"name": "BushGoblin"}
        char["deathSpawnCount"] = stats.count or 1
        char.pop("deathAreaEffectData", None)
    elif name == "GoblinHut":
        # Reworked hut: a Spear Goblin inside throws spears (range from Cards.json).
        # Cards.json gives no hut damage, so the Spear Goblin's damage is used and
        # `spawnSpeed` is read as the throw interval.
        spear = db.stats("SpearGoblin")
        spear_template = (behaviors.get("SpearGoblin") or {}).get("summonCharacterData") or {}
        projectile = copy.deepcopy(spear_template.get("projectileData") or {"speed": DEFAULT_PROJECTILE_SPEED})
        if spear is not None and spear.damage is not None:
            projectile["damage"] = spear.damage
            char["damage"] = spear.damage
        char["projectileData"] = projectile
        char["tidTarget"] = "TID_TARGETS_AIR_AND_GROUND"
        char["sightRange"] = char.get("range", DEFAULT_SIGHT_RANGE)
        if char.get("spawnPauseTime"):
            char["hitSpeed"] = char.pop("spawnPauseTime")
        char.pop("spawnNumber", None)


def build_character_entry(stats: CardStats, template_entry: Optional[dict], char_template: Optional[dict], db: CardDatabase, behaviors: Optional[Dict[str, dict]] = None) -> dict:
    entry = _base_entry(stats, template_entry)
    if char_template is not None:
        char = copy.deepcopy(char_template)
        char["_template_damage"] = char.get("damage")
        # Compose from the template but keep the Cards.json identity.
        char["name"] = stats.name
        apply_unit_stats(char, stats)
    else:
        char = synthesize_character(stats)
    _patch_structure(stats.name, char, stats, behaviors or {}, db)
    entry["summonCharacterData"] = char
    # Multi-unit cards are composed from CardSpawns.json (see battle spawn code).
    entry.pop("summonNumber", None)
    entry.pop("summonCharacterSecondData", None)
    entry.pop("summonCharacterSecondCount", None)
    entry.pop("summonRadius", None)
    patch_nested_units(entry, db, owner=stats.name)
    return entry


def build_spell_entry(stats: CardStats, template_entry: Optional[dict], db: CardDatabase) -> dict:
    entry = _base_entry(stats, template_entry)
    patch_nested_units(entry, db, owner=stats.name)
    return entry


VariantBuilder = Callable[[dict, CardStats, CardDatabase], dict]
# variant name ("evolution"/"hero") -> builder. Empty until Evos/Heroes land.
VARIANT_BUILDERS: Dict[str, VariantBuilder] = {}


def build_variant_entry(base_entry: dict, variant_stats: CardStats, db: CardDatabase) -> Optional[dict]:
    builder = VARIANT_BUILDERS.get(variant_stats.variant)
    return builder(base_entry, variant_stats, db) if builder else None


def build_entries(behaviors: Optional[Dict[str, dict]] = None, db: Optional[CardDatabase] = None) -> Dict[str, dict]:
    """Return card entries for every base Cards.json card and unit.

    Keys are Cards.json names. Units without hitpoints (projectiles,
    hidden-state duplicates such as ``TeslaHidden``) are skipped.
    """
    db = db or get_card_database()
    behaviors = behaviors if behaviors is not None else load_behaviors()
    entries: Dict[str, dict] = {}

    for stats in db.allStats():
        if stats.variant != VARIANT_BASE:
            continue
        name = stats.name
        if name.endswith("Hidden"):
            continue  # visual/stealth states, modelled by mechanics

        template_entry = behaviors.get(name)
        if stats.is_playable:
            if stats.is_spell:
                entries[name] = build_spell_entry(stats, template_entry, db)
                continue
            char_template = (template_entry or {}).get("summonCharacterData") or None
            if not isinstance(char_template, dict) or not char_template.get("hitpoints") and not char_template.get("speed"):
                char_template = None
            entries[name] = build_character_entry(
                stats, template_entry if char_template else None, char_template, db, behaviors
            )
            continue

        if stats.type not in CHARACTER_TYPES or stats.hitpoints is None:
            continue
        template_char = (template_entry or {}).get("summonCharacterData")
        entries[name] = build_character_entry(stats, None, template_char, db, behaviors)

    # Multi-unit cards have no stat line of their own (hp: null); expose the
    # first sub-unit's line at card level so card-level readers see real data.
    for name, entry in entries.items():
        stats = db.stats(name)
        units = db.spawnUnits(name)
        if not units or stats is None or stats.hitpoints is not None:
            continue
        unit_entry = entries.get(units[0][0])
        if unit_entry and isinstance(unit_entry.get("summonCharacterData"), dict):
            char = copy.deepcopy(unit_entry["summonCharacterData"])
            char["_cardsJson"] = name
            entry["summonCharacterData"] = char
            if db.stats(units[0][0]).is_air:
                entry["cardType"] = "air"

    return entries
