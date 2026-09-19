from typing import Dict, Any, Optional, Tuple
from ..card_types import (
    CardDefinition, CardKind, Rarity, TroopStats, BuildingStats, SpellStats,
    TargetingBehavior, MovementBehavior, AttackBehavior, Mechanic, Effect
)
from .mechanic_detector import detect_mechanics_from_data, detect_effects_from_data


def _units_to_tiles(value: Optional[float]) -> Optional[float]:
    """Convert raw milli-tile units to tiles, preserving None for missing data.

    NOTE: previously call sites used `char_data.get(key, 0) / 1000.0`, which
    silently turned a *missing* field into a real 0.0 value instead of None.
    That 0.0 then survived every downstream `is not None` check as if it were
    legitimate data, and any `or`-based fallback further downstream (battle.py)
    treated it as falsy and substituted grossly oversized defaults (100 tile
    range / 500 tile sight). Always route conversions through this helper so
    "missing" stays None all the way up the chain.
    """
    if value is None:
        return None
    return value / 1000.0


def _determine_card_kind(entry: Dict[str, Any]) -> CardKind:
    """Determine card kind from a card entry"""
    tid_type = entry.get("tidType")
    if tid_type == "TID_CARD_TYPE_CHARACTER":
        return "troop"
    elif tid_type == "TID_CARD_TYPE_SPELL":
        return "spell"
    elif tid_type == "TID_CARD_TYPE_BUILDING":
        return "building"
    elif entry.get("rarity") == "Champion":
        return "champion"
    return "troop"  # Default


def _create_targeting_behavior(entry: Dict[str, Any]) -> Optional[TargetingBehavior]:
    """Create targeting behavior from the card entry"""
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})
    target_type = char_data.get("tidTarget", "")

    class SimpleTargeting:
        def __init__(self, target_type_str: str):
            self.target_type_str = target_type_str

        def can_target_air(self) -> bool:
            return "AIR" in self.target_type_str

        def can_target_ground(self) -> bool:
            return "GROUND" in self.target_type_str or "BUILDINGS" in self.target_type_str

        def buildings_only(self) -> bool:
            return self.target_type_str == "TID_TARGETS_BUILDINGS"

        def get_nearest_target(self, entity, entities: dict):
            # Use existing entity targeting logic
            return entity.get_nearest_target(entities)

    return SimpleTargeting(target_type)


def _create_movement_behavior(entry: Dict[str, Any]) -> Optional[MovementBehavior]:
    """Create movement behavior for troops"""
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})
    speed = char_data.get("speed")

    if speed is None:
        return None

    class SimpleMovement:
        def __init__(self, speed_val: float):
            self.speed_val = speed_val

        def update(self, entity, dt_ms: int) -> None:
            # Use existing troop movement logic
            pass  # Movement is handled by Troop.update()

    return SimpleMovement(speed)


def _create_attack_behavior(entry: Dict[str, Any]) -> Optional[AttackBehavior]:
    """Create attack behavior from the card entry"""
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})

    class SimpleAttack:
        def __init__(self, has_projectile: bool):
            self.has_projectile = has_projectile

        def maybe_attack(self, entity, dt_ms: int) -> None:
            # Use existing entity attack logic
            pass  # Attacks are handled by Troop/Building.update()

    has_projectile = bool(char_data.get("projectileData"))
    return SimpleAttack(has_projectile)


def _create_troop_stats(entry: Dict[str, Any]) -> Optional[TroopStats]:
    """Create troop stats from a card entry"""
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})

    if not char_data:  # No character data, not a troop
        return None

    return TroopStats(
        hitpoints=char_data.get("hitpoints"),
        damage=char_data.get("damage"),
        range_tiles=_units_to_tiles(char_data.get("range")),
        hit_speed_ms=char_data.get("hitSpeed"),
        sight_range_tiles=_units_to_tiles(char_data.get("sightRange")),
        collision_radius_tiles=_units_to_tiles(char_data.get("collisionRadius")),
        speed_tiles_per_min=char_data.get("speed"),
        deploy_time_ms=char_data.get("deployTime", 1000),
        load_time_ms=char_data.get("loadTime", 1000),
        summon_count=entry.get("summonNumber")
    )


def _create_building_stats(entry: Dict[str, Any]) -> Optional[BuildingStats]:
    """Create building stats from a card entry"""
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})

    if not char_data or entry.get("tidType") != "TID_CARD_TYPE_BUILDING":
        return None

    return BuildingStats(
        hitpoints=char_data.get("hitpoints"),
        damage=char_data.get("damage"),
        range_tiles=_units_to_tiles(char_data.get("range")),
        hit_speed_ms=char_data.get("hitSpeed"),
        sight_range_tiles=_units_to_tiles(char_data.get("sightRange")),
        collision_radius_tiles=_units_to_tiles(char_data.get("collisionRadius")),
        lifetime_ms=char_data.get("lifeTime") or char_data.get("lifetime"),
        deploy_time_ms=char_data.get("deployTime", 1000)
    )


def _create_spell_stats(entry: Dict[str, Any]) -> Optional[SpellStats]:
    """Create spell stats from a card entry"""
    if entry.get("tidType") != "TID_CARD_TYPE_SPELL":
        return None

    return SpellStats(
        radius_tiles=_units_to_tiles(entry.get("radius")),
        duration_ms=entry.get("duration", 0),
        crown_tower_damage_scale=entry.get("crownTowerDamagePercent", 100) / 100.0
    )


def _extract_tags(entry: Dict[str, Any]) -> frozenset[str]:
    """Extract tags from a card entry"""
    tags = set()

    # Add tribe as tag
    if entry.get("tribe"):
        tags.add(entry["tribe"].lower())

    # Add targeting tags
    char_data = entry.get("summonCharacterData", {}) or entry.get("summonSpellData", {})
    target_type = char_data.get("tidTarget", "")
    if "AIR" in target_type:
        tags.add("anti-air")
    if "BUILDINGS" in target_type:
        tags.add("anti-building")

    # Add special mechanics tags
    if char_data.get("shieldHitpoints"):
        tags.add("shield")
    if char_data.get("chargeRange"):
        tags.add("charge")
    if char_data.get("deathSpawnCharacter"):
        tags.add("death-spawn")
    if entry.get("rarity") == "Champion":
        tags.add("champion")

    return frozenset(tags)


def card_from_gamedata(entry: Dict[str, Any]) -> CardDefinition:
    """Create CardDefinition from a card entry"""
    mechanics = detect_mechanics_from_data(entry)
    effects = detect_effects_from_data(entry)

    card_def = CardDefinition(
        id=entry["id"],
        name=entry["name"],
        kind=_determine_card_kind(entry),
        rarity=entry["rarity"],
        elixir=entry["manaCost"],
        troop_stats=_create_troop_stats(entry),
        building_stats=_create_building_stats(entry),
        spell_stats=_create_spell_stats(entry),
        targeting=_create_targeting_behavior(entry),
        movement=_create_movement_behavior(entry),
        attack=_create_attack_behavior(entry),
        mechanics=tuple(mechanics),
        effects=tuple(effects),
        tags=_extract_tags(entry),
        raw=entry
    )

    return card_def


def create_card_definition(
    name: str,
    kind: CardKind,
    rarity: Rarity,
    elixir: int,
    troop_stats: Optional[TroopStats] = None,
    building_stats: Optional[BuildingStats] = None,
    spell_stats: Optional[SpellStats] = None,
    mechanics: Optional[list[Mechanic]] = None,
    effects: Optional[list[Effect]] = None,
    tags: Optional[set[str]] = None,
    raw: Optional[Dict[str, Any]] = None
) -> CardDefinition:
    """Create a CardDefinition with given parameters"""
    return CardDefinition(
        id=0,  # Placeholder ID
        name=name,
        kind=kind,
        rarity=rarity,
        elixir=elixir,
        troop_stats=troop_stats,
        building_stats=building_stats,
        spell_stats=spell_stats,
        mechanics=tuple(mechanics or []),
        effects=tuple(effects or []),
        tags=frozenset(tags or set()),
        raw=raw or {}
    )
