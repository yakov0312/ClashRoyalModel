"""Build spell objects from Cards.json stats.

Cards.json supplies every number it has (damage, crown-tower damage, radius,
duration, counts, boosts...). The behaviour template is consulted only for
behaviour it does not describe (from ``CardBehaviors.json``, same names):
projectile travel speed, pushback, wave interval, slow percentages, hit timing.

Damage for lingering area spells (Poison, Earthquake, Tornado, Goblin Curse,
Vines) is treated as damage *per second*, the Cards.json convention.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .factory.stats_overlay import get_card_database, load_behaviors
from .spells import (
    AreaEffectSpell,
    CloneSpell,
    DirectDamageSpell,
    FreezeSpell,
    GraveyardSpell,
    LightningSpell,
    ProjectileSpell,
    RageSpell,
    RollingProjectileSpell,
    RoyalDeliverySpell,
    SpawnProjectileSpell,
    Spell,
    TornadoSpell,
    VinesSpell,
    VoidSpell,
)

# Behavioural constants that neither Cards.json nor the behaviour templates describe.
TORNADO_PULL_TILES_PER_S = 3.0
CURSE_LINGER_S = 1.0  # death-curse/amplify persists briefly after leaving the area
ROLLING_SPAWN_DELAY_S = 0.65


def _per_second(speed: Optional[float], default: float) -> float:
    """Template projectile speeds are tiles per minute -> tiles/s = speed / 60."""
    return (speed if speed is not None else default) / 60.0


def _crown_multiplier(stats, template_percent: Optional[float] = None) -> float:
    damage, crown = stats.damage, stats.crown_damage
    if damage and crown is not None:
        return crown / damage
    if template_percent is not None:
        return max(0.0, 1.0 + template_percent / 100.0)
    return 1.0


def _slow_multiplier(buff: Dict[str, Any], default_percent: float) -> float:
    percent = buff.get("speedMultiplier", -default_percent) if buff else -default_percent
    return max(0.0, 1.0 + percent / 100.0)


def _build(stats, entry: Dict[str, Any]) -> Optional[Spell]:
    name = stats.name
    cost = stats.elixir or 0
    damage = stats.damage or 0.0
    radius = stats.radius or 0.0
    projectile = entry.get("projectileData") or {}
    area = entry.get("areaEffectObjectData") or {}

    if name == "Arrows":
        return ProjectileSpell(
            name, cost, radius=radius, damage=damage,
            travel_speed=_per_second(projectile.get("speed"), 1100),
            projectile_count=stats.count or 1,
            wave_interval=(entry.get("projectileWaveInterval") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, projectile.get("crownTowerDamagePercent")),
        )
    if name in ("Fireball", "Rocket"):
        return ProjectileSpell(
            name, cost, radius=radius, damage=damage,
            travel_speed=_per_second(projectile.get("speed"), 600),
            knockback_distance=(projectile.get("pushback") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, projectile.get("crownTowerDamagePercent")),
        )
    if name == "Snowball":
        return ProjectileSpell(
            name, cost, radius=radius, damage=damage,
            travel_speed=_per_second(projectile.get("speed"), 800),
            knockback_distance=(projectile.get("pushback") or 0) / 1000.0,
            slow_duration=stats.duration or (projectile.get("buffTime") or 0) / 1000.0,
            slow_multiplier=_slow_multiplier(projectile.get("targetBuffData") or {}, 35),
            crown_tower_damage_multiplier=_crown_multiplier(stats, projectile.get("crownTowerDamagePercent")),
        )
    if name == "Zap":
        return DirectDamageSpell(
            name, cost, radius=radius, damage=damage,
            stun_duration=stats.duration or (area.get("buffTime") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, area.get("crownTowerDamagePercent")),
        )
    if name == "Lightning":
        strike = area.get("projectileData") or {}
        return LightningSpell(
            name, cost, radius=radius, damage=damage,
            max_targets=stats.count or 3,
            strike_interval=(area.get("hitSpeed") or 460) / 1000.0,
            stun_duration=stats.duration or (strike.get("buffTime") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, strike.get("crownTowerDamagePercent")),
        )
    if name == "Freeze":
        return FreezeSpell(
            name, cost, radius=radius, damage=damage,
            freeze_duration=stats.duration or (area.get("buffTime") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, area.get("crownTowerDamagePercent")),
        )
    if name in ("Poison", "Earthquake"):
        buff = area.get("buffData") or {}
        slowdown = stats.number("slowdown")
        building_multiplier = 1.0
        building_damage = stats.number("buildingDamage")
        if building_damage is not None and damage:
            building_multiplier = building_damage / damage
        return AreaEffectSpell(
            name, cost, radius=radius, damage=damage,
            duration=stats.duration or (area.get("lifeDuration") or 0) / 1000.0,
            speed_multiplier=(1.0 - slowdown / 100.0) if slowdown is not None else _slow_multiplier(buff, 0),
            slow_affects_attack=bool(buff.get("hitSpeedMultiplier")),
            hits_air=area.get("hitsAir", True),
            hits_ground=area.get("hitsGround", True),
            crown_tower_damage_multiplier=_crown_multiplier(stats, buff.get("crownTowerDamagePercent")),
            building_damage_multiplier=building_multiplier,
        )
    if name == "Tornado":
        buff = area.get("buffData") or {}
        return TornadoSpell(
            name, cost, radius=radius, damage=0,
            pull_force=TORNADO_PULL_TILES_PER_S,
            damage_per_second=damage,
            duration=stats.duration or (area.get("lifeDuration") or 0) / 1000.0,
            hits_air=area.get("hitsAir", True),
            hits_ground=area.get("hitsGround", True),
            crown_tower_damage_multiplier=_crown_multiplier(stats, buff.get("crownTowerDamagePercent")),
        )
    if name == "GoblinCurse":
        # Cards.json names this value "slowdown"; the template curse buff is a
        # damage amplifier ("damageReduction": -20), so it is applied as extra
        # damage taken. Cursed troops that die become Goblins for the caster.
        amplify = stats.number("slowdown")
        return AreaEffectSpell(
            name, cost, radius=radius, damage=damage,
            duration=stats.duration or (area.get("lifeDuration") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats),
            amplify_multiplier=1.0 + (amplify or 0) / 100.0,
            death_curse_unit="Goblin",
            curse_linger=CURSE_LINGER_S,
        )
    if name == "Rage":
        rage_area = ((entry.get("summonCharacterData") or {}).get("deathAreaEffectData")) or {}
        return RageSpell(
            name, cost, radius=radius, damage=damage,
            deploy_time=stats.deploy_time or 0.0,
            duration=stats.duration or (rage_area.get("lifeDuration") or 0) / 1000.0,
            boost_multiplier=1.0 + (stats.number("boost") or 0) / 100.0,
            boost_linger=(rage_area.get("buffTime") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats),
        )
    if name == "Clone":
        return CloneSpell(
            name, cost, radius=radius, damage=0,
            clone_hitpoints=stats.number("cloneHealth", 1.0),
            clone_shield_hitpoints=stats.number("cloneShieldHealth", 1.0),
        )
    if name == "GoblinBarrel":
        return SpawnProjectileSpell(
            name, cost, radius=radius, damage=damage,
            travel_speed=_per_second(projectile.get("speed"), 400),
            spawn_count=int(stats.number("spawnCount", 3)),
            spawn_character="Goblin",
        )
    if name == "Graveyard":
        count = int(stats.number("spawnCount", 1))
        duration = stats.duration or (area.get("lifeDuration") or 0) / 1000.0
        return GraveyardSpell(
            name, cost, radius=radius, damage=0,
            spawn_interval=duration / max(1, count),
            max_skeletons=count,
            duration=duration + 1e-3,
            skeleton_name="Larry",
        )
    if name in ("Log", "BarbarianBarrel"):
        rolling = projectile.get("spawnProjectileData") or {}
        spawn_unit = "Barbarian" if name == "BarbarianBarrel" else None
        width = stats.number("width")
        return RollingProjectileSpell(
            name, cost,
            radius=(width / 2.0) if width is not None else (projectile.get("radius") or 0) / 1000.0,
            damage=damage,
            travel_speed=rolling.get("speed", 200),
            projectile_range=stats.range or (rolling.get("projectileRange") or 0) / 1000.0,
            spawn_delay=ROLLING_SPAWN_DELAY_S,
            spawn_character=spawn_unit,
            radius_y=(projectile.get("radiusY") or 600) / 1000.0,
            knockback_distance=(rolling.get("pushback") or 0) / 1000.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats, rolling.get("crownTowerDamagePercent")),
        )
    if name == "RoyalDelivery":
        delivery = area.get("projectileData") or {}
        return RoyalDeliverySpell(
            name, cost, radius=radius, damage=damage,
            impact_delay=(area.get("lifeDuration") or 2000) / 1000.0,
            travel_speed=_per_second(delivery.get("speed"), 5000),
            spawn_count=stats.count or 1,
            spawn_character="RoyalRecruit",
            crown_tower_damage_multiplier=_crown_multiplier(stats),
        )
    if name == "Void":
        tiers = [(1, damage, stats.crown_damage if stats.crown_damage is not None else damage)]
        for min_targets, dmg_key, crown_key in ((2, "damage2to4Targets", "crownDamage2to4Targets"),
                                                (5, "damage5PlusTargets", "crownDamage5PlusTargets")):
            dmg = stats.number(dmg_key)
            if dmg is not None:
                crown = stats.number(crown_key)
                tiers.append((min_targets, dmg, crown if crown is not None else dmg))
        return VoidSpell(
            name, cost, radius=radius, damage=damage,
            duration=stats.duration or 4.0,
            first_hit_delay=1.0,
            hit_interval=1.0,
            tiers=tiers,
        )
    if name == "Vines":
        return VinesSpell(
            name, cost, radius=radius, damage=damage,
            max_targets=stats.count or 3,
            duration=stats.duration or 0.0,
            crown_tower_damage_multiplier=_crown_multiplier(stats),
        )
    return None


def load_dynamic_spells() -> Dict[str, Spell]:
    """Build every base-variant spell in Cards.json."""
    db = get_card_database()
    behaviors = load_behaviors()

    registry: Dict[str, Spell] = {}
    for stats in db.allStats():
        if not stats.is_spell or stats.variant != "base":
            continue
        spell = _build(stats, behaviors.get(stats.name, {}))
        if spell is not None:
            registry[stats.name] = spell
    return registry


if __name__ == "__main__":
    for spell_name, spell in load_dynamic_spells().items():
        print(f"  {spell_name}: {spell}")
