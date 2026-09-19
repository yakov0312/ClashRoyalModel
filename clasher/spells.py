"""Spell behaviours.

Instances are built from Cards.json stats by ``dynamic_spells.py``; the classes
here only implement behaviour. Units: tiles, seconds, damage per hit unless a
class says "per second".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from .entities import (
    Entity, Projectile, Troop, Building, AreaEffect, SpawnProjectile, RollingProjectile,
    Graveyard, ScheduledEffect, is_crown_tower,
)
from .arena import Position

if TYPE_CHECKING:
    from .battle import BattleState


def _collision_radius(entity: Entity) -> float:
    radius = getattr(getattr(entity, "card_stats", None), "collision_radius", None)
    return radius if radius is not None else 0.5


def _enemies_in_radius(
    battle_state: 'BattleState',
    player_id: int,
    center: Position,
    radius: float,
    hits_air: bool = True,
    hits_ground: bool = True,
    include_buildings: bool = True,
) -> List[Entity]:
    found = []
    for entity in list(battle_state.entities.values()):
        if entity.player_id == player_id or not entity.is_alive:
            continue
        if not isinstance(entity, (Troop, Building)):
            continue
        if isinstance(entity, Building) and not include_buildings:
            continue
        is_air = getattr(entity, "is_air_unit", False)
        if (is_air and not hits_air) or (not is_air and not hits_ground):
            continue
        if entity.position.distance_to(center) <= radius + _collision_radius(entity):
            found.append(entity)
    return found


def _add_entity(battle_state: 'BattleState', entity: Entity) -> Entity:
    entity.id = battle_state.next_entity_id
    battle_state.entities[entity.id] = entity
    battle_state.next_entity_id += 1
    return entity


def _helper_entity(cls, battle_state: 'BattleState', player_id: int, position: Position, **kwargs) -> Entity:
    base = dict(
        id=0,
        position=Position(position.x, position.y),
        player_id=player_id,
        card_stats=None,
        hitpoints=1,
        max_hitpoints=1,
        damage=0,
        range=0,
        sight_range=0,
    )
    base.update(kwargs)
    return _add_entity(battle_state, cls(**base))


@dataclass
class Spell(ABC):
    """Base class for spell effects"""
    name: str
    mana_cost: int
    radius: float = 0.0
    damage: float = 0.0

    @abstractmethod
    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Cast the spell at target position"""
        pass

    def _hitbox_overlaps_with_area(self, entity: 'Entity', area_center: Position) -> bool:
        """Check if entity's hitbox overlaps with spell area using collision detection"""
        distance = entity.position.distance_to(area_center)
        return distance <= (self.radius + _collision_radius(entity))


@dataclass
class DirectDamageSpell(Spell):
    """Spells that deal instant damage in an area (Zap)."""
    stun_duration: float = 0.0
    slow_duration: float = 0.0
    slow_multiplier: float = 1.0
    knockback_distance: float = 0.0
    hits_air: bool = True
    hits_ground: bool = True
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Deal damage to all enemies in radius"""
        targets = _enemies_in_radius(battle_state, player_id, target_pos, self.radius, self.hits_air, self.hits_ground)
        for entity in targets:
            damage = self.damage
            if is_crown_tower(entity):
                damage *= self.crown_tower_damage_multiplier
            entity.take_damage(damage)

            if self.stun_duration > 0:
                entity.apply_stun(self.stun_duration)
            if self.slow_duration > 0:
                entity.apply_slow(self.slow_duration, self.slow_multiplier)
            if self.knockback_distance > 0 and not isinstance(entity, Building):
                dx = entity.position.x - target_pos.x
                dy = entity.position.y - target_pos.y
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance > 0:
                    new_pos = Position(
                        entity.position.x + (dx / distance) * self.knockback_distance,
                        entity.position.y + (dy / distance) * self.knockback_distance,
                    )
                    if getattr(entity, "is_air_unit", False) or battle_state.is_ground_position_walkable(new_pos, entity):
                        entity.position = new_pos
        return bool(targets)


@dataclass
class ProjectileSpell(Spell):
    """Spells that fire projectiles from the King Tower (Fireball, Rocket, Arrows...)."""
    travel_speed: float = 500.0 / 60.0  # tiles/s
    projectile_count: int = 1           # Arrows: number of waves
    wave_interval: float = 0.0          # seconds between waves
    stun_duration: float = 0.0
    slow_duration: float = 0.0
    slow_multiplier: float = 1.0
    knockback_distance: float = 0.0
    hits_air: bool = True
    hits_ground: bool = True
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Fire projectile(s) toward target position"""
        launch_pos = self._get_launch_position(battle_state, player_id)

        for wave in range(max(1, self.projectile_count)):
            projectile = _helper_entity(
                Projectile, battle_state, player_id, launch_pos,
                damage=self.damage,
                target_position=Position(target_pos.x, target_pos.y),
                travel_speed=self.travel_speed,
                splash_radius=self.radius,
                stun_duration=self.stun_duration,
                slow_duration=self.slow_duration,
                slow_multiplier=self.slow_multiplier,
                knockback_distance=self.knockback_distance,
                hits_air=self.hits_air,
                hits_ground=self.hits_ground,
                crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
                activation_delay=wave * self.wave_interval,
            )
            projectile.spell_name = self.name
        return True

    def _get_launch_position(self, battle_state: 'BattleState', player_id: int) -> Position:
        """Spells are thrown from the caster's King Tower."""
        tower_pos = battle_state.arena.BLUE_KING_TOWER if player_id == 0 else battle_state.arena.RED_KING_TOWER
        return Position(tower_pos.x, tower_pos.y)


@dataclass
class AreaEffectSpell(Spell):
    """Spells that leave an area on the ground (Poison, Earthquake, Goblin Curse).

    ``damage`` is per second.
    """
    duration: float = 4.0
    freeze_effect: bool = False
    speed_multiplier: float = 1.0
    slow_affects_attack: bool = True
    hits_air: bool = True
    hits_ground: bool = True
    crown_tower_damage_multiplier: float = 1.0
    building_damage_multiplier: float = 1.0
    amplify_multiplier: float = 1.0
    death_curse_unit: Optional[str] = None
    curse_linger: float = 0.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Create area effect at target position"""
        area_effect = _helper_entity(
            AreaEffect, battle_state, player_id, target_pos,
            damage=self.damage,
            range=self.radius,
            sight_range=self.radius,
            duration=self.duration,
            freeze_effect=self.freeze_effect,
            speed_multiplier=self.speed_multiplier,
            slow_affects_attack=self.slow_affects_attack,
            radius=self.radius,
            hits_air=self.hits_air,
            hits_ground=self.hits_ground,
            crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
            building_damage_multiplier=self.building_damage_multiplier,
            amplify_multiplier=self.amplify_multiplier,
            death_curse_unit=self.death_curse_unit,
            curse_linger=self.curse_linger,
        )
        area_effect.spell_name = self.name
        return True


@dataclass
class SpawnProjectileSpell(ProjectileSpell):
    """Projectile spells that spawn units when they land (Goblin Barrel)."""
    spawn_count: int = 3
    spawn_character: str = "Goblin"
    spawn_character_data: dict = None

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Fire a projectile that spawns units on impact"""
        launch_pos = self._get_launch_position(battle_state, player_id)
        projectile = _helper_entity(
            SpawnProjectile, battle_state, player_id, launch_pos,
            damage=self.damage,
            target_position=Position(target_pos.x, target_pos.y),
            travel_speed=self.travel_speed,
            splash_radius=self.radius,
            spawn_count=self.spawn_count,
            spawn_character=self.spawn_character,
            spawn_character_data=self.spawn_character_data,
        )
        projectile.spell_name = self.name
        return True


@dataclass
class RoyalDeliverySpell(Spell):
    """Delayed impact spell that deals area damage and spawns a recruit."""
    impact_delay: float = 2.0
    travel_speed: float = 5000.0 / 60.0
    spawn_count: int = 1
    spawn_character: str = "RoyalRecruit"
    spawn_character_data: dict = None
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        projectile = _helper_entity(
            SpawnProjectile, battle_state, player_id, target_pos,
            damage=self.damage,
            target_position=Position(target_pos.x, target_pos.y),
            travel_speed=self.travel_speed,
            splash_radius=self.radius,
            spawn_count=self.spawn_count,
            spawn_character=self.spawn_character,
            spawn_character_data=self.spawn_character_data,
            activation_delay=self.impact_delay,
            hits_air=True,
            hits_ground=True,
            crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
        )
        projectile.spell_name = self.name
        return True


@dataclass
class RageSpell(Spell):
    """Rage bottle: lands after ``deploy_time``, deals damage, leaves a boost area.

    Friendly units inside move and attack ``boost_multiplier`` times faster; the
    boost lingers for ``boost_linger`` seconds after leaving the area.
    """
    deploy_time: float = 0.5
    duration: float = 4.5
    boost_multiplier: float = 1.3
    boost_linger: float = 1.0
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        center = Position(target_pos.x, target_pos.y)

        def land(state: 'BattleState', _helper: Entity) -> None:
            for entity in _enemies_in_radius(state, player_id, center, self.radius):
                damage = self.damage * (self.crown_tower_damage_multiplier if is_crown_tower(entity) else 1.0)
                entity.take_damage(damage)
            area = _helper_entity(
                AreaEffect, state, player_id, center,
                range=self.radius, sight_range=self.radius,
                duration=self.duration, radius=self.radius,
                boost_multiplier=self.boost_multiplier,
                boost_linger=self.boost_linger,
            )
            area.spell_name = self.name

        helper = _helper_entity(ScheduledEffect, battle_state, player_id, center, schedule=[(self.deploy_time, land)])
        helper.spell_name = self.name
        return True


@dataclass
class FreezeSpell(Spell):
    """Instant damage, then enemies in the area are frozen for ``freeze_duration``."""
    freeze_duration: float = 3.0
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        targets = _enemies_in_radius(battle_state, player_id, target_pos, self.radius)
        for entity in targets:
            damage = self.damage * (self.crown_tower_damage_multiplier if is_crown_tower(entity) else 1.0)
            entity.take_damage(damage)
            entity.apply_stun(self.freeze_duration)
            entity.apply_slow(self.freeze_duration, 0.0)
        return bool(targets)


@dataclass
class LightningSpell(Spell):
    """Strikes the ``max_targets`` highest-hitpoint enemies in the radius, one after another."""
    max_targets: int = 3
    strike_interval: float = 0.46
    stun_duration: float = 0.5
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        center = Position(target_pos.x, target_pos.y)
        targets = sorted(
            _enemies_in_radius(battle_state, player_id, center, self.radius),
            key=lambda e: e.hitpoints,
            reverse=True,
        )[: self.max_targets]

        def strike_for(target: Entity):
            def strike(_state: 'BattleState', _helper: Entity) -> None:
                if not target.is_alive:
                    return
                damage = self.damage * (self.crown_tower_damage_multiplier if is_crown_tower(target) else 1.0)
                target.take_damage(damage)
                target.apply_stun(self.stun_duration)
            return strike

        schedule = [(i * self.strike_interval, strike_for(t)) for i, t in enumerate(targets)]
        if schedule:
            helper = _helper_entity(ScheduledEffect, battle_state, player_id, center, schedule=schedule)
            helper.spell_name = self.name
        return bool(targets)


@dataclass
class VoidSpell(Spell):
    """Void: pulses ``pulses`` times; per-target damage depends on how many are hit.

    ``tiers`` is ``[(min_targets, damage, crown_damage), ...]`` sorted by
    ``min_targets`` (Cards.json: 1 / 2-4 / 5+ targets).
    """
    duration: float = 4.0
    first_hit_delay: float = 1.0
    hit_interval: float = 1.0
    tiers: List[tuple] = field(default_factory=list)

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        center = Position(target_pos.x, target_pos.y)

        def pulse(state: 'BattleState', _helper: Entity) -> None:
            targets = _enemies_in_radius(state, player_id, center, self.radius)
            if not targets:
                return
            damage, crown_damage = self._tier(len(targets))
            for target in targets:
                target.take_damage(crown_damage if is_crown_tower(target) else damage)

        times = []
        t = self.first_hit_delay
        while t < self.duration - 1e-6:
            times.append(t)
            t += self.hit_interval
        helper = _helper_entity(ScheduledEffect, battle_state, player_id, center, schedule=[(t, pulse) for t in times])
        helper.spell_name = self.name
        return True

    def _tier(self, count: int) -> tuple:
        chosen = self.tiers[0][1:] if self.tiers else (self.damage, self.damage)
        for min_targets, damage, crown in self.tiers:
            if count >= min_targets:
                chosen = (damage, crown)
        return chosen


@dataclass
class VinesSpell(Spell):
    """Vines: grab the ``max_targets`` highest-HP enemies, root them (pulling air
    units down) and deal damage per second for ``duration``."""
    max_targets: int = 3
    duration: float = 2.0
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        center = Position(target_pos.x, target_pos.y)
        targets = sorted(
            _enemies_in_radius(battle_state, player_id, center, self.radius),
            key=lambda e: e.hitpoints,
            reverse=True,
        )[: self.max_targets]
        for target in targets:
            if not isinstance(target, Building):
                target.apply_root(self.duration)
            # Per-target damage-over-time area pinned on the target.
            area = _helper_entity(
                AreaEffect, battle_state, player_id, target.position,
                damage=self.damage, range=0.0, sight_range=0.0, duration=self.duration, radius=0.0,
                crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
            )
            area.position = target.position  # follows the target
            area.spell_name = self.name
        return bool(targets)


@dataclass
class CloneSpell(Spell):
    """Clones friendly troops in the radius; clones have ``clone_hitpoints`` HP."""
    clone_hitpoints: float = 1.0
    clone_shield_hitpoints: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        from .mechanics.shared.shield import Shield

        originals = [
            e for e in list(battle_state.entities.values())
            if isinstance(e, Troop) and e.player_id == player_id and e.is_alive
            and not getattr(e, "is_clone", False)
            and e.position.distance_to(target_pos) <= self.radius + _collision_radius(e)
        ]
        for troop in originals:
            offset = 0.5 if player_id == 0 else -0.5
            clone = battle_state._create_troop(
                Position(troop.position.x + offset, troop.position.y), player_id, troop.card_stats
            )
            clone.deploy_delay_remaining = 0.0
            clone.hitpoints = clone.max_hitpoints = self.clone_hitpoints
            clone.is_clone = True
            for mechanic in clone.mechanics:
                if isinstance(mechanic, Shield):
                    mechanic.current_shield = min(mechanic.current_shield, self.clone_shield_hitpoints)
        return bool(originals)


@dataclass
class HealSpell(Spell):
    """Legacy instant heal (the Heal Spirit is now a troop)."""
    heal_amount: float = 400.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        hit = False
        for entity in battle_state.entities.values():
            if entity.player_id == player_id and entity.is_alive and isinstance(entity, Troop):
                if entity.position.distance_to(target_pos) <= self.radius:
                    entity.heal(self.heal_amount)
                    hit = True
        return hit


@dataclass
class RollingProjectileSpell(Spell):
    """Spells that spawn at location and roll forward (Log, Barbarian Barrel)"""
    travel_speed: float = 200.0
    projectile_range: float = 10.0
    spawn_delay: float = 0.65
    spawn_character: str = None
    spawn_character_data: dict = None
    radius_y: float = 0.6
    knockback_distance: float = 1.5
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        """Spawn rolling projectile at target position"""
        rolling_projectile = _helper_entity(
            RollingProjectile, battle_state, player_id, target_pos,
            damage=self.damage,
            range=self.radius,
            travel_speed=self.travel_speed,
            projectile_range=self.projectile_range,
            spawn_delay=self.spawn_delay,
            spawn_character=self.spawn_character,
            spawn_character_data=self.spawn_character_data,
            radius_y=self.radius_y,
            knockback_distance=self.knockback_distance,
            crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
        )
        rolling_projectile.spell_name = self.name
        return True


@dataclass
class TornadoSpell(Spell):
    """Pulls enemies towards the centre and deals damage per second."""
    pull_force: float = 3.0
    damage_per_second: float = 35.0
    duration: float = 3.0
    hits_air: bool = True
    hits_ground: bool = True
    crown_tower_damage_multiplier: float = 1.0

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        tornado = _helper_entity(
            AreaEffect, battle_state, player_id, target_pos,
            damage=self.damage_per_second,
            range=self.radius,
            sight_range=self.radius,
            duration=self.duration,
            radius=self.radius,
            hits_air=self.hits_air,
            hits_ground=self.hits_ground,
            crown_tower_damage_multiplier=self.crown_tower_damage_multiplier,
        )
        tornado.spell_name = self.name
        tornado.pull_force = self.pull_force
        tornado.is_tornado = True
        return True


@dataclass
class GraveyardSpell(Spell):
    """Spawns ``max_skeletons`` skeletons evenly over ``duration`` inside the radius."""
    spawn_interval: float = 0.5
    max_skeletons: int = 20
    duration: float = 10.0
    skeleton_name: str = "Larry"
    skeleton_data: dict = None

    def cast(self, battle_state: 'BattleState', player_id: int, target_pos: Position) -> bool:
        graveyard = _helper_entity(
            Graveyard, battle_state, player_id, target_pos,
            range=self.radius,
            sight_range=self.radius,
            spawn_interval=self.spawn_interval,
            max_skeletons=self.max_skeletons,
            spawn_radius=self.radius,
            duration=self.duration,
            skeleton_name=self.skeleton_name,
            skeleton_data=self.skeleton_data,
        )
        graveyard.spell_name = self.name
        return True


def _load_dynamic_spell_registry() -> Dict[str, Spell]:
    """Build every spell from Cards.json (+ CardBehaviors.json templates)."""
    from .dynamic_spells import load_dynamic_spells
    return load_dynamic_spells()


SPELL_REGISTRY = _load_dynamic_spell_registry()
