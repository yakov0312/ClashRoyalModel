from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number

if TYPE_CHECKING:
    from ..entities import Entity


def _distance_tiles(a, b) -> float:
    distance = a.position.distance_to(b.position)
    # Unit-test mocks use milli-tile coordinates.
    return distance / 1000.0 if distance > 100 else distance


@dataclass
class ElectroWizardSpawnZap(BaseMechanic):
    """Deploy zap: Cards.json ``spawnDamage`` + stun around the Electro Wizard."""
    radius_tiles: float = 4.0  # Deploy zap radius
    stun_duration_ms: int = 500  # 0.5 seconds
    damage_scale: float = 0.5  # fallback when no absolute spawn damage is known
    spawn_damage: Optional[float] = None

    def on_attach(self, entity: 'Entity') -> None:
        raw = getattr(entity.card_stats, "_raw_entry", {}) if hasattr(entity, "card_stats") else {}
        area_data = raw.get("areaEffectObjectData", {}) if isinstance(raw, dict) else {}
        if isinstance(area_data, dict):
            if area_data.get("radius") is not None:
                self.radius_tiles = area_data["radius"] / 1000.0
            if area_data.get("buffTime") is not None:
                self.stun_duration_ms = area_data["buffTime"]
        self.spawn_damage = card_number(entity, "spawnDamage", self.spawn_damage)
        stun = card_number(entity, "duration")
        if stun is not None:
            self.stun_duration_ms = int(stun * 1000)

    def on_spawn(self, entity: 'Entity') -> None:
        """Zap on deploy - stuns and damages enemies"""
        if not hasattr(entity, 'battle_state'):
            return
        battle_state = entity.battle_state
        damage = self.spawn_damage if self.spawn_damage is not None else (entity.damage or 0) * self.damage_scale

        for other in list(battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive:
                continue
            if type(other).__name__ not in {"Troop", "Building", "MockEntity"}:
                continue
            if _distance_tiles(entity, other) <= self.radius_tiles:
                other.take_damage(damage)
                if hasattr(other, 'apply_stun'):
                    other.apply_stun(self.stun_duration_ms / 1000.0)


@dataclass
class ElectroWizardStunAttack(BaseMechanic):
    """Electro Wizard fires two bolts per attack (one per target, or both into a
    single target); every bolt deals full damage and stuns."""
    stun_duration_ms: int = 500  # 0.5 seconds
    chain_targets: int = 2  # Attack splits to 2 targets
    chain_damage_scale: float = 1.0

    def on_attach(self, entity: 'Entity') -> None:
        stun = card_number(entity, "duration")
        if stun is not None:
            self.stun_duration_ms = int(stun * 1000)

    def on_attack_hit(self, entity: 'Entity', target: 'Entity') -> None:
        """Apply stun when Electro Wizard hits a target"""
        if hasattr(target, 'apply_stun'):
            target.apply_stun(self.stun_duration_ms / 1000.0)

        if hasattr(entity, 'battle_state') and self.chain_targets > 1:
            self._apply_chain_lightning(entity, target)

    def _apply_chain_lightning(self, entity: 'Entity', primary_target: 'Entity') -> None:
        battle_state = entity.battle_state
        range_value = entity.range if hasattr(entity, 'range') else 3.0
        radius_tiles = range_value / 1000.0 if range_value > 100 else range_value

        secondary_target = None
        min_distance = float('inf')
        for other in list(battle_state.entities.values()):
            if other == primary_target or other.player_id == entity.player_id or not other.is_alive:
                continue
            if type(other).__name__ not in {"Troop", "Building", "MockEntity"}:
                continue
            distance_tiles = _distance_tiles(entity, other)
            if distance_tiles <= radius_tiles and distance_tiles < min_distance:
                secondary_target = other
                min_distance = distance_tiles

        # With a single enemy in range, both bolts strike it.
        bolt_target = secondary_target or primary_target
        if not getattr(bolt_target, "is_alive", True):
            return
        damage = (entity.damage or 0) * self.chain_damage_scale
        bolt_target.take_damage(damage)
        if hasattr(bolt_target, 'apply_stun'):
            bolt_target.apply_stun(self.stun_duration_ms / 1000.0)
