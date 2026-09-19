from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number, card_range, char_data

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class MegaKnightSlam(BaseMechanic):
    """Mega Knight: ``spawnDamage`` slam on deploy, and a ``jumpDamage`` leap onto
    targets between ``jumpRange`` min/max tiles. Normal hits are area damage via
    the generic ``areaDamageRadius`` splash."""
    spawn_radius: float = 2.0
    spawn_damage: float = 0.0
    jump_damage: float = 0.0
    jump_min_range: float = 3.5
    jump_max_range: float = 5.0
    slam_radius: float = 1.3
    leap_duration_ms: int = 600
    jump_cooldown_ms: int = 1000

    def on_attach(self, entity: 'Entity') -> None:
        char = char_data(entity)
        if char.get("areaDamageRadius"):
            self.slam_radius = char["areaDamageRadius"] / 1000.0
        self.spawn_damage = card_number(entity, "spawnDamage", getattr(entity, "damage", 0) * 2)
        self.jump_damage = card_number(entity, "jumpDamage", char.get("dashDamage") or getattr(entity, "damage", 0) * 2)
        jump_range = card_range(entity, "jumpRange")
        if jump_range:
            self.jump_min_range, self.jump_max_range = jump_range
        self._leap_target = None
        self._leap_origin = None
        self._leap_elapsed = 0.0
        self._last_jump_ms = -10**9

    def on_spawn(self, entity: 'Entity') -> None:
        self._slam(entity, self.spawn_radius, self.spawn_damage)

    def on_tick(self, entity: 'Entity', dt_ms: int) -> None:
        state = getattr(entity, "battle_state", None)
        if state is None or entity.deploy_delay_remaining > 0:
            return
        if self._leap_target is not None:
            self._leap_elapsed += dt_ms
            t = min(1.0, self._leap_elapsed / self.leap_duration_ms)
            (sx, sy), (tx, ty) = self._leap_origin, self._leap_target
            entity.position.x = sx + (tx - sx) * t
            entity.position.y = sy + (ty - sy) * t
            if t >= 1.0:
                self._leap_target = None
                self._slam(entity, self.slam_radius, self.jump_damage)
            return

        target = state.entities.get(entity.target_id) if entity.target_id else None
        now_ms = int(state.time * 1000)
        if target is None or not target.is_alive or now_ms - self._last_jump_ms < self.jump_cooldown_ms:
            return
        if getattr(target, "is_air_unit", False):
            return
        distance = entity.position.distance_to(target.position)
        if self.jump_min_range <= distance <= self.jump_max_range:
            self._leap_origin = (entity.position.x, entity.position.y)
            self._leap_target = (target.position.x, target.position.y)
            self._leap_elapsed = 0.0
            self._last_jump_ms = now_ms

    def _slam(self, entity: 'Entity', radius: float, damage: float) -> None:
        if not hasattr(entity, 'battle_state') or not damage:
            return
        from ..entities import Building, Troop
        for other in list(entity.battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive:
                continue
            if not isinstance(other, (Troop, Building)) or getattr(other, "is_air_unit", False):
                continue
            other_radius = getattr(other.card_stats, "collision_radius", 0.5) or 0.5
            if entity.position.distance_to(other.position) <= radius + other_radius:
                other.take_damage(damage)
