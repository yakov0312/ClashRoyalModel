from dataclasses import dataclass
from typing import TYPE_CHECKING
import math

from ..arena import Position
from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class FirecrackerRecoil(BaseMechanic):
    """Firecracker: the rocket bursts at the target into ``count`` sparks
    (Cards.json) that each deal ``damage`` in a cone behind the target; the
    Firecracker recoils after every shot."""
    recoil_distance: float = 1.0
    shard_half_angle: float = 0.45
    shard_hit_width: float = 0.4
    shard_count: int = 5
    shard_range: float = 5.0

    def on_attach(self, entity: 'Entity') -> None:
        projectile = getattr(getattr(entity, "card_stats", None), "projectile_data", None) or {}
        sparks = projectile.get("spawnProjectileData") or {}
        self.shard_count = int(card_number(entity, "count", sparks.get("spawnCount") or self.shard_count))
        if sparks.get("projectileRange"):
            self.shard_range = sparks["projectileRange"] / 1000.0
        if sparks.get("radius"):
            self.shard_hit_width = sparks["radius"] / 1000.0

    def custom_attack(self, entity: 'Entity', target: 'Entity', battle_state) -> bool:
        self._apply_shards(entity, target)
        dx = entity.position.x - target.position.x
        dy = entity.position.y - target.position.y
        length = math.hypot(dx, dy)
        if length > 0:
            candidate = Position(
                entity.position.x + (dx / length) * self.recoil_distance,
                entity.position.y + (dy / length) * self.recoil_distance,
            )
            if battle_state.is_ground_position_walkable(candidate, entity):
                entity.position = candidate
        return True

    def _apply_shards(self, entity: 'Entity', target: 'Entity') -> None:
        from ..entities import Building, Troop
        damage = entity.damage
        dx = target.position.x - entity.position.x
        dy = target.position.y - entity.position.y
        base_len = math.hypot(dx, dy)
        if base_len == 0 or damage <= 0:
            return
        base_angle = math.atan2(dy, dx)

        for idx in range(self.shard_count):
            if self.shard_count == 1:
                shard_angle = base_angle
            else:
                t = idx / (self.shard_count - 1)
                shard_angle = base_angle - self.shard_half_angle + (2 * self.shard_half_angle * t)
            ux = math.cos(shard_angle)
            uy = math.sin(shard_angle)
            for other in list(entity.battle_state.entities.values()):
                if other.player_id == entity.player_id or not other.is_alive:
                    continue
                if not isinstance(other, (Troop, Building)):
                    continue
                rel_x = other.position.x - target.position.x
                rel_y = other.position.y - target.position.y
                other_radius = getattr(other.card_stats, "collision_radius", 0.5) or 0.5
                along = rel_x * ux + rel_y * uy
                if along < -other_radius or along > self.shard_range:
                    continue
                perp = abs(rel_x * uy - rel_y * ux)
                if perp <= (self.shard_hit_width + other_radius):
                    other.take_damage(entity.damage_against(other, damage))
