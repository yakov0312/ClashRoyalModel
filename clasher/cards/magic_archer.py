from dataclasses import dataclass
from typing import TYPE_CHECKING
import math

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class MagicArcherPierce(BaseMechanic):
    """Magic Archer's arrow keeps flying through its target up to Cards.json
    ``projectileRange`` (measured from the archer), hitting everything on the line."""
    projectile_range: float = 11.0
    perpendicular_tolerance: float = 0.25
    damage_decay: float = 1.0

    def on_attach(self, entity: 'Entity') -> None:
        self.projectile_range = card_number(entity, "projectileRange", self.projectile_range)
        projectile = getattr(getattr(entity, "card_stats", None), "projectile_data", None) or {}
        if projectile.get("radius"):
            self.perpendicular_tolerance = projectile["radius"] / 1000.0

    def on_attack_hit(self, entity: 'Entity', target: 'Entity') -> None:
        if not hasattr(entity, 'battle_state'):
            return
        from ..entities import Building, Troop
        dx = target.position.x - entity.position.x
        dy = target.position.y - entity.position.y
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        damage = entity.damage * self.damage_decay
        for other in list(entity.battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive or other.id == target.id:
                continue
            if not isinstance(other, (Troop, Building)):
                continue
            rel_x = other.position.x - entity.position.x
            rel_y = other.position.y - entity.position.y
            along = rel_x * ux + rel_y * uy
            if along <= length or along > self.projectile_range:
                continue
            radius = getattr(other.card_stats, "collision_radius", 0.5) or 0.5
            if abs(rel_x * uy - rel_y * ux) <= self.perpendicular_tolerance + radius:
                other.take_damage(entity.damage_against(other, damage))
