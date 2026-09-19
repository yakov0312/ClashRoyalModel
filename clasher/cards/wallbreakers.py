from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import char_data

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class WallBreakersDemolition(BaseMechanic):
    """Wall Breakers explode on reaching a building: Cards.json ``damage`` is the
    full explosion damage, dealt to every ground enemy within the behaviour-template
    ``areaDamageRadius``. The unit is kamikaze, so the engine removes it after."""
    explosion_radius: float = 1.5

    def on_attach(self, entity: 'Entity') -> None:
        radius = char_data(entity).get("areaDamageRadius")
        if radius:
            self.explosion_radius = radius / 1000.0

    def custom_attack(self, entity: 'Entity', target: 'Entity', battle_state) -> bool:
        from ..entities import Building, Troop
        center = target.position
        for other in list(battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive:
                continue
            if not isinstance(other, (Troop, Building)) or getattr(other, "is_air_unit", False):
                continue
            other_radius = getattr(other.card_stats, "collision_radius", 0.0) or 0.0
            if center.distance_to(other.position) <= self.explosion_radius + other_radius or other is target:
                other.take_damage(entity.damage)
        return True
