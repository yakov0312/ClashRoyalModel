from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number, char_data

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class IceSpiritFreeze(BaseMechanic):
    """Ice Spirit jumps onto its target: Cards.json ``damage`` as area damage and a
    freeze of Cards.json ``duration`` seconds to every enemy in the splash radius
    (behaviour-template projectile ``radius``). Kamikaze, so the engine removes it after."""
    freeze_radius: float = 1.5
    freeze_duration_ms: int = 1100

    def on_attach(self, entity: 'Entity') -> None:
        radius = (char_data(entity).get("projectileData") or {}).get("radius")
        if radius:
            self.freeze_radius = radius / 1000.0
        self.freeze_duration_ms = int(card_number(entity, "duration", self.freeze_duration_ms / 1000.0) * 1000)

    def custom_attack(self, entity: 'Entity', target: 'Entity', battle_state) -> bool:
        from ..entities import Building, Troop
        center = target.position
        for other in list(battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive:
                continue
            if not isinstance(other, (Troop, Building)):
                continue
            other_radius = getattr(other.card_stats, "collision_radius", 0.0) or 0.0
            if center.distance_to(other.position) <= self.freeze_radius + other_radius or other is target:
                damage = entity.damage_against(other, entity.damage) if hasattr(entity, "damage_against") else entity.damage
                other.take_damage(damage)
                other.apply_stun(self.freeze_duration_ms / 1000.0)
        return True
