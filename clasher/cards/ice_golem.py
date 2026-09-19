from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import char_data

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class IceGolemChill(BaseMechanic):
    """Ice Golem death nova slow. The death *damage* (Cards.json ``deathDamage``)
    is dealt by the generic DeathDamage mechanic; radius/slow come from its behaviour template."""
    slow_radius: float = 2.0
    slow_multiplier: float = 0.65
    slow_duration_ms: int = 2000

    def on_attach(self, entity: 'Entity') -> None:
        area = char_data(entity).get("deathAreaEffectData") or {}
        if area.get("radius"):
            self.slow_radius = area["radius"] / 1000.0
        if area.get("buffTime"):
            self.slow_duration_ms = int(area["buffTime"])
        percent = (area.get("buffData") or {}).get("speedMultiplier")
        if percent is not None:
            self.slow_multiplier = max(0.0, 1.0 + percent / 100.0)

    def on_death(self, entity: 'Entity') -> None:
        if not hasattr(entity, 'battle_state'):
            return
        for other in list(entity.battle_state.entities.values()):
            if other.player_id == entity.player_id or not other.is_alive:
                continue
            if type(other).__name__ not in {"Troop", "Building"}:
                continue
            if entity.position.distance_to(other.position) <= self.slow_radius:
                other.apply_slow(self.slow_duration_ms / 1000.0, self.slow_multiplier)
