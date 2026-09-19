from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..arena import Position
from ..mechanics.mechanic_base import BaseMechanic

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class LumberjackRage(BaseMechanic):
    """On death the Lumberjack drops a Rage spell (the Rage card's Cards.json stats)."""
    spell_name: str = "Rage"

    def on_death(self, entity: 'Entity') -> None:
        if not hasattr(entity, 'battle_state'):
            return
        from ..spells import SPELL_REGISTRY
        rage = SPELL_REGISTRY.get(self.spell_name)
        if rage is not None:
            rage.cast(entity.battle_state, entity.player_id, Position(entity.position.x, entity.position.y))
