from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class ValkyrieSpin(BaseMechanic):
    """Valkyrie's 360-degree spin: her area damage (behaviour-template ``areaDamageRadius``)
    is centred on herself rather than on her target."""

    def on_attach(self, entity: 'Entity') -> None:
        entity._splash_center_self = True
