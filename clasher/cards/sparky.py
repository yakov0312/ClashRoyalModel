from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..mechanics.mechanic_base import BaseMechanic

if TYPE_CHECKING:
    from ..entities import Troop


@dataclass
class SparkyChargeUp(BaseMechanic):
    """Sparky: every shot deals the full Cards.json damage, but the gun needs a
    full ``hitSpeed`` to charge, starts uncharged, and any stun resets the charge.

    (Stun already restarts the attack timer in ``Entity.apply_stun``; this
    mechanic makes Sparky deploy uncharged.)
    """
    reset_on_stun: bool = True

    def on_spawn(self, entity) -> None:
        entity.attack_cooldown = max(entity.attack_cooldown, entity.get_attack_interval_seconds())

    def handle_stun(self, entity) -> None:
        if self.reset_on_stun:
            entity.attack_cooldown = entity.get_attack_interval_seconds()
