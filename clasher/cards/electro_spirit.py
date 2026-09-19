from dataclasses import dataclass
from typing import TYPE_CHECKING, Set

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class ElectroSpiritChain(BaseMechanic):
    """Electro Spirit: on impact the zap chains to nearby enemies (Cards.json
    ``chainedAttacks`` targets in total, full damage and ``stunDuration`` stun each)."""
    chain_range: float = 4.0
    max_targets: int = 9
    stun_duration_ms: int = 500

    def on_attach(self, entity: 'Entity') -> None:
        self.max_targets = int(card_number(entity, "chainedAttacks", self.max_targets))
        self.stun_duration_ms = int(card_number(entity, "stunDuration", self.stun_duration_ms / 1000.0) * 1000)

    def on_attack_hit(self, entity: 'Entity', target: 'Entity') -> None:
        """The spirit's own projectile hits ``target``; chain to the rest."""
        if not hasattr(entity, 'battle_state'):
            return
        battle_state = entity.battle_state
        visited: Set[int] = {target.id}
        current_pos = target.position
        for _ in range(self.max_targets - 1):
            nxt = self._find_next_target(battle_state, entity.player_id, visited, current_pos)
            if not nxt:
                break
            visited.add(nxt.id)
            nxt.take_damage(entity.damage_against(nxt, entity.damage) if hasattr(entity, "damage_against") else entity.damage)
            nxt.apply_stun(self.stun_duration_ms / 1000.0)
            current_pos = nxt.position

    def _find_next_target(self, battle_state, player_id: int, visited: Set[int], origin) -> 'Entity':
        from ..entities import Building, Troop
        best = None
        best_distance = self.chain_range + 1.0
        for other in list(battle_state.entities.values()):
            if other.player_id == player_id or not other.is_alive or other.id in visited:
                continue
            if not isinstance(other, (Troop, Building)):
                continue
            distance = origin.distance_to(other.position)
            if distance <= self.chain_range and distance < best_distance:
                best_distance = distance
                best = other
        return best
