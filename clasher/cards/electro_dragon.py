from dataclasses import dataclass
from typing import TYPE_CHECKING, Set

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number

if TYPE_CHECKING:
    from ..entities import Entity


@dataclass
class ElectroDragonChainLightning(BaseMechanic):
    """Electro Dragon's bolt hits ``chainedAttacks`` targets in total, each for full
    damage and a ``duration`` stun (Cards.json)."""
    chain_range: float = 4.5
    max_bounces: int = 2
    damage_decay: float = 1.0
    stun_duration_ms: int = 500

    def on_attach(self, entity: 'Entity') -> None:
        chained = card_number(entity, "chainedAttacks")
        if chained is not None and chained > 0:
            self.max_bounces = int(chained) - 1
        self.stun_duration_ms = int(card_number(entity, "duration", self.stun_duration_ms / 1000.0) * 1000)

    def on_attack_hit(self, entity: 'Entity', target: 'Entity') -> None:
        if not hasattr(entity, 'battle_state'):
            return
        battle_state = entity.battle_state
        visited: Set[int] = {target.id}
        previous = target
        damage = entity.damage * self.damage_decay
        for _ in range(self.max_bounces):
            candidate = self._find_next_target(battle_state, entity.player_id, visited, previous.position)
            if not candidate:
                break
            visited.add(candidate.id)
            candidate.take_damage(damage)
            candidate.apply_stun(self.stun_duration_ms / 1000.0)
            previous = candidate
            damage *= self.damage_decay

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
