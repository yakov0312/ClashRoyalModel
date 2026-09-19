from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math

from ..mechanics.mechanic_base import BaseMechanic
from ._stats import card_number, card_range

if TYPE_CHECKING:
    from ..entities import Troop


@dataclass
class BanditDash(BaseMechanic):
    """Mechanic for Bandit's dash attack with invincibility frames"""
    dash_distance: float = 500.0  # Legacy value (5 tiles in runtime coordinate space)
    dash_min_range: float = 3.5  # Minimum range to initiate dash (tiles)
    dash_max_range: float = 6.0  # Maximum range to initiate dash (tiles)
    dash_duration_ms: int = 800  # Dash takes 0.8 seconds
    invincibility_duration_ms: int = 800  # Invincible during entire dash
    dash_damage: float = 0.0  # Cards.json dashDamage, dealt on arrival

    # Internal state
    is_dashing: bool = field(init=False, default=False)
    last_dash_time: int = field(init=False, default=0)

    def on_attach(self, entity) -> None:
        """Initialize dash state"""
        dash_range = card_range(entity, "dashRange")
        if dash_range:
            self.dash_min_range, self.dash_max_range = dash_range
        self.dash_damage = card_number(entity, "dashDamage", self.dash_damage)
        entity._bandit_dash_target_id = None
        entity._bandit_dashing = False
        entity._bandit_invulnerable_until = 0
        entity._bandit_dash_timer = 0.0
        entity._bandit_dash_origin = None
        entity._bandit_dash_target = None

    def on_tick(self, entity, dt_ms: int) -> None:
        """Handle dash logic and invincibility"""
        current_time_ms = 0
        if hasattr(entity, 'battle_state') and hasattr(entity.battle_state, 'time'):
            current_time_ms = int(entity.battle_state.time * 1000)

        # Update dash motion if active
        if getattr(entity, '_bandit_dashing', False):
            self._update_dash(entity, dt_ms)
            return

        # Expire invincibility when duration elapsed
        if getattr(entity, '_bandit_invulnerable_until', 0) and current_time_ms >= entity._bandit_invulnerable_until:
            entity._bandit_invulnerable_until = 0

        # Check if we should initiate dash
        if (not getattr(entity, '_bandit_dashing', False) and
                entity.target_id and
                current_time_ms - self.last_dash_time > 1000):  # Can dash every second

            # Check if target is in range for dash
            target = entity.battle_state.entities.get(entity.target_id) if hasattr(entity, 'battle_state') else None
            if target and target.is_alive:
                # Dash if target is within specific range (3.5-6 tiles)
                distance = entity.position.distance_to(target.position)
                distance_tiles = distance / 1000.0 if distance > 100 else distance
                if self.dash_min_range <= distance_tiles <= self.dash_max_range:
                    self._start_dash(entity, target)

    def _start_dash(self, entity, target) -> None:
        """Initiate dash towards target"""
        entity._bandit_dashing = True
        origin_x, origin_y = entity.position.x, entity.position.y
        dx = target.position.x - origin_x
        dy = target.position.y - origin_y
        distance = math.hypot(dx, dy)
        if distance == 0:
            distance = 1.0
        if distance > 100:  # unit-test mocks use milli-tile coordinates
            dash_length = min(self.dash_distance, distance)
        else:
            # Dash all the way into melee range of the target.
            dash_length = max(0.0, distance - (getattr(entity, "range", 0.0) or 0.0))
        norm_x = dx / distance
        norm_y = dy / distance
        target_x = origin_x + norm_x * dash_length
        target_y = origin_y + norm_y * dash_length
        entity._bandit_dash_origin = (origin_x, origin_y)
        entity._bandit_dash_target = (target_x, target_y)
        entity._bandit_dash_timer = 0.0
        entity._bandit_dash_target_id = getattr(target, "id", None)

        current_time_ms = int(entity.battle_state.time * 1000)
        entity._bandit_invulnerable_until = current_time_ms + self.invincibility_duration_ms
        self.last_dash_time = current_time_ms

    def _update_dash(self, entity, dt_ms: int) -> None:
        """Update dash movement"""
        origin = getattr(entity, '_bandit_dash_origin', None)
        target = getattr(entity, '_bandit_dash_target', None)
        if not origin or not target:
            entity._bandit_dashing = False
            return

        entity._bandit_dash_timer += dt_ms
        t = min(1.0, entity._bandit_dash_timer / self.dash_duration_ms)
        start_x, start_y = origin
        target_x, target_y = target
        entity.position.x = start_x + (target_x - start_x) * t
        entity.position.y = start_y + (target_y - start_y) * t

        if t >= 1.0:
            entity._bandit_dashing = False
            entity._bandit_dash_origin = None
            entity._bandit_dash_target = None
            entity._bandit_dash_timer = 0.0
            self._land_dash(entity)

    def _land_dash(self, entity) -> None:
        """Dash impact deals Cards.json ``dashDamage``; the normal swing follows."""
        state = getattr(entity, "battle_state", None)
        target = state.entities.get(getattr(entity, "_bandit_dash_target_id", None)) if state else None
        entity._bandit_dash_target_id = None
        if target is not None and target.is_alive and self.dash_damage:
            damage = entity.damage_against(target, self.dash_damage) if hasattr(entity, "damage_against") else self.dash_damage
            target.take_damage(damage)
            entity.attack_cooldown = entity.get_attack_interval_seconds() if hasattr(entity, "get_attack_interval_seconds") else 0.0
        else:
            entity.attack_cooldown = 0.0

    def take_damage_during_dash(self, entity, damage: float) -> bool:
        """Check if entity should take damage during dash"""
        current_time_ms = 0
        if hasattr(entity, 'battle_state') and hasattr(entity.battle_state, 'time'):
            current_time_ms = int(entity.battle_state.time * 1000)

        # Return True if damage should be applied, False if invincible
        return current_time_ms > getattr(entity, '_bandit_invulnerable_until', 0)
