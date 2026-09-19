from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math
import random

from ..mechanic_base import BaseMechanic
from ...factory.dynamic_factory import troop_from_values

if TYPE_CHECKING:
    from ...battle import BattleState


@dataclass
class PeriodicSpawner(BaseMechanic):
    """Mechanic that periodically spawns units"""
    unit_name: str
    spawn_interval_ms: int
    count: int = 1
    max_spawns: int = -1  # -1 for unlimited
    spawn_radius_tiles: float = 1.0

    # Internal state
    time_since_spawn_ms: int = field(init=False, default=0)
    spawns_created: int = field(init=False, default=0)

    def on_tick(self, entity, dt_ms: int) -> None:
        """Check if it's time to spawn units"""
        if not hasattr(entity, 'battle_state'):
            return

        self.time_since_spawn_ms += dt_ms

        # Check if we should spawn and haven't reached max
        if (self.time_since_spawn_ms >= self.spawn_interval_ms and
                (self.max_spawns == -1 or self.spawns_created < self.max_spawns)):

            print(f"[Mechanic] PeriodicSpawner tick on {getattr(entity.card_stats, 'name', 'Unknown')} interval={self.spawn_interval_ms}ms count={self.count}")
            self._spawn_unit(entity)
            self.time_since_spawn_ms = 0
            self.spawns_created += 1

    def _spawn_unit(self, entity) -> None:
        """Spawn a single unit"""
        battle_state = entity.battle_state

        # Try to get spawn stats from card loader
        spawn_stats = battle_state.card_loader.get_card(self.unit_name)

        # If not found, create minimal stats
        if not spawn_stats:
            spawn_stats = troop_from_values(
                self.unit_name,
                hitpoints=100,
                damage=25,
                speed_tiles_per_min=60.0,
                range_tiles=1.0,
                sight_range_tiles=5.0,
                hit_speed_ms=1000,
                collision_radius_tiles=0.5,
            )

        from ...arena import Position
        print(f"[Mechanic] Spawning {self.count}x {self.unit_name} around {getattr(entity.card_stats, 'name', 'Unknown')}")

        from ...entities import Building
        if isinstance(entity, Building):
            positions = building_exit_positions(battle_state, entity, spawn_stats, max(1, self.count))
        else:
            positions = []
            for _ in range(max(1, self.count)):
                angle = random.random() * 2 * math.pi
                distance = random.random() * self.spawn_radius_tiles
                positions.append(Position(entity.position.x + distance * math.cos(angle),
                                          entity.position.y + distance * math.sin(angle)))
        for position in positions:
            battle_state._spawn_troop(position, entity.player_id, spawn_stats)


def building_exit_positions(battle_state, building, unit_stats, count: int):
    """Spots just outside ``building``'s hitbox, fanned out on the side facing
    the enemy (Tombstone skeletons, hut Barbarians/Goblins step out in front
    of the building instead of appearing inside it and getting stuck)."""
    from ...arena import Position

    building_radius = getattr(building.card_stats, "collision_radius", 1.0) or 1.0
    unit_radius = getattr(unit_stats, "collision_radius", 0.5) or 0.5
    distance = building_radius + unit_radius + 0.05
    forward = math.pi / 2 if building.player_id == 0 else -math.pi / 2  # towards the enemy
    spread = math.radians(35)
    offsets = [spread * (i - (count - 1) / 2) for i in range(count)]
    fallbacks = [math.radians(a) for a in (60, -60, 90, -90, 120, -120, 150, -150, 180)]
    positions = []
    for offset in offsets:
        for candidate in [offset] + [offset + f for f in fallbacks]:
            angle = forward + candidate
            pos = Position(building.position.x + distance * math.cos(angle),
                           building.position.y + distance * math.sin(angle))
            if battle_state.arena.is_walkable(pos):
                positions.append(pos)
                break
        else:
            positions.append(Position(building.position.x, building.position.y))
    return positions
