from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math
import random
import copy
import json

from .entities import Entity, Troop, Building
from .player import PlayerState
from .arena import TileGrid, Position
from .data import CardDataLoader
from .card_types import CardStatsCompat
from .factory.dynamic_factory import (
    building_from_values,
    troop_from_character_data,
    troop_from_values,
)
from .spells import SPELL_REGISTRY
from .mechanics.shared.death_effects import DeathSpawn
from .towers import CROWN_TOWER_NAMES, KING_TOWER, PRINCESS_TOWER, load_tower_stats, tower_footprint_size

# Sane, tile-scale fallbacks used only if a card genuinely carries no
# range/sight data at all. Real CR values: range 0.4-7 tiles, sight ~5.5-7.5
# tiles. Never raise these — an oversized value makes a unit think it's
# already in attack range from spawn, or lets it "see" targets clear across
# the arena (this was the actual cause of units not moving / troops chasing
# far-away units instead of the nearby tower).
DEFAULT_RANGE_TILES = 1.2
DEFAULT_SIGHT_RANGE_TILES = 5.5


DEPLOY_ANYWHERE_CARDS = {"Miner", "GoblinDrill"}


@dataclass(frozen=True)
class _ElixirCost:
    mana_cost: float


def _is_air(card_stats: CardStatsCompat) -> bool:
    return bool(getattr(card_stats, "is_air", False))


def _safe_range(card_stats: CardStatsCompat) -> float:
    value = getattr(card_stats, "range", None)
    return value if value is not None else DEFAULT_RANGE_TILES


def _safe_sight_range(card_stats: CardStatsCompat) -> float:
    value = getattr(card_stats, "sight_range", None)
    return value if value is not None else DEFAULT_SIGHT_RANGE_TILES


@dataclass
class BattleState:
    # Core state
    entities: Dict[int, Entity] = field(default_factory=dict)
    players: List[PlayerState] = field(default_factory=lambda: [PlayerState(0), PlayerState(1)])
    arena: TileGrid = field(default_factory=TileGrid)
    
    # Timing
    time: float = 0.0
    tick: int = 0
    dt: float = 0.033  # 33ms per tick (~30 FPS)
    
    # Game state
    double_elixir: bool = False
    triple_elixir: bool = False
    overtime: bool = False
    sudden_death: bool = False
    game_over: bool = False
    winner: Optional[int] = None
    overtime_start_time: float = 240.0
    sudden_death_start_time: float = 300.0
    tiebreaker_time: float = 360.0
    
    # Data
    card_loader: CardDataLoader = field(default_factory=CardDataLoader)
    next_entity_id: int = 1
    _starting_total_tower_hp: Dict[int, float] = field(default_factory=dict, init=False)
    _sudden_death_crowns: Tuple[int, int] = field(default=(0, 0), init=False)
    
    def __post_init__(self) -> None:
        """Initialize battle state"""
        self.card_loader.load_cards()
        # Preload factory card definitions to enable mechanic detection/prints
        try:
            self.card_loader.load_card_definitions()
        except Exception as e:
            print(f"[Warn] load_card_definitions failed: {e}")
        self._create_towers()
        self._starting_total_tower_hp = {
            0: self.players[0].king_tower_hp + self.players[0].left_tower_hp + self.players[0].right_tower_hp,
            1: self.players[1].king_tower_hp + self.players[1].left_tower_hp + self.players[1].right_tower_hp,
        }
    
    def _create_towers(self) -> None:
        """Create both players' crown towers from ``Towers.json``."""
        stats = {name: self._tower_card_stats(name) for name in (PRINCESS_TOWER, KING_TOWER)}
        layout = (
            (0, self.arena.BLUE_LEFT_TOWER, PRINCESS_TOWER),
            (0, self.arena.BLUE_RIGHT_TOWER, PRINCESS_TOWER),
            (0, self.arena.BLUE_KING_TOWER, KING_TOWER),
            (1, self.arena.RED_LEFT_TOWER, PRINCESS_TOWER),
            (1, self.arena.RED_RIGHT_TOWER, PRINCESS_TOWER),
            (1, self.arena.RED_KING_TOWER, KING_TOWER),
        )
        for player_id, pos, name in layout:
            tower = self._spawn_entity(Building, Position(pos.x, pos.y), player_id, stats[name])
            # Crown towers are up from the first tick (no deploy / first-hit delay).
            tower.deploy_delay_remaining = 0.0
            tower.attack_cooldown = 0.0
        for player in self.players:
            player.king_tower_hp = stats[KING_TOWER].hitpoints
            player.left_tower_hp = player.right_tower_hp = stats[PRINCESS_TOWER].hitpoints

    @staticmethod
    def _tower_card_stats(name: str) -> CardStatsCompat:
        data = load_tower_stats()[name]
        return building_from_values(
            name=name,
            hitpoints=data["hp"],
            damage=data["damage"],
            range_tiles=data["range"],
            sight_range_tiles=data["range"],
            hit_speed_ms=int(round(data["hitSpeed"] * 1000)),
            deploy_time_ms=0,
            collision_radius_tiles=data["collisionRadius"],
            lifetime_ms=None,
            elixir=0,
            rarity="Common",
            projectile_speed=data["projectileSpeed"] * 60.0,  # engine unit: tiles per minute
            projectile_damage=data["damage"],
            target_type="TID_TARGETS_AIR_AND_GROUND",
            raw_overrides={"statsPrescaled": True},
        )

    def step(self, speed_factor: float = 1.0) -> None:
        """Advance battle by one tick"""
        if self.game_over:
            return
        
        dt = self.dt * speed_factor
        self.time += dt
        self.tick += 1
        
        # Update elixir modes
        if self.time >= 180.0 and not self.double_elixir:
            self.double_elixir = True
        if self.time >= self.overtime_start_time and not self.overtime:
            self.overtime = True
        if self.time >= 240.0 and not self.triple_elixir:
            self.triple_elixir = True
        
        # Regenerate elixir
        base_regen = 2.8
        if self.triple_elixir:
            base_regen = 0.93
        elif self.double_elixir:
            base_regen = 1.4
        
        for player in self.players:
            player.regenerate_elixir(dt, base_regen)
        
        # Update all entities
        for entity in list(self.entities.values()):
            entity.update(dt, self)

        # Resolve simple body collision to reduce unit stacking.
        self._resolve_troop_collisions()
        
        # Remove dead entities
        self._cleanup_dead_entities()
        
        # Check win conditions
        self._check_win_conditions()
    
    def deploy_card(self, player_id: int, card_name: str, position: Position) -> bool:
        """Deploy a card at the given position"""
        player = self.players[player_id]
        resolved_name = card_name
        card_stats = self.card_loader.get_card(card_name)

        if not card_stats or not card_stats.is_playable or card_name not in player.hand:
            return False

        # Mirror replays the last card played for one more elixir.
        cost_stats = card_stats
        if resolved_name == "Mirror":
            mirrored = getattr(player, "last_played_card", None)
            mirrored_stats = self.card_loader.get_card(mirrored) if mirrored else None
            if mirrored_stats is None:
                return False
            cost_stats = _ElixirCost(mirrored_stats.mana_cost + 1)
            resolved_name, card_stats = mirrored, mirrored_stats

        if not player.can_play_card(card_name, cost_stats):
            return False

        is_spell = resolved_name in SPELL_REGISTRY
        spell_obj = SPELL_REGISTRY.get(resolved_name) if is_spell else None
        # Miner / Goblin Drill can be deployed almost anywhere (except blocked/tower tiles).
        if resolved_name in DEPLOY_ANYWHERE_CARDS:
            tile_pos = (int(position.x), int(position.y))
            if not self.arena.is_valid_position(position):
                return False
            if tile_pos in self.arena.BLOCKED_TILES:
                return False
            if self.arena.is_tower_tile(position, self):
                return False
        else:
            if not self.arena.can_deploy_at(position, player_id, self, is_spell, spell_obj):
                return False

        # Special deployment validation for Royal Recruits (center 6 tiles only)
        if resolved_name == 'RoyalRecruits':
            if not (6 <= position.x <= 11):
                return False  # Royal Recruits can only be deployed in center 6 tiles

        if not is_spell:
            # Buildings are solid obstacles; do not allow overlapping deployment.
            card_type = str(getattr(card_stats, "card_type", "") or "").lower()
            is_building_card = card_type == "building"
            if is_building_card:
                if self.is_building_placement_occupied(position, card_stats):
                    return False
            else:
                probe_radius = getattr(card_stats, "collision_radius", 0.5) or 0.5
                if self.is_position_occupied_by_building(position, probe_radius):
                    return False

        # Play the card
        if not player.play_card(card_name, cost_stats):
            return False
        player.last_played_card = resolved_name

        # Check if it's a spell
        if resolved_name in SPELL_REGISTRY:
            spell = SPELL_REGISTRY[resolved_name]
            spell.cast(self, player_id, position)
        else:
            # Spawn troop or building based on card type (robust to missing/None fields)
            card_type = getattr(card_stats, "card_type", None)
            card_type_str = str(card_type).lower() if card_type is not None else ""
            if card_type_str == "building":
                self._spawn_entity(Building, position, player_id, card_stats)
            else:
                self._spawn_troop(position, player_id, card_stats)
        
        return True
    
    def _spawn_troop(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> None:
        """Spawn a troop card: a single unit, or every sub-unit of a multi-unit card."""
        # Guard: if this card is actually a building, route to building spawner
        ctype = getattr(card_stats, "card_type", None)
        ctype_str = str(ctype).lower() if ctype is not None else ""
        if ctype_str == "building":
            self._spawn_entity(Building, position, player_id, card_stats)
            return

        units = list(getattr(card_stats, "spawn_units", None) or [])
        if units:
            self._spawn_formation(position, player_id, card_stats, units)
            return

        self._create_troop(position, player_id, card_stats)

    # Kept for callers/tests that spawn one unit at an exact spot.
    def _spawn_single_troop(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> Troop:
        return self._create_troop(position, player_id, card_stats)

    def _spawn_unit_at_position(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> Troop:
        return self._create_troop(position, player_id, card_stats)

    def _create_troop(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> Troop:
        """Create, register and fire spawn hooks for one troop entity."""
        scaled_hp = card_stats.scaled_hitpoints or card_stats.hitpoints or 1
        scaled_damage = card_stats.scaled_damage
        if scaled_damage is None:
            scaled_damage = card_stats.damage or 0

        troop = Troop(
            id=self.next_entity_id,
            position=Position(position.x, position.y),
            player_id=player_id,
            card_stats=card_stats,
            hitpoints=scaled_hp,
            max_hitpoints=scaled_hp,
            damage=scaled_damage,
            range=_safe_range(card_stats),
            sight_range=_safe_sight_range(card_stats),
            speed=card_stats.speed or 60.0,
            is_air_unit=_is_air(card_stats),
        )

        troop.deploy_delay_remaining = max(0.0, (getattr(card_stats, "deploy_time", 0) or 0) / 1000.0)
        troop.attack_cooldown = max(troop.attack_cooldown, (getattr(card_stats, "load_time", 0) or 0) / 1000.0)
        troop.battle_state = self
        self._attach_mechanics(troop, card_stats)

        self.entities[self.next_entity_id] = troop
        self.next_entity_id += 1
        troop.on_spawn()
        return troop

    def _attach_mechanics(self, entity: Entity, card_stats: CardStatsCompat) -> None:
        """Attach fresh copies of the card definition's mechanics."""
        try:
            card_def = getattr(card_stats, 'card_definition', None)
            if not card_def or not getattr(card_def, 'mechanics', None):
                defn_name = getattr(card_stats, 'name', None)
                card_def = self.card_loader.get_card_definition(defn_name) if defn_name else None
            if card_def and getattr(card_def, 'mechanics', None):
                entity.mechanics = [copy.deepcopy(m) for m in card_def.mechanics]
                for mech in entity.mechanics:
                    mech.on_attach(entity)
        except Exception as e:
            print(f"[Warn] Failed attaching mechanics for {getattr(card_stats, 'name', 'Unknown')}: {e}")

    def _spawn_formation(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, units: List[Tuple[str, int]]) -> None:
        """Deploy a multi-unit card using CardSpawns.json layout extents."""
        resolved: List[Tuple[CardStatsCompat, int]] = []
        for unit_name, count in units:
            unit_stats = card_stats if unit_name == card_stats.name else self.card_loader.get_card(unit_name)
            if unit_stats is None or count <= 0:
                continue
            resolved.append((unit_stats, int(count)))
        if not resolved:
            return

        layout = getattr(card_stats, "spawn_layout", None) or {}
        spread_x = float(layout.get("maxDistanceX", 2.0))
        spread_y = float(layout.get("maxDistanceY", 2.0))

        if card_stats.name in ('RoyalRecruits', 'RoyalRecruits_Chess'):
            unit_stats, count = resolved[0]
            self._spawn_royal_recruits_line(center_pos, player_id, unit_stats, count)
            return

        if len(resolved) == 1:
            unit_stats, count = resolved[0]
            for pos in self._ring_positions(center_pos, count, spread_x / 2.0, spread_y / 2.0):
                self._create_troop(self._snap_to_valid_position(pos, player_id), player_id, unit_stats)
            return

        # Mixed swarms: melee units in front, ranged units behind (Goblin Gang, Rascals, Goblinstein).
        forward = 1.0 if player_id == 0 else -1.0
        front = [(s, n) for s, n in resolved if _safe_range(s) < 2.0]
        back = [(s, n) for s, n in resolved if _safe_range(s) >= 2.0]
        rows = [(front, spread_y / 4.0), (back, -spread_y / 4.0)] if front and back else [(resolved, 0.0)]
        for row, y_offset in rows:
            row_units = [s for s, n in row for _ in range(n)]
            for i, unit_stats in enumerate(row_units):
                if len(row_units) == 1:
                    x = center_pos.x
                else:
                    x = center_pos.x - spread_x / 2.0 + spread_x * i / (len(row_units) - 1)
                pos = Position(x, center_pos.y + forward * y_offset)
                self._create_troop(self._snap_to_valid_position(pos, player_id), player_id, unit_stats)

    @staticmethod
    def _ring_positions(center: Position, count: int, semi_x: float, semi_y: float) -> List[Position]:
        """Deterministic, evenly filled placement inside an ellipse (sunflower pattern)."""
        if count <= 1:
            return [Position(center.x, center.y)]
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        positions = []
        for i in range(count):
            r = math.sqrt((i + 0.5) / count)
            theta = i * golden_angle
            positions.append(Position(center.x + semi_x * r * math.cos(theta), center.y + semi_y * r * math.sin(theta)))
        return positions

    def _spawn_royal_recruits_line(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, count: int) -> None:
        """Spawn Royal Recruits in a horizontal line across the battlefield, avoiding towers"""
        # Royal Recruits: 6 units spaced 2.5 tiles apart, center at deploy position
        spacing = 2.5  # tiles between each recruit
        
        # Get tower-blocked X ranges for this Y coordinate
        blocked_ranges = self.arena.get_tower_blocked_x_ranges(center_pos.y, self)
        
        # Calculate initial line positions
        total_width = (count - 1) * spacing
        leftmost_x = center_pos.x - (total_width / 2)
        
        # Generate all recruit X positions
        recruit_positions = []
        for i in range(count):
            recruit_x = leftmost_x + (i * spacing)
            recruit_positions.append(recruit_x)
        
        # Check if any positions would overlap with towers
        needs_adjustment = False
        for recruit_x in recruit_positions:
            for x_min, x_max in blocked_ranges:
                if x_min <= recruit_x <= x_max:
                    needs_adjustment = True
                    break
            if needs_adjustment:
                break
        
        # If line overlaps with towers, find alternative positioning
        if needs_adjustment:
            recruit_positions = self._find_safe_recruit_positions(center_pos, count, spacing, blocked_ranges)
        
        # Ensure all positions are within arena bounds
        recruit_positions = [max(0.5, min(17.5, x)) for x in recruit_positions]
        
        # Spawn each recruit
        for recruit_x in recruit_positions:
            recruit_pos = Position(recruit_x, center_pos.y)
            recruit_pos = self._snap_to_valid_position(recruit_pos, player_id)
            self._spawn_unit_at_position(recruit_pos, player_id, card_stats)
    
    def _find_safe_recruit_positions(self, center_pos: Position, count: int, spacing: float, blocked_ranges: List[Tuple[float, float]]) -> List[float]:
        """Find safe X positions for Royal Recruits that avoid tower collisions"""
        
        # Create list of all blocked X coordinates
        blocked_x_coords = set()
        for x_min, x_max in blocked_ranges:
            # Add all positions in blocked range with 0.5 precision
            x = x_min
            while x <= x_max:
                blocked_x_coords.add(round(x * 2) / 2)  # Round to nearest 0.5
                x += 0.5
        
        # Find all safe X positions across the arena
        safe_positions = []
        for x_half in range(1, 36):  # 0.5 to 17.5 in 0.5 increments
            x = x_half / 2.0
            if x not in blocked_x_coords and 0.5 <= x <= 17.5:
                safe_positions.append(x)
        
        # If we have enough safe positions, try to maintain spacing
        if len(safe_positions) >= count:
            # Try to find positions with good spacing starting from center
            selected_positions = []
            
            # Find safe position closest to center
            center_candidates = [pos for pos in safe_positions if abs(pos - center_pos.x) <= 1.0]
            if not center_candidates:
                center_candidates = safe_positions
            
            center_safe = min(center_candidates, key=lambda x: abs(x - center_pos.x))
            selected_positions.append(center_safe)
            
            # For remaining positions, try to maintain spacing while staying safe
            while len(selected_positions) < count:
                best_candidate = None
                best_score = float('inf')
                
                for candidate in safe_positions:
                    if candidate in selected_positions:
                        continue
                    
                    # Score based on distance from ideal spacing positions
                    min_spacing_score = float('inf')
                    for existing_pos in selected_positions:
                        spacing_distance = abs(abs(candidate - existing_pos) - spacing)
                        min_spacing_score = min(min_spacing_score, spacing_distance)
                    
                    # Prefer positions that maintain good spacing
                    if min_spacing_score < best_score:
                        best_score = min_spacing_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected_positions.append(best_candidate)
                else:
                    # If no good candidate, just pick the first available
                    for pos in safe_positions:
                        if pos not in selected_positions:
                            selected_positions.append(pos)
                            break
            
            positions = selected_positions
        else:
            # Not enough safe positions, use what we have
            positions = safe_positions[:count]
        
        # Ensure we have exactly the right count
        while len(positions) < count:
            # Add fallback positions at arena edges
            for x in [0.5, 1.0, 17.0, 17.5, 16.5, 16.0]:
                if x not in positions and x not in blocked_x_coords:
                    positions.append(x)
                    if len(positions) >= count:
                        break
        
        # Sort and return exact count
        positions.sort()
        return positions[:count]
    
    def _snap_to_valid_position(self, position: Position, player_id: int) -> Position:
        """Snap position to nearest valid playable area"""
        # Check if position is already valid (walkable and not on a tower)
        if self.arena.is_walkable(position) and not self.arena.is_tower_tile(position, self):
            return position
        
        # Try to find nearest valid position within reasonable distance
        search_radius = 2.0  # tiles
        best_position = position
        min_distance = float('inf')
        
        # Search in a grid around the original position
        for x_offset in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            for y_offset in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                test_x = position.x + x_offset
                test_y = position.y + y_offset
                test_pos = Position(test_x, test_y)
                
                # Check bounds first
                if not self.arena.is_valid_position(test_pos):
                    continue
                
                # Check if walkable and not on tower
                if self.arena.is_walkable(test_pos) and not self.arena.is_tower_tile(test_pos, self):
                    distance = position.distance_to(test_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_position = test_pos
        
        # If no valid position found nearby, clamp to arena bounds and find closest walkable
        if min_distance == float('inf'):
            # Clamp to arena bounds
            clamped_x = max(0.5, min(17.5, position.x))
            clamped_y = max(0.5, min(31.5, position.y))
            clamped_pos = Position(clamped_x, clamped_y)
            
            # If clamped position is walkable and not on tower, use it
            if self.arena.is_walkable(clamped_pos) and not self.arena.is_tower_tile(clamped_pos, self):
                best_position = clamped_pos
            else:
                # Fallback: move towards arena center until we find walkable area
                center_x, center_y = 9.0, 16.0
                dx = center_x - clamped_x
                dy = center_y - clamped_y
                
                for step in [0.5, 1.0, 1.5, 2.0]:
                    fallback_x = clamped_x + dx * step * 0.1
                    fallback_y = clamped_y + dy * step * 0.1
                    fallback_pos = Position(fallback_x, fallback_y)
                    
                    if (self.arena.is_valid_position(fallback_pos) and 
                        self.arena.is_walkable(fallback_pos) and
                        not self.arena.is_tower_tile(fallback_pos, self)):
                        best_position = fallback_pos
                        break
        
        return best_position
    
    def _create_card_stats_from_data(self, unit_data: dict, name: str) -> CardStatsCompat:
        """Create CardStatsCompat from raw unit data (for secondary units in mixed swarms)."""
        if unit_data:
            rarity = unit_data.get("rarity", "Common")
            return troop_from_character_data(name, unit_data, elixir=0, rarity=rarity)

        # Fallback minimal definition if no data is provided
        return troop_from_values(
            name,
            hitpoints=100,
            damage=10,
            speed_tiles_per_min=60.0,
            range_tiles=1.0,
            sight_range_tiles=5.0,
            hit_speed_ms=1000,
            collision_radius_tiles=0.5,
        )
    
    def _spawn_entity(self, entity_class, position: Position, player_id: int, card_stats: CardStatsCompat) -> Entity:
        """Spawn any type of entity"""
        # Use level-scaled stats for hitpoints and damage
        scaled_hp = card_stats.scaled_hitpoints or 100
        scaled_damage = card_stats.scaled_damage or 10

        entity = entity_class(
            id=self.next_entity_id,
            position=position,
            player_id=player_id,
            card_stats=card_stats,
            hitpoints=scaled_hp,
            max_hitpoints=scaled_hp,
            damage=scaled_damage,
            range=_safe_range(card_stats),
            sight_range=_safe_sight_range(card_stats)
        )

        entity.deploy_delay_remaining = max(0.0, (getattr(card_stats, "deploy_time", 0) or 0) / 1000.0)
        entity.attack_cooldown = max(entity.attack_cooldown, (getattr(card_stats, "load_time", 0) or 0) / 1000.0)

        # Add battle_state reference for mechanics
        entity.battle_state = self

        # Attach mechanics if using compat wrapper with card_definition
        if hasattr(card_stats, 'card_definition') and getattr(card_stats, 'card_definition'):
            entity.mechanics = [copy.deepcopy(m) for m in card_stats.card_definition.mechanics]
            for mech in entity.mechanics:
                mech.on_attach(entity)
            if entity.mechanics:
                print(f"[Attach] {getattr(card_stats, 'name', 'Building')}: {len(entity.mechanics)} mechanic(s)")

        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1

        if isinstance(entity, Building) and getattr(card_stats, "name", "") == KING_TOWER:
            entity._tower_active = False
            entity._is_king_tower = True
        elif isinstance(entity, Building):
            entity._tower_active = True
            entity._is_king_tower = False

        # Call on_spawn for all mechanics
        entity.on_spawn()
        return entity
    
    def _cleanup_dead_entities(self) -> None:
        """Remove dead entities from the game and handle death spawns"""
        dead_ids = [eid for eid, entity in self.entities.items() if not entity.is_alive]
        
        # Handle death spawns before removing entities (skip if handled by mechanics)
        for eid in dead_ids:
            entity = self.entities[eid]
            if isinstance(entity, Troop) and getattr(entity.card_stats, 'death_spawn_character', None):
                has_mechanic_spawn = any(isinstance(m, DeathSpawn) for m in getattr(entity, 'mechanics', []))
                if not has_mechanic_spawn:
                    self._spawn_death_units(entity)
        
        # Update player state for dead towers before removing entities
        for eid in dead_ids:
            entity = self.entities[eid]
            if isinstance(entity, Building):
                player = self.players[entity.player_id]
                pos = entity.position
                
                # Set tower HP to 0 when entity dies
                if entity.player_id == 0:  # Blue player
                    if (pos.x == self.arena.BLUE_KING_TOWER.x and 
                        pos.y == self.arena.BLUE_KING_TOWER.y):
                        player.king_tower_hp = 0
                    elif (pos.x == self.arena.BLUE_LEFT_TOWER.x and 
                          pos.y == self.arena.BLUE_LEFT_TOWER.y):
                        player.left_tower_hp = 0
                        self._activate_king_tower(0)
                    elif (pos.x == self.arena.BLUE_RIGHT_TOWER.x and 
                          pos.y == self.arena.BLUE_RIGHT_TOWER.y):
                        player.right_tower_hp = 0
                        self._activate_king_tower(0)
                else:  # Red player
                    if (pos.x == self.arena.RED_KING_TOWER.x and 
                        pos.y == self.arena.RED_KING_TOWER.y):
                        player.king_tower_hp = 0
                    elif (pos.x == self.arena.RED_LEFT_TOWER.x and 
                          pos.y == self.arena.RED_LEFT_TOWER.y):
                        player.left_tower_hp = 0
                        self._activate_king_tower(1)
                    elif (pos.x == self.arena.RED_RIGHT_TOWER.x and 
                          pos.y == self.arena.RED_RIGHT_TOWER.y):
                        player.right_tower_hp = 0
                        self._activate_king_tower(1)
        
        # Remove dead entities
        for eid in dead_ids:
            del self.entities[eid]
    
    def _spawn_death_units(self, troop: Troop) -> None:
        """Spawn death units when a troop dies"""
        death_spawn_name = troop.card_stats.death_spawn_character
        death_spawn_count = troop.card_stats.death_spawn_count or 1
        
        # Get death spawn card stats using the factory-driven loader
        death_spawn_stats = self.card_loader.get_card(death_spawn_name)

        if not death_spawn_stats and getattr(troop.card_stats, 'death_spawn_character_data', None):
            death_spawn_stats = troop_from_character_data(
                death_spawn_name,
                troop.card_stats.death_spawn_character_data,
                elixir=0,
                rarity="Common",
            )

        if not death_spawn_stats:
            death_spawn_stats = troop_from_values(
                death_spawn_name,
                hitpoints=100,
                damage=25,
                speed_tiles_per_min=60.0,
                range_tiles=1.0,
                sight_range_tiles=5.0,
                hit_speed_ms=1000,
                collision_radius_tiles=0.5,
            )
        
        # Spawn multiple units in a small radius around the death position
        spawn_radius = 0.5  # tiles
        
        for _ in range(death_spawn_count):
            # Random position around the death location
            angle = random.random() * 2 * 3.14159
            distance = random.random() * spawn_radius
            spawn_x = troop.position.x + distance * math.cos(angle)
            spawn_y = troop.position.y + distance * math.sin(angle)
            
            # Create and spawn the death unit
            self._spawn_troop(Position(spawn_x, spawn_y), troop.player_id, death_spawn_stats)
    
    def _check_win_conditions(self) -> None:
        """Check if game should end"""
        # Update player tower HP from entities
        self._update_tower_hp()
        
        # Check if king towers are destroyed
        for i, player in enumerate(self.players):
            if not player.is_alive():
                self.game_over = True
                self.winner = 1 - i
                return
        
        player0_crowns = self.crowns(0)
        player1_crowns = self.crowns(1)

        # End of 5:00 match timer. If tied, enter sudden death.
        if self.time >= self.sudden_death_start_time and not self.sudden_death:
            if player0_crowns != player1_crowns:
                self.game_over = True
                self.winner = 0 if player0_crowns > player1_crowns else 1
                return
            self.sudden_death = True
            self._sudden_death_crowns = (player0_crowns, player1_crowns)

        # Sudden death: first crown advantage wins instantly.
        if self.sudden_death:
            if player0_crowns != player1_crowns:
                self.game_over = True
                self.winner = 0 if player0_crowns > player1_crowns else 1
                return

            # Long-running tie fallback: tiebreaker by total tower HP damage dealt.
            if self.time >= self.tiebreaker_time:
                p0_damage = self._tower_damage_dealt_by_player(0)
                p1_damage = self._tower_damage_dealt_by_player(1)
                if p0_damage > p1_damage:
                    self.winner = 0
                elif p1_damage > p0_damage:
                    self.winner = 1
                else:
                    self.winner = None
                self.game_over = True
    
    def _update_tower_hp(self) -> None:
        """Update player tower HP from building entities"""
        for entity in self.entities.values():
            if isinstance(entity, Building):
                player = self.players[entity.player_id]
                pos = entity.position
                
                # Update HP based on tower position (compare coordinates)
                if entity.player_id == 0:  # Blue player
                    if (pos.x == self.arena.BLUE_KING_TOWER.x and 
                        pos.y == self.arena.BLUE_KING_TOWER.y):
                        player.king_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.BLUE_LEFT_TOWER.x and 
                          pos.y == self.arena.BLUE_LEFT_TOWER.y):
                        player.left_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.BLUE_RIGHT_TOWER.x and 
                          pos.y == self.arena.BLUE_RIGHT_TOWER.y):
                        player.right_tower_hp = entity.hitpoints
                else:  # Red player
                    if (pos.x == self.arena.RED_KING_TOWER.x and 
                        pos.y == self.arena.RED_KING_TOWER.y):
                        player.king_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.RED_LEFT_TOWER.x and 
                          pos.y == self.arena.RED_LEFT_TOWER.y):
                        player.left_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.RED_RIGHT_TOWER.x and 
                          pos.y == self.arena.RED_RIGHT_TOWER.y):
                        player.right_tower_hp = entity.hitpoints
    
    def get_state_summary(self) -> Dict:
        """Get current battle state summary"""
        return {
            "time": self.time,
            "tick": self.tick,
            "entities": len(self.entities),
            "players": [
                {
                    "elixir": p.elixir,
                    "crowns": self.crowns(p.player_id),
                    "king_hp": p.king_tower_hp,
                    "left_hp": p.left_tower_hp,
                    "right_hp": p.right_tower_hp,
                    "next_card": p.get_next_card(),
                }
                for p in self.players
            ],
            "game_over": self.game_over,
            "winner": self.winner
        }

    def crowns(self, player_id: int) -> int:
        """Crowns ``player_id`` has won (enemy towers destroyed)."""
        return self.players[1 - player_id].crowns_conceded()

    def _get_king_tower_entity(self, player_id: int) -> Optional[Building]:
        king_pos = self.arena.BLUE_KING_TOWER if player_id == 0 else self.arena.RED_KING_TOWER
        for entity in self.entities.values():
            if isinstance(entity, Building) and entity.player_id == player_id:
                if entity.position.x == king_pos.x and entity.position.y == king_pos.y:
                    return entity
        return None

    def _activate_king_tower(self, player_id: int) -> None:
        king = self._get_king_tower_entity(player_id)
        if king is not None:
            king._tower_active = True

    def _tower_damage_dealt_by_player(self, player_id: int) -> float:
        enemy_id = 1 - player_id
        current_enemy_hp = (
            self.players[enemy_id].king_tower_hp
            + self.players[enemy_id].left_tower_hp
            + self.players[enemy_id].right_tower_hp
        )
        start_enemy_hp = self._starting_total_tower_hp.get(enemy_id, current_enemy_hp)
        return max(0.0, start_enemy_hp - current_enemy_hp)

    def is_position_occupied_by_building(
        self,
        position: Position,
        mover_radius: float = 0.5,
        ignore_building_id: Optional[int] = None,
    ) -> bool:
        """Return True when a position overlaps any live building footprint."""
        for entity in self.entities.values():
            if not isinstance(entity, Building) or not entity.is_alive:
                continue
            if ignore_building_id is not None and entity.id == ignore_building_id:
                continue
            building_radius = getattr(entity.card_stats, "collision_radius", 1.0) or 1.0
            if position.distance_to(entity.position) < (building_radius + mover_radius) * 0.95:
                return True
        return False

    def _building_footprint_size_tiles(self, card_stats: CardStatsCompat) -> int:
        """Return placement footprint in board tiles."""
        name = getattr(card_stats, "name", "")
        if name in CROWN_TOWER_NAMES:
            return tower_footprint_size(name)
        # Clash Royale exception: Tesla has a smaller footprint than most buildings.
        if name == "Tesla":
            return 2
        return 3

    def _footprint_bounds(
        self,
        position: Position,
        card_stats: CardStatsCompat,
    ) -> tuple[float, float, float, float]:
        size = float(self._building_footprint_size_tiles(card_stats))
        half = size / 2.0
        return (position.x - half, position.x + half, position.y - half, position.y + half)

    def is_building_placement_occupied(
        self,
        position: Position,
        card_stats: CardStatsCompat,
    ) -> bool:
        """Return True when a new building footprint overlaps any live building footprint."""
        x1, x2, y1, y2 = self._footprint_bounds(position, card_stats)
        for entity in self.entities.values():
            if not isinstance(entity, Building) or not entity.is_alive:
                continue
            ex1, ex2, ey1, ey2 = self._footprint_bounds(entity.position, entity.card_stats)
            overlap_x = x1 < ex2 and x2 > ex1
            overlap_y = y1 < ey2 and y2 > ey1
            if overlap_x and overlap_y:
                return True
        return False

    def is_ground_position_walkable(
        self,
        position: Position,
        mover: Optional[Entity] = None,
    ) -> bool:
        """Ground movement validator including arena terrain and building footprints."""
        if not self.arena.is_walkable(position):
            return False
        mover_radius = 0.5
        if mover is not None:
            mover_radius = getattr(mover.card_stats, "collision_radius", 0.5) or 0.5
        return not self.is_position_occupied_by_building(position, mover_radius)

    def _resolve_troop_collisions(self) -> None:
        """Simple separation pass to reduce troop stacking."""
        troops = [e for e in self.entities.values() if isinstance(e, Troop) and e.is_alive]
        for i in range(len(troops)):
            a = troops[i]
            if getattr(a, "is_air_unit", False):
                continue
            ra = max(0.2, getattr(a.card_stats, "collision_radius", 0.5) or 0.5)
            for j in range(i + 1, len(troops)):
                b = troops[j]
                if getattr(b, "is_air_unit", False):
                    continue
                rb = max(0.2, getattr(b.card_stats, "collision_radius", 0.5) or 0.5)
                dx = b.position.x - a.position.x
                dy = b.position.y - a.position.y
                dist = math.hypot(dx, dy)
                same_team = a.player_id == b.player_id
                min_dist = (ra + rb) * (0.8 if same_team else 0.65)
                if dist == 0:
                    dist = 0.001
                    dx, dy = 0.001, 0.0
                if dist < min_dist:
                    overlap = (min_dist - dist) / 2.0
                    ux, uy = dx / dist, dy / dist
                    new_a = Position(a.position.x - ux * overlap, a.position.y - uy * overlap)
                    new_b = Position(b.position.x + ux * overlap, b.position.y + uy * overlap)
                    if self.is_ground_position_walkable(new_a, a):
                        a.position = new_a
                    if self.is_ground_position_walkable(new_b, b):
                        b.position = new_b
