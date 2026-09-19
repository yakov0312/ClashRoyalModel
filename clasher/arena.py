from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass 
class Position:
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class TileGrid:
    width: int = 18  # 18 tiles wide (x-axis)
    height: int = 32  # 32 tiles tall (y-axis) 
    tile_size: float = 100.0  
    
    # Tower centres on the 18x32 tile board. Footprints (Towers.json ``size``):
    # King 4x4 on tiles x7-10 / y1-4, leaving back row 0 free to deploy behind
    # it; Princess 3x3 on tiles x2-4 (x13-15) / y5-7. Red mirrors blue.
    # Player 0 (bottom)
    BLUE_KING_TOWER = Position(9.0, 3.0)
    BLUE_LEFT_TOWER = Position(3.5, 6.5)
    BLUE_RIGHT_TOWER = Position(14.5, 6.5)

    # Player 1 (top)
    RED_KING_TOWER = Position(9.0, 29.0)
    RED_LEFT_TOWER = Position(3.5, 25.5)
    RED_RIGHT_TOWER = Position(14.5, 25.5)
    
    # Bridge positions across river (3 tiles wide each)
    LEFT_BRIDGE = Position(3.5, 16.0)   # Left bridge center of center tile (tiles 2,3,4 -> center at 3.5)
    RIGHT_BRIDGE = Position(14.5, 16.0) # Right bridge center of center tile (tiles 13,14,15 -> center at 14.5)
    
    # River covers tile rows 15 and 16, i.e. 15.0 <= y < 17.0. (RIVER_Y2 is the
    # last river *row*; deploy zones start at RIVER_Y2 + 1.)
    RIVER_Y1 = 15.0
    RIVER_Y2 = 16.0
    
    # Blocked tiles (unplayable and unwalkable areas) - these appear as gray fences/barriers
    # Pattern from real Clash Royale: 6 gray, 6 green, 6 gray on top/bottom rows
    BLOCKED_TILES = [
        # Edge tiles next to river
        (0, 14),   # Left edge of arena, bottom land next to river
        (0, 17),   # Left edge of arena, top land next to river
        (17, 14),  # Right edge of arena, bottom land next to river  
        (17, 17),  # Right edge of arena, top land next to river
        
        # Top row (y=0): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),           # Left 6 gray fence tiles
        (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0),     # Right 6 gray fence tiles
        
        # Bottom row (y=31): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 31), (1, 31), (2, 31), (3, 31), (4, 31), (5, 31),     # Left 6 gray fence tiles
        (12, 31), (13, 31), (14, 31), (15, 31), (16, 31), (17, 31), # Right 6 gray fence tiles
    ]
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within arena bounds"""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_blocked_tile(self, x: int, y: int) -> bool:
        """Check if a tile coordinate is blocked (unplayable)"""
        return (x, y) in self.BLOCKED_TILES
    
    def is_walkable(self, pos: Position) -> bool:
        """Check if a position is walkable (not river, not blocked tiles)"""
        # Check bounds
        if not self.is_valid_position(pos):
            return False
        
        # Check if it's a blocked tile
        tile_x, tile_y = int(pos.x), int(pos.y)
        if self.is_blocked_tile(tile_x, tile_y):
            return False
        
        # Check if it's in the river (rows 15-16), unless it's on a bridge
        if self.RIVER_Y1 <= pos.y < self.RIVER_Y2 + 1.0:
            # Check if on bridge (3 tiles wide each, allowing fractional positions within tiles)
            on_left_bridge = 2.0 <= pos.x < 5.0   # Left bridge spans tiles 2,3,4 (x=2.0 to x=4.999...)
            on_right_bridge = 13.0 <= pos.x < 16.0  # Right bridge spans tiles 13,14,15 (x=13.0 to x=15.999...)
            return on_left_bridge or on_right_bridge
        
        return True
    
    def get_deploy_zones(self, player_id: int, battle_state=None) -> List[Tuple[float, float, float, float]]:
        """Get valid deployment zones for a player (x1, y1, x2, y2)
        Expands to include bridge areas and 4 tiles back when towers are destroyed"""
        zones = []
        
        if player_id == 0:  # Player 0 (bottom half)
            # Basic deployment zone (bottom half, excluding river)
            zones.append((0, 1, self.width, self.RIVER_Y1))  # y=1 to y=14
            
            # Always include the 6 blocks behind own king tower (including row 0)
            zones.append((6, 0, 12, 6))  # Behind blue king: x=6-11, y=0-5 (6 tiles behind own king, including edge row)
            
            # Check for expanded zones based on destroyed enemy towers
            if battle_state:
                # If red left tower is destroyed, blue player can spawn on left half of arena and 4 tiles back
                if battle_state.players[1].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y2 + 1, 9, self.RIVER_Y2 + 5))  # Left half: x=0-8, y=17-20
                
                # If red right tower is destroyed, blue player can spawn on right half of arena and 4 tiles back  
                if battle_state.players[1].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y2 + 1, self.width, self.RIVER_Y2 + 5))  # Right half: x=9-17, y=17-20
                    
        else:  # Player 1 (top half)  
            # Basic deployment zone (top half, excluding river)
            zones.append((0, self.RIVER_Y2 + 1, self.width, 31))  # y=17 to y=30
            
            # Always include the 6 blocks behind own king tower (including row 31)
            zones.append((6, 26, 12, 32))  # Behind red king: x=6-11, y=26-31 (6 tiles behind own king, including edge row)
            
            # Check for expanded zones based on destroyed enemy towers
            if battle_state:
                # If blue left tower is destroyed, red player can spawn on left half of arena and 4 tiles back
                if battle_state.players[0].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y1 - 4, 9, self.RIVER_Y1))  # Left half: x=0-8, y=11-14
                
                # If blue right tower is destroyed, red player can spawn on right half of arena and 4 tiles back
                if battle_state.players[0].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y1 - 4, self.width, self.RIVER_Y1))  # Right half: x=9-17, y=11-14
        
        return zones
    
    def can_deploy_at(self, pos: Position, player_id: int, battle_state=None, is_spell=False, spell_obj=None) -> bool:
        """Check if position is valid for deployment"""
        # Check basic bounds
        if not self.is_valid_position(pos):
            return False
        
        # Check if this is a blocked tile (gray, unplayable)
        tile_pos = (int(pos.x), int(pos.y))
        if tile_pos in self.BLOCKED_TILES:
            return False
        
        # Check if position overlaps with living tower area (troops cannot deploy on towers)
        if not is_spell and self.is_tower_tile(pos, battle_state):
            return False
        
        # Most spells can be deployed anywhere on the battlefield
        # However, rolling projectiles (Log, Barbarian Barrel) follow troop deployment rules
        if is_spell:
            # Check if this is a rolling projectile spell that requires deployment territory validation
            if self._is_rolling_projectile_spell(spell_obj):
                # Rolling projectiles must follow troop deployment rules
                pass  # Continue to deployment zone validation below
            else:
                # Regular spells can be deployed anywhere
                return True
        
        # Special restriction: only middle 6 tiles (x=6-11) playable on rows 0 and 31
        if pos.y == 0 or pos.y == 31:
            if not (6 <= pos.x <= 11):
                return False
        
        # Check deployment zones (including expanded zones after tower destruction)
        zones = self.get_deploy_zones(player_id, battle_state)
        for x1, y1, x2, y2 in zones:
            if x1 <= pos.x < x2 and y1 <= pos.y < y2:
                return True
        
        # Special case: allow middle 6 tiles on edge rows for respective players
        if player_id == 0 and pos.y == 0 and 6 <= pos.x <= 11:
            return True
        elif player_id == 1 and pos.y == 31 and 6 <= pos.x <= 11:
            return True
            
        return False
    
    def _is_rolling_projectile_spell(self, spell_obj) -> bool:
        """Check if spell is a rolling projectile that requires territory validation"""
        if not spell_obj:
            return False
        
        # Import here to avoid circular imports
        from .spells import RollingProjectileSpell
        return isinstance(spell_obj, RollingProjectileSpell)
    
    def world_to_tile(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to tile coordinates"""
        return int(x / self.tile_size), int(y / self.tile_size)
    
    def tile_to_world(self, tile_x: int, tile_y: int) -> Position:
        """Convert tile coordinates to world position"""
        return Position(
            tile_x * self.tile_size + self.tile_size / 2,
            tile_y * self.tile_size + self.tile_size / 2
        )
    
    def tower_footprints(self):
        """(centre, half_size, player_id) for every crown tower."""
        from .towers import KING_TOWER, PRINCESS_TOWER, tower_footprint_size
        king = tower_footprint_size(KING_TOWER) / 2.0
        princess = tower_footprint_size(PRINCESS_TOWER) / 2.0
        return [
            (self.BLUE_LEFT_TOWER, princess, 0),
            (self.BLUE_RIGHT_TOWER, princess, 0),
            (self.BLUE_KING_TOWER, king, 0),
            (self.RED_LEFT_TOWER, princess, 1),
            (self.RED_RIGHT_TOWER, princess, 1),
            (self.RED_KING_TOWER, king, 1),
        ]

    def is_tower_tile(self, pos: Position, battle_state=None) -> bool:
        """True when ``pos`` lies inside a living crown tower's footprint.

        Footprints are half-open squares on tile boundaries, so the tile right
        behind the King Tower (row 0 / row 31) is free.
        """
        for tower_pos, half, player_id in self.tower_footprints():
            if battle_state and not self._is_tower_alive(tower_pos, player_id, battle_state):
                continue
            if (tower_pos.x - half <= pos.x < tower_pos.x + half
                    and tower_pos.y - half <= pos.y < tower_pos.y + half):
                return True
        return False

    def _is_tower_alive(self, tower_pos: Position, player_id: int, battle_state) -> bool:
        """Check if tower at given position is still alive"""
        if not battle_state:
            return True  # Assume alive if no battle state
        
        # Find tower entity at this position
        for entity in battle_state.entities.values():
            if (hasattr(entity, 'position') and 
                entity.position.x == tower_pos.x and 
                entity.position.y == tower_pos.y and
                getattr(entity, 'player_id', -1) == player_id):
                return entity.is_alive
        
        return False  # Tower not found, assume dead
    
    def get_tower_blocked_x_ranges(self, y: float, battle_state=None) -> List[Tuple[float, float]]:
        """X ranges covered by living crown-tower footprints on row ``y``."""
        blocked_ranges = []
        for tower_pos, half, player_id in self.tower_footprints():
            if battle_state and not self._is_tower_alive(tower_pos, player_id, battle_state):
                continue
            if tower_pos.y - half <= y < tower_pos.y + half:
                blocked_ranges.append((tower_pos.x - half, tower_pos.x + half))
        return blocked_ranges
