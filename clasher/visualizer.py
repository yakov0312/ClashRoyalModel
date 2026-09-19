#!/usr/bin/env python3
"""
Real-time Battle Visualization

Shows the battlefield with:
- Arena layout with river and towers
- Unit positions and movement (now using real card art)
- Health bars and stats
- Real-time battle progression
"""

import pygame
import pygame.gfxdraw
import sys
import time
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple
from clasher.engine import BattleEngine
from clasher.arena import Position
from clasher.card_art import CARD_IMAGE_DIR as DEFAULT_CARD_IMAGE_DIR, art_name
from clasher.towers import CROWN_TOWER_NAMES, KING_TOWER, tower_footprint_size
from clasher import render_textures as tex

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
GREEN = (100, 255, 100)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
YELLOW = (255, 255, 100)
PURPLE = (255, 100, 255)
CYAN = (100, 255, 255)
ORANGE = (255, 165, 0)

# --- Clash-Royale-ish palette (used by the redesigned chrome/UI) ---
UI_BG = (22, 26, 38)            # dark slate panel background
UI_BG_LIGHT = (32, 38, 54)
UI_BORDER = (10, 12, 18)
UI_GOLD = (255, 205, 90)
UI_GOLD_DARK = (150, 110, 30)
UI_TEXT = (235, 235, 240)
UI_TEXT_DIM = (165, 170, 185)
ELIXIR_PURPLE = (188, 64, 214)
ELIXIR_PURPLE_LIGHT = (224, 130, 240)
ELIXIR_EMPTY = (46, 30, 55)
GRASS_A = (94, 176, 74)
GRASS_B = (86, 166, 68)
RIVER_TOP = (92, 200, 235)
RIVER_BOTTOM = (46, 140, 190)
SHADOW_COLOR = (0, 0, 0, 90)

# Screen settings
SCREEN_WIDTH = 1200
# Make tiles square: 18 wide × 32 tall
TILE_SIZE = 22  # Square tiles
ARENA_WIDTH = 18 * TILE_SIZE   # 396 pixels
ARENA_HEIGHT = 32 * TILE_SIZE  # 704 pixels
ARENA_X = 50

# Layout, top to bottom: enemy (P1/red) hand bar -> arena (P1 side on top,
# P0 side on bottom) -> P0 (blue) hand bar. Blue is always the bottom player.
HAND_BAR_HEIGHT = 96
TOP_HAND_Y = 10
HAND_ARENA_GAP = 30

ARENA_Y = TOP_HAND_Y + HAND_BAR_HEIGHT + HAND_ARENA_GAP
BOTTOM_HAND_Y = ARENA_Y + ARENA_HEIGHT + HAND_ARENA_GAP
SCREEN_HEIGHT = BOTTOM_HAND_Y + HAND_BAR_HEIGHT + 10

# Total number of tile-rows in the arena (used to flip world-y -> screen-y so
# player 0, whose towers sit at low world-y, always renders at the bottom).
ARENA_ROWS = 32

# Card art directory - one PNG per Cards.json card (Knight.png, Archers.png).
# Sub-units show the art of the card that deploys them (see clasher/card_art.py).
# Override with the CARD_IMAGE_DIR env var if needed.
CARD_IMAGE_DIR = Path(os.environ.get("CARD_IMAGE_DIR", str(DEFAULT_CARD_IMAGE_DIR)))


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def draw_aa_filled_circle(surface, center, radius, color):
    """Anti-aliased filled circle. Falls back gracefully for radius <= 0."""
    if radius <= 0:
        return
    x, y = int(center[0]), int(center[1])
    r = int(radius)
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def draw_aa_circle_outline(surface, center, radius, color, width=2):
    """Anti-aliased ring. Emulated by stacking aacircle outlines since
    gfxdraw only draws 1px rings natively."""
    if radius <= 0:
        return
    x, y = int(center[0]), int(center[1])
    r = int(radius)
    for i in range(max(1, width)):
        pygame.gfxdraw.aacircle(surface, x, y, max(1, r - i), color)


def draw_soft_shadow(surface, center_x, bottom_y, radius_x, radius_y=None, alpha=90):
    """Draws a soft blurred-looking ellipse shadow under an entity using a
    few stacked translucent ellipses (cheap fake blur, no per-frame surface
    allocation cost worth mentioning at this scale)."""
    radius_y = radius_y if radius_y is not None else max(3, int(radius_x * 0.4))
    layers = 3
    shadow_surf = pygame.Surface((radius_x * 2 + 4, radius_y * 2 + 4), pygame.SRCALPHA)
    cx, cy = shadow_surf.get_width() // 2, shadow_surf.get_height() // 2
    for i in range(layers, 0, -1):
        t = i / layers
        rx = int(radius_x * t)
        ry = int(radius_y * t)
        a = int(alpha * (1.0 - t) + alpha * 0.35)
        if rx > 0 and ry > 0:
            pygame.draw.ellipse(
                shadow_surf, (0, 0, 0, a),
                pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2)
            )
    surface.blit(shadow_surf, (center_x - cx, bottom_y - cy))


def draw_rounded_panel(surface, rect, radius, fill=None, border=None, border_width=2):
    if fill is not None:
        pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, border_width, border_radius=radius)


class BattleVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clash Royale Battle Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        self.elixir_font = pygame.font.Font(None, 40)

        # Per-tower wake-up time, for the King activation flash.
        self._king_awake_since: Dict[int, float] = {}

        # Card art cache: name -> raw loaded pygame.Surface (unscaled).
        self._card_image_cache: Dict[str, "pygame.Surface | None"] = {}
        # Cache of pre-scaled art keyed by (name, size) so we never call
        # smoothscale more than once per (card, pixel-size) combination -
        # this was the main hidden cost in the old per-frame rescale.
        self._scaled_card_cache: Dict[Tuple[str, int], "pygame.Surface | None"] = {}
        # Pre-rendered background surfaces, built once and blitted every frame.
        self._arena_bg_surface: "pygame.Surface | None" = None
        self._backdrop_surface: "pygame.Surface | None" = None

        if not CARD_IMAGE_DIR.exists():
            print(f"Warning: CARD_IMAGE_DIR does not exist: {CARD_IMAGE_DIR}")
            print("Falling back to shape-based rendering. Set CARD_IMAGE_DIR env var to fix.")

        # Battle setup
        self.engine = BattleEngine()
        self.battle = self.engine.create_battle()

        # Visualization settings - square tiles
        self.tile_size = TILE_SIZE  # Square tiles
        self.tile_width = TILE_SIZE
        self.tile_height = TILE_SIZE

        # Auto-deploy for testing
        self.setup_test_battle()

    # ------------------------------------------------------------------
    # Art loading / caching
    # ------------------------------------------------------------------

    def get_card_image(self, name: str) -> "pygame.Surface | None":
        """Load (and cache) a card's PNG art. Returns None if missing."""
        if name in self._card_image_cache:
            return self._card_image_cache[name]

        lookup_name = art_name(name, str(CARD_IMAGE_DIR))
        path = CARD_IMAGE_DIR / f"{lookup_name or name}.png"
        image = None
        if lookup_name is not None:
            try:
                image = pygame.image.load(str(path)).convert_alpha()
            except pygame.error as exc:
                print(f"Warning: failed to load image for '{name}': {exc}")
                image = None
        else:
            # Only warn once per missing name so this doesn't spam every frame.
            print(f"Warning: no image found for '{name}' at {path}")

        self._card_image_cache[name] = image
        return image

    def get_scaled_card_image(self, name: str, diameter: int) -> "pygame.Surface | None":
        """Like get_card_image but returns art pre-scaled to fit within a
        diameter x diameter box, cached per (name, diameter) so repeated
        draws of the same unit at the same zoom never rescale twice."""
        key = (name, diameter)
        if key in self._scaled_card_cache:
            return self._scaled_card_cache[key]

        image = self.get_card_image(name)
        if image is None:
            self._scaled_card_cache[key] = None
            return None

        img_w, img_h = image.get_size()
        scale = diameter / max(img_w, img_h)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        scaled = pygame.transform.smoothscale(image, new_size)
        self._scaled_card_cache[key] = scaled
        return scaled

    def draw_card_image(self, name: str, center_x: int, center_y: int, diameter: int) -> bool:
        """Draw a card image centered at (center_x, center_y), scaled to fit
        within a diameter x diameter box (aspect ratio preserved). Returns
        True if an image was drawn, False if no image was available."""
        scaled = self.get_scaled_card_image(name, diameter)
        if scaled is None:
            return False
        rect = scaled.get_rect(center=(center_x, center_y))
        self.screen.blit(scaled, rect)
        return True

    def setup_test_battle(self):
        """Setup a test battle with units"""
        print("Setting up test battle with corrected arena dimensions...")

        # Give players more elixir for testing
        self.battle.players[0].elixir = 10.0
        self.battle.players[1].elixir = 10.0

        # Deploy some units for visualization using corrected positions
        deployments = [
            # Player 0 (Blue - bottom half, y < 15)
            (0, 'Knight', Position(8.5, 12), "P0 Knight center"),
            (0, 'Archers', Position(4, 8), "P0 Archers left"),
            (0, 'Archers', Position(13, 8), "P0 Archers right"),

            # Player 1 (Red - top half, y > 16)
            (1, 'Knight', Position(8.5, 20), "P1 Knight center"),
            (1, 'Archers', Position(4, 24), "P1 Archers left"),
            (1, 'Archers', Position(13, 24), "P1 Archers right"),
        ]

        for player_id, card, pos, desc in deployments:
            result = self.battle.deploy_card(player_id, card, pos)
            print(f"  {desc}: {result}")
            # Reset elixir for testing
            self.battle.players[player_id].elixir = 10.0

        # Enable coordinate display for debugging
        self.show_coords = True

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.

        The y-axis is flipped: player 0's towers sit at low world-y, and we
        always want player 0 (blue) rendered at the BOTTOM of the screen, so
        low world-y maps to high screen-y.
        """
        screen_x = int(ARENA_X + x * self.tile_size)
        screen_y = int(ARENA_Y + (ARENA_ROWS - y) * self.tile_size)
        return screen_x, screen_y

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------

    def _build_arena_background(self):
        """Pre-render the static arena once (stone floor, river, bridges,
        fence blocks); every frame just blits it. Geometry comes from the
        simulator's TileGrid so the picture always matches the rules."""
        arena = self.battle.arena
        t = self.tile_size
        surf = tex.build_floor(arena.width, arena.height, t)

        def row_top(world_row_end: float) -> int:
            # screen y of the world row boundary (y axis flipped, blue at bottom)
            return int((ARENA_ROWS - world_row_end) * t)

        river_top = row_top(arena.RIVER_Y2 + 1.0)
        river_h = int((arena.RIVER_Y2 + 1.0 - arena.RIVER_Y1) * t)
        surf.blit(tex.build_river(ARENA_WIDTH, river_h), (0, river_top))

        # Bridges: every walkable river column block (from TileGrid.is_walkable).
        mid_y = (arena.RIVER_Y1 + arena.RIVER_Y2 + 1.0) / 2.0
        x = 0
        while x < arena.width:
            if arena.is_walkable(Position(x + 0.5, mid_y)):
                start = x
                while x < arena.width and arena.is_walkable(Position(x + 0.5, mid_y)):
                    x += 1
                bridge = tex.build_bridge((x - start) * t, river_h + t // 2, t)
                surf.blit(bridge, (start * t, river_top - t // 4))
            x += 1

        wall = tex.build_wall_block(t)
        for (bx, by) in arena.BLOCKED_TILES:
            surf.blit(wall, (bx * t, (ARENA_ROWS - by - 1) * t))

        # Soft vignette so the board edges read as walls.
        vignette = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)
        for i in range(10):
            pygame.draw.rect(vignette, (0, 0, 0, 10), vignette.get_rect().inflate(-i * 2, -i * 2), 2)
        surf.blit(vignette, (0, 0))
        pygame.draw.rect(surf, UI_BORDER, surf.get_rect(), 3)
        self._arena_bg_surface = surf

    def _build_backdrop(self):
        """A soft vertical gradient behind everything instead of flat white,
        so the arena reads as sitting 'on a table' rather than floating on
        a blank page."""
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        top = (26, 30, 42)
        bottom = (14, 16, 24)
        for y in range(SCREEN_HEIGHT):
            t = y / max(1, SCREEN_HEIGHT - 1)
            pygame.draw.line(surf, _lerp_color(top, bottom, t), (0, y), (SCREEN_WIDTH, y))
        self._backdrop_surface = surf

    def draw_backdrop(self):
        if self._backdrop_surface is None:
            self._build_backdrop()
        self.screen.blit(self._backdrop_surface, (0, 0))

    def draw_arena(self):
        """Blit the pre-rendered arena, then draw dynamic per-frame overlays
        (deployment zone tints, blocked tiles, coordinate labels) on top."""
        if self._arena_bg_surface is None:
            self._build_arena_background()

        self.screen.blit(self._arena_bg_surface, (ARENA_X, ARENA_Y))
        pygame.draw.rect(self.screen, UI_BORDER, (ARENA_X, ARENA_Y, ARENA_WIDTH, ARENA_HEIGHT), 3)

        # Add deployment zone tints
        self.draw_deployment_zones()

        # Add coordinate labels for clarity
        if hasattr(self, 'show_coords') and self.show_coords:
            for y in range(0, 32, 4):
                coord_text = self.small_font.render(str(y), True, UI_TEXT_DIM)
                _, label_screen_y = self.world_to_screen(0, y)
                self.screen.blit(coord_text, (ARENA_X - 25, label_screen_y))

            for x in range(0, 18, 3):
                coord_text = self.small_font.render(str(x), True, UI_TEXT_DIM)
                self.screen.blit(coord_text, (ARENA_X + x * self.tile_size, ARENA_Y - 20))

    def draw_deployment_zones(self):
        """Draw colored tints for deployment zones"""
        if not hasattr(self, 'battle') or not self.battle:
            return

        # Get deployment zones for both players
        blue_zones = self.battle.arena.get_deploy_zones(0, self.battle)
        red_zones = self.battle.arena.get_deploy_zones(1, self.battle)

        # Create a grid to track which tiles belong to which player
        tile_ownership = {}  # (x, y) -> set of player_ids

        # Mark blue zones (excluding blocked tiles)
        for x1, y1, x2, y2 in blue_zones:
            for x in range(int(x1), int(x2)):
                for y in range(int(y1), int(y2)):
                    # Skip blocked tiles entirely - don't add them to tile_ownership
                    if self.battle.arena.is_blocked_tile(x, y):
                        continue
                    if (x, y) not in tile_ownership:
                        tile_ownership[(x, y)] = set()
                    tile_ownership[(x, y)].add(0)  # Blue player

        # Mark red zones (excluding blocked tiles)
        for x1, y1, x2, y2 in red_zones:
            for x in range(int(x1), int(x2)):
                for y in range(int(y1), int(y2)):
                    # Skip blocked tiles entirely - don't add them to tile_ownership
                    if self.battle.arena.is_blocked_tile(x, y):
                        continue
                    if (x, y) not in tile_ownership:
                        tile_ownership[(x, y)] = set()
                    tile_ownership[(x, y)].add(1)  # Red player

        # Draw tints based on ownership
        for (tile_x, tile_y), owners in tile_ownership.items():
            # Skip blocked tiles - they will be drawn gray instead
            if self.battle.arena.is_blocked_tile(tile_x, tile_y):
                continue

            # Skip if both players can deploy (no tint)
            if len(owners) > 1:
                continue

            # Determine tint color - same for all playable areas
            if 0 in owners:  # Blue only
                tint_color = (100, 150, 255, 30)
            elif 1 in owners:  # Red only
                tint_color = (255, 120, 100, 30)
            else:
                continue

            # Draw the tint - flip the row to match world_to_screen's y-flip
            # (row tile_y occupies world-y [tile_y, tile_y+1), which under the
            # flip maps to screen rows starting at ARENA_ROWS - tile_y - 1).
            screen_x = ARENA_X + tile_x * TILE_SIZE
            screen_y = ARENA_Y + (ARENA_ROWS - tile_y - 1) * TILE_SIZE

            tint_rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
            s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            s.fill(tint_color)
            self.screen.blit(s, tint_rect)

        # Draw blocked tiles in gray (no red/blue tints)
        for tile_x in range(18):  # Arena width
            for tile_y in range(32):  # Arena height
                if self.battle.arena.is_blocked_tile(tile_x, tile_y):
                    screen_x = ARENA_X + tile_x * TILE_SIZE
                    screen_y = ARENA_Y + (ARENA_ROWS - tile_y - 1) * TILE_SIZE

                    # Draw gray blocked tile
                    blocked_rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
                    s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    s.fill((128, 128, 128, 160))  # Semi-transparent gray
                    self.screen.blit(s, blocked_rect)

                    # Add darker border to make it more visible
                    pygame.draw.rect(self.screen, (64, 64, 64), blocked_rect, 1)

    def draw_towers(self):
        """Crown towers drawn from the live tower entities: square platforms
        of their real footprint, team colours, HP bars, and the King's
        sleeping / awake state (grey with "Zz" until activated, then a gold
        crown and a short activation flash)."""
        now = self.battle.time
        live = {}
        for entity in self.battle.entities.values():
            name = getattr(getattr(entity, "card_stats", None), "name", None)
            if name in CROWN_TOWER_NAMES and entity.is_alive:
                live[(entity.player_id, entity.position.x, entity.position.y)] = entity

        for tower_pos, half, player_id in self.battle.arena.tower_footprints():
            entity = live.get((player_id, tower_pos.x, tower_pos.y))
            is_king = half * 2 == tower_footprint_size(KING_TOWER)
            size_px = int(half * 2 * self.tile_size)
            left, top = self.world_to_screen(tower_pos.x - half, tower_pos.y + half)
            awake = True
            if entity is not None and is_king:
                awake = bool(getattr(entity, "_tower_active", True))
                if awake and entity.id not in self._king_awake_since:
                    self._king_awake_since[entity.id] = now
            sprite = tex.tower_sprite("king" if is_king else "princess", player_id, size_px,
                                      awake, entity is None)
            draw_soft_shadow(self.screen, left + size_px // 2, top + size_px, size_px // 2, alpha=70)
            self.screen.blit(sprite, (left, top))
            if entity is None:
                continue
            cx, cy = left + size_px // 2, top + size_px // 2

            if is_king and not awake:
                bob = int(3 * math.sin(now * 3.0))
                zz = self.small_font.render("Zz", True, WHITE)
                self._blit_outlined_text(zz, zz.get_rect(center=(cx + size_px // 3, top + 4 + bob)), "Zz", self.small_font, color=WHITE)
            elif is_king:
                since = now - self._king_awake_since.get(entity.id, -10.0)
                if 0 <= since < 1.2:  # activation flash
                    fade = 1.0 - since / 1.2
                    radius = int(size_px * (0.55 + 0.6 * (1 - fade)))
                    draw_aa_circle_outline(self.screen, (cx, cy), radius, (255, 214, 90), width=max(1, int(4 * fade)))

            # Muzzle flash right after a shot (sleeping Kings never fire).
            if awake and entity.target_id and getattr(entity, "last_attack_time", 99.0) < 0.12:
                draw_aa_filled_circle(self.screen, (cx, cy - size_px // 6), max(3, size_px // 8), (255, 240, 170))

            ratio = entity.hitpoints / entity.max_hitpoints if entity.max_hitpoints else 0.0
            bar_w = size_px
            bar_x = left
            bar_y = top - 10 if player_id == 1 else top + size_px + 4
            self.draw_health_bar(bar_x, bar_y, bar_w, 6, ratio)
            hp_label = f"{int(math.ceil(entity.hitpoints))}"
            hp_text = self.small_font.render(hp_label, True, UI_TEXT)
            hp_rect = hp_text.get_rect(midbottom=(cx, bar_y - 1)) if player_id == 1 else hp_text.get_rect(midtop=(cx, bar_y + 7))
            self._blit_outlined_text(hp_text, hp_rect, hp_label, self.small_font)

    def _blit_outlined_text(self, _unused_surface, rect, text, font, color=UI_TEXT, outline=BLACK):
        """Draws text with a 1px dark outline so labels stay readable over
        busy backgrounds/art instead of relying on flat white boxes."""
        base = font.render(text, True, color)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shadow = font.render(text, True, outline)
            self.screen.blit(shadow, (rect.x + dx, rect.y + dy))
        self.screen.blit(base, rect)

    def draw_health_bar(self, x, y, width, height, ratio, show_bg=True):
        """Rounded, gradient-shaded health bar instead of a flat rect."""
        ratio = max(0.0, min(1.0, ratio))
        rect = pygame.Rect(x, y, width, height)
        if show_bg:
            pygame.draw.rect(self.screen, (25, 25, 25), rect.inflate(2, 2), border_radius=height // 2 + 1)
        if ratio > 0.6:
            color = (90, 210, 90)
        elif ratio > 0.3:
            color = (235, 200, 60)
        else:
            color = (225, 70, 70)
        fill_rect = pygame.Rect(x, y, max(1, int(width * ratio)), height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=height // 2)
        pygame.draw.rect(self.screen, UI_BORDER, rect, 1, border_radius=height // 2 + 1)

    def _draw_projectile(self, entity, entity_name: str, screen_x: int, screen_y: int) -> None:
        """Arrows for Princess Towers, cannonballs for the King, glowing orbs
        with a short trail for everything else; oriented along the flight."""
        target = getattr(entity, "target_position", None)
        dx, dy = 0.0, -1.0
        if target is not None:
            tx, ty = self.world_to_screen(target.x, target.y)
            length = math.hypot(tx - screen_x, ty - screen_y)
            if length > 0:
                dx, dy = (tx - screen_x) / length, (ty - screen_y) / length
        team = tex.TEAM_COLORS[entity.player_id]
        if entity_name == "PrincessTower":
            tail = (screen_x - dx * 12, screen_y - dy * 12)
            pygame.draw.line(self.screen, (90, 60, 30), tail, (screen_x, screen_y), 2)
            head = [(screen_x + dx * 4, screen_y + dy * 4),
                    (screen_x - dy * 3, screen_y + dx * 3),
                    (screen_x + dy * 3, screen_y - dx * 3)]
            pygame.draw.polygon(self.screen, (225, 225, 230), head)
            return
        if entity_name == KING_TOWER:
            draw_aa_filled_circle(self.screen, (screen_x, screen_y), 5, (50, 50, 56))
            draw_aa_filled_circle(self.screen, (screen_x - 1, screen_y - 1), 2, (150, 150, 160))
            return
        for k in range(1, 4):
            draw_aa_filled_circle(self.screen, (int(screen_x - dx * 4 * k), int(screen_y - dy * 4 * k)),
                                  max(1, 4 - k), team[0])
        draw_aa_filled_circle(self.screen, (screen_x, screen_y), 4, (255, 245, 200))
        draw_aa_circle_outline(self.screen, (screen_x, screen_y), 4, team[2], width=1)

    def _draw_timed_explosive(self, entity, screen_x: int, screen_y: int) -> None:
        """Death bombs (Balloon, Giant Skeleton, Bomb Tower) and the Skeleton
        Barrel's falling container: their own sprite, a countdown ring and the
        blast radius, instead of the parent card's art."""
        remaining = max(0.0, entity.explosion_timer - entity.time_alive)
        fraction = remaining / entity.explosion_timer if entity.explosion_timer > 0 else 0.0
        size = int(self.tile_size * 1.1)
        container = bool(entity.death_spawn_name and entity.death_spawn_count)

        blast = int(entity.explosion_radius * self.tile_size)
        if blast > size // 2:
            draw_aa_circle_outline(self.screen, (screen_x, screen_y), blast, (255, 120, 60), width=1)
        draw_soft_shadow(self.screen, screen_x, screen_y + size // 3, size // 2, alpha=70)
        sprite = tex.barrel_sprite(size, entity.player_id) if container else tex.bomb_sprite(size, entity.player_id)
        self.screen.blit(sprite, sprite.get_rect(center=(screen_x, screen_y)))

        # Countdown ring shrinking clockwise, flashing red in the last 0.5s.
        ring = pygame.Rect(0, 0, size + 8, size + 8)
        ring.center = (screen_x, screen_y)
        urgent = remaining < 0.5 and int(self.battle.time * 10) % 2 == 0
        color = (255, 70, 50) if urgent else (255, 214, 90)
        if fraction > 0:
            pygame.draw.arc(self.screen, color, ring, math.pi / 2, math.pi / 2 + 2 * math.pi * fraction, 3)
        if not container:
            # lit fuse spark
            flicker = 2 + int(2 * abs(math.sin(self.battle.time * 25)))
            draw_aa_filled_circle(self.screen, (screen_x, screen_y - size // 2 - 1), flicker, (255, 200, 80))
        label = f"{remaining:.1f}"
        text = self.small_font.render(label, True, WHITE)
        self._blit_outlined_text(text, text.get_rect(midtop=(screen_x, screen_y + size // 2 + 4)), label, self.small_font, color=WHITE)

    def draw_entities(self):
        """Draw all entities (troops, projectiles, etc.)"""
        for entity in self.battle.entities.values():
            if not entity.is_alive:
                continue

            # Skip towers (drawn separately)
            if (hasattr(entity, 'card_stats') and entity.card_stats and
                entity.card_stats.name in CROWN_TOWER_NAMES):
                continue

            screen_x, screen_y = self.world_to_screen(entity.position.x, entity.position.y)

            # Determine color and shape based on player and type
            color = BLUE if entity.player_id == 0 else RED

            # Get entity type and handle special cases
            entity_name = "Unknown"
            entity_type = type(entity).__name__
            if hasattr(entity, 'card_stats') and entity.card_stats:
                entity_name = entity.card_stats.name

            # Special handling for spell entities
            if entity_type == "AreaEffect":
                area_radius = int(entity.radius * self.tile_size)
                draw_aa_circle_outline(self.screen, (screen_x, screen_y), area_radius, (235, 210, 60), width=2)
                time_ratio = 1.0 - (entity.time_alive / entity.duration) if entity.duration > 0 else 0
                timer_color = (255, int(255 * time_ratio), 0)
                draw_aa_filled_circle(self.screen, (screen_x, screen_y), 5, timer_color)
                spell_name = getattr(entity, 'spell_name', 'AREA')
                if not self.draw_card_image(spell_name, screen_x, screen_y - 15, int(self.tile_size * 1.4)):
                    text_surface = self.small_font.render(spell_name.upper(), True, UI_TEXT)
                    text_rect = text_surface.get_rect(center=(screen_x, screen_y - 15))
                    self.screen.blit(text_surface, text_rect)
                continue
            elif entity_type == "TimedExplosive":
                self._draw_timed_explosive(entity, screen_x, screen_y)
                continue
            elif entity_type in ["Projectile", "SpawnProjectile"]:
                self._draw_projectile(entity, entity_name, screen_x, screen_y)
                continue
            elif entity_type == "RollingProjectile":
                width = int(entity.rolling_radius * 2 * self.tile_size)
                height = int(entity.radius_y * 2 * self.tile_size)
                rect = pygame.Rect(screen_x - width//2, screen_y - height//2, width, height)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=4)

                if entity.time_alive < entity.spawn_delay:
                    countdown = entity.spawn_delay - entity.time_alive
                    status_text = f"{countdown:.1f}s"
                else:
                    status_text = "ROLLING"

                spell_name = getattr(entity, 'spell_name', 'ROLLING')
                spell_text = self.small_font.render(f"{spell_name.upper()}", True, UI_TEXT)
                status_surface = self.small_font.render(status_text, True, UI_TEXT)

                spell_rect = spell_text.get_rect(center=(screen_x, screen_y - 20))
                status_rect = status_surface.get_rect(center=(screen_x, screen_y - 5))

                self.screen.blit(spell_text, spell_rect)
                self.screen.blit(status_surface, status_rect)
                continue

            # Regular entities: circular hitbox from the unit's collision radius.
            hitbox_radius_tiles = getattr(entity.card_stats, "collision_radius", None) or 0.5
            hitbox_radius = max(4, int(hitbox_radius_tiles * self.tile_size))
            flying = bool(getattr(entity, "is_air_unit", False))
            lift = int(self.tile_size * 0.45) if flying else 0

            # Shadow stays on the ground; flyers are drawn raised above it.
            draw_soft_shadow(self.screen, screen_x, screen_y + int(hitbox_radius * 0.6), hitbox_radius,
                             alpha=60 if flying else 90)
            screen_y -= lift

            badge = max(10, int(hitbox_radius * 2 * 1.25))
            image = self.get_card_image(entity_name)
            light, mid, dark = tex.TEAM_COLORS[entity.player_id]
            draw_aa_filled_circle(self.screen, (screen_x, screen_y), badge // 2 + 2, dark)
            if image is not None:
                art = tex.circular_art(id(image), badge, image)
                self.screen.blit(art, art.get_rect(center=(screen_x, screen_y)))
                drew_image = True
            else:
                drew_image = False
            draw_aa_circle_outline(self.screen, (screen_x, screen_y), badge // 2 + 2, mid, width=3)

            if not drew_image:
                visual_radius = int(hitbox_radius_tiles * 0.7 * self.tile_size)
                draw_aa_filled_circle(self.screen, (screen_x, screen_y), visual_radius, color)
                draw_aa_circle_outline(self.screen, (screen_x, screen_y), visual_radius, BLACK, width=2)

            # Small colored dot under the art for team ownership at a glance.
            team_dot_y = screen_y + hitbox_radius - 4
            draw_aa_filled_circle(self.screen, (screen_x, team_dot_y), 5, color)
            draw_aa_circle_outline(self.screen, (screen_x, team_dot_y), 5, BLACK, width=1)

            # AoE radius for ground troops with area damage (only for 0.5s after attack)
            if (entity_type in ["Troop", "Building"] and
                hasattr(entity, 'card_stats') and entity.card_stats):
                area_damage_radius = getattr(entity.card_stats, 'area_damage_radius', None)
                projectile_speed = getattr(entity.card_stats, 'projectile_speed', None)

                is_melee_with_aoe = (area_damage_radius and area_damage_radius > 0 and
                                    (not projectile_speed or projectile_speed == 0))

                if is_melee_with_aoe and hasattr(entity, 'last_attack_time'):
                    time_since_attack = self.battle.time - entity.last_attack_time
                    if 0 <= time_since_attack <= 0.5:
                        aoe_radius_tiles = area_damage_radius / 1000.0
                        aoe_radius_pixels = int(aoe_radius_tiles * self.tile_size)
                        fade = 1.0 - time_since_attack / 0.5
                        ring_color = (255, int(100 * fade + 50), int(100 * fade + 50))
                        draw_aa_circle_outline(self.screen, (screen_x, screen_y), aoe_radius_pixels, ring_color, width=2)
                        draw_aa_filled_circle(self.screen, (screen_x, screen_y), 3, (200, 50, 50))

            # Health bar above the unit
            if hasattr(entity, 'hitpoints') and hasattr(entity, 'max_hitpoints'):
                health_ratio = entity.hitpoints / entity.max_hitpoints
                bar_y_offset = hitbox_radius + 22
                bar_width = int(self.tile_size * 0.9)
                bar_height = 4
                bar_x = screen_x - bar_width // 2
                bar_y = screen_y - bar_y_offset
                self.draw_health_bar(bar_x, bar_y, bar_width, bar_height, health_ratio)

            # Sight range circle (dotted) - debug overlay, toggle with V.
            if getattr(self, "show_sight_ranges", False) and getattr(entity, 'sight_range', None):
                sight_radius = int(entity.sight_range * self.tile_size)
                dim = (color[0]//3, color[1]//3, color[2]//3)
                for angle in range(0, 360, 20):
                    x = screen_x + int(sight_radius * math.cos(math.radians(angle)))
                    y = screen_y + int(sight_radius * math.sin(math.radians(angle)))
                    draw_aa_filled_circle(self.screen, (x, y), 2, dim)

            # Targeting line - shows what troop is currently fixated on
            if hasattr(entity, 'target_id') and entity.target_id:
                target = self.battle.entities.get(entity.target_id)
                if target and target.is_alive:
                    distance_to_target = entity.position.distance_to(target.position)
                    arrow_color = BLUE if entity.player_id == 0 else RED

                    if distance_to_target <= entity.sight_range:
                        arrow_target_x, arrow_target_y = self.world_to_screen(target.position.x, target.position.y)
                    else:
                        if hasattr(entity, '_get_pathfind_target'):
                            pathfind_target = entity._get_pathfind_target(target)
                            forward_y = entity.position.y + (3.0 if entity.player_id == 0 else -3.0)
                            forward_pos = (entity.position.x, forward_y)
                            pathfind_pos = (pathfind_target.x, pathfind_target.y)

                            if abs(pathfind_pos[0] - forward_pos[0]) > 1.0 or abs(pathfind_pos[1] - forward_pos[1]) > 1.0:
                                arrow_target_x, arrow_target_y = self.world_to_screen(pathfind_target.x, pathfind_target.y)
                            else:
                                arrow_target_x, arrow_target_y = self.world_to_screen(forward_pos[0], forward_pos[1])
                        else:
                            forward_y = entity.position.y + (3.0 if entity.player_id == 0 else -3.0)
                            arrow_target_x, arrow_target_y = self.world_to_screen(entity.position.x, forward_y)

                    line_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(line_surf, (*arrow_color, 130),
                                       (screen_x, screen_y), (arrow_target_x, arrow_target_y), 2)
                    self.screen.blit(line_surf, (0, 0))
                    draw_aa_circle_outline(self.screen, (arrow_target_x, arrow_target_y), 7, arrow_color, width=2)

            # Entity label, small and dim so it doesn't compete with the art
            label = self.small_font.render(entity_name, True, UI_TEXT_DIM)
            label_rect = label.get_rect(center=(screen_x, screen_y + 24))
            self.screen.blit(label, label_rect)

    # ------------------------------------------------------------------
    # Hand / elixir bar
    # ------------------------------------------------------------------

    def _get_player_hand(self, player) -> "list[str] | None":
        """Best-effort lookup of a player's current hand as a list of card
        names. Tries several likely attribute names/shapes since the exact
        Player class layout wasn't available when this was written. Returns
        None if nothing recognizable was found (caller should skip drawing)."""
        for attr in ("hand", "current_hand", "cards_in_hand", "hand_cards"):
            value = getattr(player, attr, None)
            if not value:
                continue
            names = []
            for card in value:
                if isinstance(card, str):
                    names.append(card)
                else:
                    name = getattr(card, "name", None)
                    if name:
                        names.append(name)
            if names:
                return names
        return None

    def draw_hand_bar(self, player_id: int, bar_y: int, label: str) -> None:
        """Clash-Royale-style hand bar:

            [elixir #] [ card ] [ card ] [ card ] [ card ]
                       [====== elixir bar, 10 segments ======]

        The elixir number sits in its own drop-shaped badge to the LEFT of
        the hand, and the segmented elixir bar runs the width of the hand
        directly BELOW the card icons - not off to the side, not as a raw
        pygame.draw.rect strip."""
        player = self.battle.players[player_id]
        hand = self._get_player_hand(player) or []
        elixir = float(getattr(player, "elixir", 0.0))

        accent = BLUE if player_id == 0 else RED
        icon_size = HAND_BAR_HEIGHT - 46
        spacing = 8
        badge_diameter = icon_size
        badge_gap = 14

        total_hand_width = (
            len(hand) * icon_size + max(0, len(hand) - 1) * spacing
            if hand else icon_size * 4 + spacing * 3
        )
        block_width = badge_diameter + badge_gap + total_hand_width
        block_left = ARENA_X + ARENA_WIDTH // 2 - block_width // 2

        icons_y = bar_y + 20
        badge_center = (block_left + badge_diameter // 2, icons_y + icon_size // 2)

        # Label above the whole block
        label_surface = self.font.render(label, True, accent)
        label_rect = label_surface.get_rect(midtop=(ARENA_X + ARENA_WIDTH // 2, bar_y - 2))
        self._blit_outlined_text(None, label_rect, label, self.font, color=accent)

        # --- Elixir number badge (drop shape) to the LEFT of the hand ---
        drop_color = ELIXIR_PURPLE if elixir < 10 else UI_GOLD
        draw_aa_filled_circle(self.screen, badge_center, badge_diameter // 2, drop_color)
        draw_aa_circle_outline(self.screen, badge_center, badge_diameter // 2, UI_BORDER, width=2)
        draw_aa_filled_circle(
            self.screen,
            (badge_center[0] - badge_diameter // 6, badge_center[1] - badge_diameter // 6),
            max(1, badge_diameter // 6),
            ELIXIR_PURPLE_LIGHT,
        )
        elixir_text = str(int(elixir))
        text_surf = self.elixir_font.render(elixir_text, True, WHITE)
        text_rect = text_surf.get_rect(center=badge_center)
        self._blit_outlined_text(None, text_rect, elixir_text, self.elixir_font, color=WHITE)

        # --- Hand card icons ---
        hand_start_x = block_left + badge_diameter + badge_gap
        card_positions = []
        card_x = hand_start_x
        cards_to_draw = hand if hand else [None] * 4
        for card_name in cards_to_draw:
            box_rect = pygame.Rect(card_x, icons_y, icon_size, icon_size)
            card_positions.append((box_rect, card_name))
            card_x += icon_size + spacing

        for box_rect, card_name in card_positions:
            draw_rounded_panel(self.screen, box_rect, radius=8, fill=UI_BG_LIGHT, border=UI_GOLD_DARK, border_width=3)
            inner = box_rect.inflate(-8, -8)
            draw_rounded_panel(self.screen, inner, radius=6, fill=(58, 66, 88))

            if card_name:
                if not self.draw_card_image(card_name, box_rect.centerx, box_rect.centery, icon_size - 10):
                    text = self.small_font.render(card_name[:8], True, UI_TEXT)
                    text_rect = text.get_rect(center=box_rect.center)
                    self.screen.blit(text, text_rect)

        # --- Segmented elixir bar directly below the hand, spanning its width ---
        bar_top = icons_y + icon_size + 8
        bar_left = hand_start_x
        bar_width = (card_positions[-1][0].right - hand_start_x) if card_positions else total_hand_width
        bar_height = 14
        max_elixir = 10
        segment_gap = 2
        segment_width = (bar_width - segment_gap * (max_elixir - 1)) / max_elixir

        bar_bg_rect = pygame.Rect(bar_left - 3, bar_top - 3, bar_width + 6, bar_height + 6)
        draw_rounded_panel(self.screen, bar_bg_rect, radius=bar_height // 2 + 3, fill=UI_BORDER)

        fractional = elixir - int(elixir)
        for i in range(max_elixir):
            seg_x = bar_left + i * (segment_width + segment_gap)
            seg_rect = pygame.Rect(int(seg_x), bar_top, int(segment_width), bar_height)
            if i < int(elixir):
                fill_t = 1.0
            elif i == int(elixir):
                fill_t = fractional
            else:
                fill_t = 0.0

            draw_rounded_panel(self.screen, seg_rect, radius=4, fill=ELIXIR_EMPTY)
            if fill_t > 0:
                filled_rect = pygame.Rect(seg_rect.x, seg_rect.y, max(2, int(seg_rect.width * fill_t)), seg_rect.height)
                pygame.draw.rect(self.screen, ELIXIR_PURPLE, filled_rect, border_radius=4)
                highlight_rect = pygame.Rect(filled_rect.x + 1, filled_rect.y + 1, max(1, filled_rect.width - 2), max(1, filled_rect.height // 3))
                pygame.draw.rect(self.screen, ELIXIR_PURPLE_LIGHT, highlight_rect, border_radius=3)

    def draw_hands(self) -> None:
        """Draw both hand bars: P1 (red/enemy) on top, P0 (blue) on bottom -
        matching the fixed board orientation (blue always at bottom)."""
        self.draw_hand_bar(1, TOP_HAND_Y, "P1 HAND (enemy)")
        self.draw_hand_bar(0, BOTTOM_HAND_Y, "P0 HAND (you)")

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def draw_ui(self):
        """Draw the side status panel. No troop-name breakdown here anymore -
        you can already see every unit on the board, listing them again in
        text was pure redundant clutter."""
        panel_rect = pygame.Rect(ARENA_X + ARENA_WIDTH + 20, ARENA_Y, 400, ARENA_HEIGHT)
        draw_rounded_panel(self.screen, panel_rect, radius=10, fill=UI_BG, border=UI_BORDER, border_width=2)

        ui_x = panel_rect.x + 16
        ui_y = panel_rect.y + 14
        line_height = 25

        title = self.large_font.render("Battle Status", True, UI_GOLD)
        self.screen.blit(title, (ui_x, ui_y))
        ui_y += 40

        time_text = self.font.render(f"Time: {self.battle.time:.1f}s", True, UI_TEXT)
        self.screen.blit(time_text, (ui_x, ui_y))
        ui_y += line_height

        tick_text = self.font.render(f"Tick: {self.battle.tick}", True, UI_TEXT)
        self.screen.blit(tick_text, (ui_x, ui_y))
        ui_y += line_height * 2

        for i, player in enumerate(self.battle.players):
            color = BLUE if i == 0 else RED
            player_text = self.font.render(f"Player {i}:", True, color)
            self.screen.blit(player_text, (ui_x, ui_y))
            ui_y += line_height

            elixir_text = self.font.render(f"  Elixir: {player.elixir:.1f}/10", True, UI_TEXT)
            self.screen.blit(elixir_text, (ui_x, ui_y))
            ui_y += line_height

            crowns_text = self.font.render(f"  Crowns: {self.battle.crowns(i)}", True, UI_TEXT)
            self.screen.blit(crowns_text, (ui_x, ui_y))
            ui_y += line_height

            towers_text = self.font.render(f"  King: {int(player.king_tower_hp)}", True, UI_TEXT)
            self.screen.blit(towers_text, (ui_x, ui_y))
            ui_y += line_height

            towers_text2 = self.font.render(f"  Towers: {int(player.left_tower_hp)}/{int(player.right_tower_hp)}", True, UI_TEXT)
            self.screen.blit(towers_text2, (ui_x, ui_y))
            ui_y += line_height * 2

        alive_entities = sum(1 for e in self.battle.entities.values() if e.is_alive)
        entities_text = self.font.render(f"Entities on board: {alive_entities}", True, UI_TEXT)
        self.screen.blit(entities_text, (ui_x, ui_y))
        ui_y += line_height

        # Game state
        ui_y += line_height
        if self.battle.game_over:
            winner_text = self.font.render(f"WINNER: Player {self.battle.winner}!", True, UI_GOLD)
            self.screen.blit(winner_text, (ui_x, ui_y))
        elif self.battle.double_elixir:
            elixir_text = self.font.render("DOUBLE ELIXIR!", True, ELIXIR_PURPLE_LIGHT)
            self.screen.blit(elixir_text, (ui_x, ui_y))

        # Controls
        ui_y = panel_rect.bottom - 120
        pygame.draw.line(self.screen, UI_BORDER, (ui_x, ui_y - 8), (panel_rect.right - 16, ui_y - 8), 1)
        controls_title = self.font.render("Controls:", True, UI_GOLD)
        self.screen.blit(controls_title, (ui_x, ui_y))
        ui_y += line_height

        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Battle",
            "1-5: Speed (1x to 5x)",
            "V: Sight ranges on/off",
            "ESC: Exit"
        ]

        for control in controls:
            control_text = self.small_font.render(control, True, UI_TEXT_DIM)
            self.screen.blit(control_text, (ui_x, ui_y))
            ui_y += 20

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not getattr(self, 'paused', False)
                elif event.key == pygame.K_r:
                    # Reset battle
                    self.engine = BattleEngine()
                    self.battle = self.engine.create_battle()
                    self.setup_test_battle()
                elif event.key >= pygame.K_1 and event.key <= pygame.K_5:
                    # Set speed multiplier
                    self.speed = event.key - pygame.K_0
                elif event.key == pygame.K_i:
                    # Toggle investigation mode
                    self.investigation_mode = not getattr(self, 'investigation_mode', False)
                    if self.investigation_mode:
                        print("🔍 Investigation mode ON - taking screenshots every 30 ticks")
                        self.investigation_counter = 0
                        # Create investigation folder
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.investigation_folder = f"investigation/{timestamp}"
                        os.makedirs(self.investigation_folder, exist_ok=True)
                    else:
                        print("🔍 Investigation mode OFF")
                elif event.key == pygame.K_s:
                    # Take single screenshot
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_screenshot_{timestamp}.png"
                    self.take_screenshot(filename)
                elif event.key == pygame.K_v:
                    self.show_sight_ranges = not getattr(self, "show_sight_ranges", False)
                elif event.key == pygame.K_p:
                    # Take pathfinding debug screenshot
                    self.take_pathfinding_debug_screenshot()

        return True

    def take_screenshot(self, filename: str = "battle_screenshot.png"):
        """Take a screenshot of the current battle state"""
        pygame.image.save(self.screen, filename)
        print(f"📸 Screenshot saved as: {filename}")
        return filename

    def take_pathfinding_debug_screenshot(self):
        """Take a screenshot with pathfinding debug info overlaid"""
        # Draw pathfinding debug info
        for entity in self.battle.entities.values():
            if hasattr(entity, '_get_pathfind_target') and hasattr(entity, 'target_id'):
                if entity.target_id:
                    target = self.battle.entities.get(entity.target_id)
                    if target:
                        # Get what the pathfinding target would be
                        pathfind_target = entity._get_pathfind_target(target, self.battle)

                        # Draw pathfinding target as bright green circle
                        screen_x, screen_y = self.world_to_screen(pathfind_target.x, pathfind_target.y)
                        pygame.draw.circle(self.screen, (0, 255, 0), (screen_x, screen_y), 15, 3)

                        # Draw line from entity to pathfind target
                        entity_x, entity_y = self.world_to_screen(entity.position.x, entity.position.y)
                        pygame.draw.line(self.screen, (0, 255, 0), (entity_x, entity_y), (screen_x, screen_y), 2)

                        # Add text showing pathfind target type
                        current_side = 0 if entity.position.y < 16.0 else 1
                        target_side = 0 if target.position.y < 16.0 else 1
                        need_to_cross = current_side != target_side
                        distance_to_target = entity.position.distance_to(target.position)
                        on_bridge = (abs(entity.position.x - 3.0) <= 1.5 or abs(entity.position.x - 14.0) <= 1.5) and abs(entity.position.y - 16.0) <= 1.0

                        debug_text = ""
                        if distance_to_target <= entity.sight_range:
                            debug_text = "DIRECT"
                        elif on_bridge:
                            debug_text = "TO_TOWER"
                        elif need_to_cross:
                            debug_text = "TO_BRIDGE"
                        else:
                            debug_text = "DIRECT"

                        text_surface = self.font.render(debug_text, True, (0, 255, 0))
                        self.screen.blit(text_surface, (screen_x + 20, screen_y - 10))

        pygame.display.flip()

        # Take screenshot
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pathfinding_debug_{timestamp}.png"
        self.take_screenshot(filename)

    def run_replay_mode(self):
        """Run battle and screenshot every tick until first attack"""
        print("🎮 Starting Battle Replay Mode")
        print("📸 Will screenshot every tick until first attack occurs")

        # Create replay folder with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_folder = f"replay/{timestamp}"
        os.makedirs(replay_folder, exist_ok=True)

        print(f"📁 Replay folder: {replay_folder}")

        tick = 0
        attack_detected = False

        # Track initial HP of all entities to detect attacks
        initial_hp = {}
        for entity_id, entity in self.battle.entities.items():
            if hasattr(entity, 'hitpoints'):
                initial_hp[entity_id] = entity.hitpoints

        while not attack_detected and tick < 200:  # Max 200 ticks safety limit
            # Step the battle
            old_hp = {}
            for entity_id, entity in self.battle.entities.items():
                if hasattr(entity, 'hitpoints'):
                    old_hp[entity_id] = entity.hitpoints

            self.battle.step(speed_factor=1.0)

            # Check for attacks (HP changes)
            for entity_id, entity in self.battle.entities.items():
                if entity_id in old_hp and hasattr(entity, 'hitpoints'):
                    if entity.hitpoints < old_hp[entity_id]:
                        attack_detected = True
                        print(f"⚔️  Attack detected at tick {tick}! {entity.card_stats.name if entity.card_stats else 'Entity'} took {old_hp[entity_id] - entity.hitpoints} damage")
                        break

            # Draw and screenshot
            self.draw_backdrop()
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_hands()
            self.draw_ui()

            # Add tick counter to screen
            tick_text = self.large_font.render(f"Tick: {tick}", True, UI_TEXT)
            self.screen.blit(tick_text, (10, 10))

            pygame.display.flip()

            # Save screenshot
            screenshot_path = f"{replay_folder}/tick_{tick:04d}.png"
            pygame.image.save(self.screen, screenshot_path)

            tick += 1

            # Small delay to see progress
            time.sleep(0.1)

        print(f"📸 Replay complete! {tick} screenshots saved to {replay_folder}")
        print(f"🎬 To view replay: open files tick_0000.png through tick_{tick-1:04d}.png")

        # Keep window open briefly
        time.sleep(2)
        pygame.quit()

        return replay_folder

    def run_and_screenshot(self):
        """Run battle for a few steps and take screenshot"""
        print("🎮 Starting Battle Visualization (Auto-Screenshot Mode)")

        # Run for a few ticks to get interesting state
        for i in range(50):
            self.battle.step(speed_factor=1.0)

            # Draw everything every 10 ticks to show progression
            if i % 10 == 0:
                self.draw_backdrop()
                self.draw_arena()
                self.draw_towers()
                self.draw_entities()
                self.draw_hands()
                self.draw_ui()
                pygame.display.flip()

                if i == 30:  # Take screenshot at tick 30
                    screenshot_file = self.take_screenshot()

        # Final screenshot
        self.draw_backdrop()
        self.draw_arena()
        self.draw_towers()
        self.draw_entities()
        self.draw_hands()
        self.draw_ui()
        pygame.display.flip()

        final_screenshot = self.take_screenshot("final_battle_state.png")

        # Keep window open briefly to show result
        time.sleep(2)
        pygame.quit()

        return final_screenshot

    def run_auto_investigation(self, max_ticks=15):
        """Automatic investigation mode - runs battle and takes screenshots automatically"""
        print("🔍 Starting automatic pathfinding investigation...")
        print(f"Will capture ticks 0-{max_ticks} and terminate automatically")

        # Create investigation folder
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        investigation_folder = f"investigation/{timestamp}"
        os.makedirs(investigation_folder, exist_ok=True)
        print(f"📁 Screenshots will be saved to: {investigation_folder}")

        # Take screenshots for ticks 0-15 only
        tick = 0

        # Take initial screenshot before any ticks
        self.draw_backdrop()
        self.draw_arena()
        self.draw_towers()
        self.draw_entities()
        self.draw_hands()
        self.draw_ui()

        # Add tick info
        tick_text = self.large_font.render(f"Tick: {tick}", True, UI_TEXT)
        self.screen.blit(tick_text, (10, 10))

        pygame.display.flip()

        # Take screenshot
        filename = f"{investigation_folder}/tick_{tick:04d}_initial.png"
        pygame.image.save(self.screen, filename)
        print(f"📸 Screenshot: tick_{tick:04d}_initial.png")

        while tick < 15:
            # Step battle
            self.battle.step(speed_factor=1.0)
            tick += 1

            # Always take screenshot for ticks 1-15
            should_screenshot = True
            step_info = f"step_{tick}"

            if should_screenshot:
                # Draw everything
                self.draw_backdrop()
                self.draw_arena()
                self.draw_towers()
                self.draw_entities()
                self.draw_hands()
                self.draw_ui()

                # Add tick info
                tick_text = self.large_font.render(f"Tick: {tick}", True, UI_TEXT)
                self.screen.blit(tick_text, (10, 10))

                pygame.display.flip()

                # Take screenshot
                filename = f"{investigation_folder}/tick_{tick:04d}_{step_info}.png"
                pygame.image.save(self.screen, filename)
                print(f"📸 Screenshot: tick_{tick:04d}_{step_info}.png")

            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

        print(f"✅ Investigation complete! Screenshots saved to {investigation_folder}")
        pygame.quit()
        return investigation_folder

    def run(self):
        """Main visualization loop"""
        print("🎮 Starting Battle Visualization")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset Battle")
        print("  1-5: Speed multiplier (1x to 5x)")
        print("  I: Toggle investigation mode (auto screenshots)")
        print("  S: Take manual screenshot")
        print("  P: Take pathfinding debug screenshot")
        print("  ESC: Exit")

        self.paused = False
        self.speed = 1
        running = True

        while running:
            # Handle events
            running = self.handle_events()

            # Update battle
            if not self.paused and not self.battle.game_over:
                for _ in range(self.speed):
                    self.battle.step(speed_factor=1.0)

                # Investigation mode - take screenshots at intervals
                if getattr(self, 'investigation_mode', False):
                    if not hasattr(self, 'investigation_counter'):
                        self.investigation_counter = 0
                    self.investigation_counter += self.speed

                    # Take screenshot every 30 ticks
                    if self.investigation_counter >= 30:
                        self.investigation_counter = 0
                        # Draw everything first
                        self.draw_backdrop()
                        self.draw_arena()
                        self.draw_towers()
                        self.draw_entities()
                        self.draw_hands()
                        self.draw_ui()
                        pygame.display.flip()

                        # Take investigation screenshot
                        tick = getattr(self.battle, 'tick', 0)
                        filename = f"{self.investigation_folder}/tick_{tick:04d}.png"
                        self.take_screenshot(filename)

            # Draw everything
            self.draw_backdrop()
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_hands()
            self.draw_ui()

            # Show pause indicator
            if self.paused:
                pause_text = self.large_font.render("PAUSED", True, (235, 90, 90))
                pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, 30))
                self.screen.blit(pause_text, pause_rect)

            # Show speed indicator
            if self.speed > 1:
                speed_text = self.font.render(f"Speed: {self.speed}x", True, ELIXIR_PURPLE_LIGHT)
                speed_rect = speed_text.get_rect(topleft=(10, 10))
                self.screen.blit(speed_text, speed_rect)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS display

        pygame.quit()

def main():
    """Run the battle visualization"""
    try:
        visualizer = BattleVisualizer()

        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "screenshot":
                screenshot_file = visualizer.run_and_screenshot()
                return screenshot_file
            elif sys.argv[1] == "replay":
                replay_folder = visualizer.run_replay_mode()
                return replay_folder
            elif sys.argv[1] == "investigate":
                investigation_folder = visualizer.run_auto_investigation()
                return investigation_folder
        else:
            visualizer.run()
            return None
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
        return None
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
