"""Procedural textures for the pygame battle renderer.

Everything is drawn from code (no extra asset files), pre-rendered once per
size and cached, so per-frame cost is a handful of blits. The look follows the
stone "League" arena: sandstone floor tiles with bevels and cracks, a stone
banked river, stone bridges with pillars, golden-edged tower platforms and
crenellated towers in team colours.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Tuple

import numpy as np
import pygame

Color = Tuple[int, int, int]

FLOOR_BASE: Color = (219, 190, 142)
FLOOR_DARK: Color = (196, 164, 116)
GROUT: Color = (170, 138, 96)
WATER_TOP: Color = (74, 170, 228)
WATER_BOTTOM: Color = (34, 104, 178)
BANK: Color = (150, 118, 80)
STONE_LIGHT: Color = (206, 196, 180)
STONE: Color = (168, 158, 144)
STONE_DARK: Color = (112, 104, 94)
BRIDGE_DECK: Color = (178, 146, 104)
GOLD: Color = (236, 184, 64)
GOLD_DARK: Color = (160, 112, 30)

TEAM_COLORS = {
    0: ((96, 140, 236), (54, 86, 176), (28, 44, 104)),   # light, mid, dark
    1: ((236, 104, 96), (178, 58, 56), (104, 28, 30)),
}


def _clamp(value: float) -> int:
    return max(0, min(255, int(value)))


def _shade(color: Color, delta: float) -> Color:
    return (_clamp(color[0] + delta), _clamp(color[1] + delta), _clamp(color[2] + delta))


def build_floor(cols: int, rows: int, tile: int, seed: int = 7) -> pygame.Surface:
    """Sandstone floor: one bevelled slab per board tile, with colour jitter,
    a few big two-tile slabs and cracks, plus fine grain noise."""
    rng = random.Random(seed)
    surf = pygame.Surface((cols * tile, rows * tile))
    surf.fill(GROUT)
    for ty in range(rows):
        for tx in range(cols):
            jitter = rng.uniform(-10, 10)
            base = _shade(FLOOR_BASE if (tx + ty) % 2 == 0 else FLOOR_DARK, jitter)
            rect = pygame.Rect(tx * tile + 1, ty * tile + 1, tile - 2, tile - 2)
            surf.fill(base, rect)
            # bevel: light top/left, dark bottom/right
            pygame.draw.line(surf, _shade(base, 22), rect.topleft, rect.topright)
            pygame.draw.line(surf, _shade(base, 22), rect.topleft, rect.bottomleft)
            pygame.draw.line(surf, _shade(base, -28), rect.bottomleft, rect.bottomright)
            pygame.draw.line(surf, _shade(base, -28), rect.topright, rect.bottomright)
            if rng.random() < 0.07:  # crack
                x0 = rect.x + rng.randint(2, tile - 4)
                y0 = rect.y + rng.randint(2, tile - 4)
                points = [(x0, y0)]
                for _ in range(3):
                    x0 = min(rect.right - 2, max(rect.x + 1, x0 + rng.randint(-4, 4)))
                    y0 = min(rect.bottom - 2, max(rect.y + 1, y0 + rng.randint(1, 4)))
                    points.append((x0, y0))
                pygame.draw.lines(surf, _shade(base, -45), False, points, 1)
            if rng.random() < 0.05:  # chipped corner
                chip = rng.randint(3, max(4, tile // 3))
                pygame.draw.polygon(surf, _shade(base, -18), [
                    (rect.right - chip, rect.bottom), (rect.right, rect.bottom), (rect.right, rect.bottom - chip)])
    _grain(surf, seed, 5.0)
    return surf


def _grain(surface: pygame.Surface, seed: int, strength: float) -> None:
    arr = pygame.surfarray.pixels3d(surface)
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, strength, arr.shape[:2]).astype(np.float32)
    arr[...] = np.clip(arr.astype(np.float32) + grain[..., None], 0, 255).astype(np.uint8)
    del arr


def build_river(width: int, height: int, seed: int = 11) -> pygame.Surface:
    """Water band with depth gradient, caustic streaks and stone banks."""
    surf = pygame.Surface((width, height))
    for i in range(height):
        t = i / max(1, height - 1)
        color = tuple(_clamp(WATER_TOP[c] + (WATER_BOTTOM[c] - WATER_TOP[c]) * t) for c in range(3))
        pygame.draw.line(surf, color, (0, i), (width, i))
    rng = random.Random(seed)
    for _ in range(width // 6):
        x = rng.randint(0, width)
        y = rng.randint(4, max(5, height - 5))
        length = rng.randint(6, 18)
        pygame.draw.line(surf, (150, 214, 245), (x, y), (x + length, y), 1)
    _grain(surf, seed, 3.0)
    bank = max(3, height // 10)
    for y0, sign in ((0, 1), (height - bank, -1)):
        pygame.draw.rect(surf, BANK, (0, y0, width, bank))
        pygame.draw.line(surf, _shade(BANK, 30 * sign), (0, y0 + (0 if sign > 0 else bank - 1)),
                         (width, y0 + (0 if sign > 0 else bank - 1)), 1)
    return surf


def build_bridge(width: int, height: int, tile: int) -> pygame.Surface:
    """Stone bridge deck with side rails and corner pillars."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    deck = pygame.Rect(0, 0, width, height)
    surf.fill(BRIDGE_DECK, deck)
    course = max(4, tile // 2)
    rng = random.Random(width * 31 + height)
    for row, y in enumerate(range(0, height, course)):
        offset = (course if row % 2 else 0)
        for x in range(-offset, width, course * 2):
            block = pygame.Rect(x + 1, y + 1, course * 2 - 2, course - 2)
            surf.fill(_shade(BRIDGE_DECK, rng.uniform(-16, 12)), block.clip(deck))
    rail = max(3, tile // 5)
    for x in (0, width - rail):
        pygame.draw.rect(surf, STONE_DARK, (x, 0, rail, height))
    pillar = max(rail + 2, tile // 2)
    for px in (0, width - pillar):
        for py in (0, height - pillar):
            pygame.draw.rect(surf, STONE_LIGHT, (px, py, pillar, pillar), border_radius=2)
            pygame.draw.rect(surf, STONE_DARK, (px, py, pillar, pillar), 1, border_radius=2)
    return surf


def build_wall_block(tile: int) -> pygame.Surface:
    """Raised stone block for unplayable fence tiles."""
    surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.rect(surf, STONE_DARK, (0, 2, tile, tile - 2), border_radius=3)
    pygame.draw.rect(surf, STONE, (1, 0, tile - 2, tile - 4), border_radius=3)
    pygame.draw.line(surf, STONE_LIGHT, (3, 1), (tile - 4, 1))
    pygame.draw.line(surf, STONE_DARK, (tile // 2, 2), (tile // 2, tile // 2), 1)
    return surf


@lru_cache(maxsize=64)
def tower_sprite(kind: str, team: int, size_px: int, awake: bool, destroyed: bool) -> pygame.Surface:
    """Top-down crown tower on its golden platform. ``kind``: princess | king."""
    surf = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    plat = pygame.Rect(0, 0, size_px, size_px)
    pygame.draw.rect(surf, GOLD_DARK, plat, border_radius=max(3, size_px // 12))
    pygame.draw.rect(surf, GOLD, plat.inflate(-4, -4), border_radius=max(3, size_px // 12))
    inner = plat.inflate(-size_px // 7, -size_px // 7)
    pygame.draw.rect(surf, STONE_DARK, inner, border_radius=4)

    if destroyed:
        rng = random.Random(size_px + team)
        for _ in range(9):
            r = rng.randint(size_px // 14, size_px // 7)
            cx = rng.randint(inner.left + r, inner.right - r)
            cy = rng.randint(inner.top + r, inner.bottom - r)
            pygame.draw.circle(surf, _shade(STONE, rng.uniform(-30, 10)), (cx, cy), r)
            pygame.draw.circle(surf, STONE_DARK, (cx, cy), r, 1)
        return surf

    light, mid, dark = TEAM_COLORS[team]
    if not awake:
        # Sleeping King: desaturated stone, the team colour only on the trim.
        light, mid = _shade(STONE, 10), STONE
    body = inner.inflate(-size_px // 10, -size_px // 10)
    pygame.draw.rect(surf, dark, body.move(0, 3), border_radius=5)
    pygame.draw.rect(surf, mid, body, border_radius=5)
    pygame.draw.rect(surf, light, body.inflate(-body.w // 3, -body.h // 3), border_radius=4)
    # crenellations around the rim
    merlon = max(3, body.w // 7)
    for i in range(0, body.w - merlon + 1, merlon * 2):
        for y in (body.top - merlon // 2, body.bottom - merlon // 2):
            pygame.draw.rect(surf, _shade(mid, 25), (body.left + i, y, merlon, merlon), border_radius=1)
            pygame.draw.rect(surf, dark, (body.left + i, y, merlon, merlon), 1, border_radius=1)
    pygame.draw.rect(surf, dark, body, 2, border_radius=5)

    cx, cy = body.center
    if kind == "king":
        crown = GOLD if awake else (150, 146, 140)
        w = body.w // 2
        h = body.h // 3
        pts = [(cx - w // 2, cy + h // 2), (cx - w // 2, cy - h // 2), (cx - w // 4, cy),
               (cx, cy - h // 2 - 2), (cx + w // 4, cy), (cx + w // 2, cy - h // 2), (cx + w // 2, cy + h // 2)]
        pygame.draw.polygon(surf, crown, pts)
        pygame.draw.polygon(surf, GOLD_DARK if awake else STONE_DARK, pts, 2)
    else:
        r = max(3, body.w // 5)
        pygame.draw.circle(surf, (242, 214, 170), (cx, cy), r)       # princess on top
        pygame.draw.circle(surf, dark, (cx, cy), r, 2)
        pygame.draw.arc(surf, (120, 72, 36), pygame.Rect(cx - r - 3, cy - r - 3, 2 * r + 6, 2 * r + 6),
                        math.radians(200), math.radians(340), 2)       # bow
    return surf


@lru_cache(maxsize=512)
def circular_art(image_id: int, diameter: int, image: pygame.Surface) -> pygame.Surface:
    """Card art cropped to a circle (cached per image and size)."""
    del image_id  # cache key only
    w, h = image.get_size()
    scale = diameter / min(w, h)
    scaled = pygame.transform.smoothscale(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    out = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    out.blit(scaled, ((diameter - scaled.get_width()) // 2, (diameter - scaled.get_height()) // 2))
    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (diameter // 2, diameter // 2), diameter // 2)
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return out


@lru_cache(maxsize=32)
def bomb_sprite(size_px: int, team: int) -> pygame.Surface:
    """Round black bomb with a highlight, a fuse cap and a team-coloured band."""
    surf = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    r = size_px // 2 - 2
    c = (size_px // 2, size_px // 2 + 1)
    pygame.draw.circle(surf, (18, 18, 22), c, r)
    pygame.draw.circle(surf, TEAM_COLORS[team][1], c, r, max(2, size_px // 10))
    pygame.draw.circle(surf, (90, 90, 104), (c[0] - r // 3, c[1] - r // 3), max(2, r // 4))
    cap = pygame.Rect(0, 0, max(4, r // 2), max(3, r // 3))
    cap.midbottom = (c[0], c[1] - r + 2)
    pygame.draw.rect(surf, (120, 110, 96), cap, border_radius=2)
    return surf


@lru_cache(maxsize=32)
def barrel_sprite(size_px: int, team: int) -> pygame.Surface:
    """Wooden barrel seen from above (Skeleton Barrel's falling container)."""
    surf = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    r = size_px // 2 - 2
    c = (size_px // 2, size_px // 2)
    pygame.draw.circle(surf, (132, 88, 48), c, r)
    for k in (0.62, 0.3):
        pygame.draw.circle(surf, (92, 60, 30), c, max(2, int(r * k)), 2)
    pygame.draw.circle(surf, TEAM_COLORS[team][1], c, r, max(2, size_px // 10))
    return surf
