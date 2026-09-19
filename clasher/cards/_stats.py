"""Helpers for card mechanics to read their numbers from Cards.json.

Mechanics keep constructor defaults (used by unit tests with mock entities and
by legacy gamedata-only cards) but override them in ``on_attach`` from the
entity's Cards.json stat line.
"""

from typing import Any, Optional, Tuple

LEGACY_LEVEL_MULTIPLIER = 1.1 ** 10  # level-1 gamedata value -> level 11 (legacy convention)


def card_stat(entity: Any, key: str, default: Any = None) -> Any:
    card_stats = getattr(entity, "card_stats", None)
    stat = getattr(card_stats, "stat", None)
    if not callable(stat):
        return default
    value = stat(key, None)
    return default if value is None else value


def card_number(entity: Any, key: str, default: Optional[float] = None) -> Optional[float]:
    value = card_stat(entity, key, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def card_range(entity: Any, key: str) -> Optional[Tuple[float, float]]:
    stats = getattr(getattr(entity, "card_stats", None), "stats", None)
    return stats.range_pair(key) if stats is not None else None


def char_data(entity: Any) -> dict:
    return getattr(getattr(entity, "card_stats", None), "summon_character_data", None) or {}


def legacy_scaled(value: Optional[float]) -> Optional[float]:
    """Scale a template-only level-1 number the way the simulator always has."""
    return None if value is None else value * LEGACY_LEVEL_MULTIPLIER
