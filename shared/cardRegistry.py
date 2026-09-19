"""Authoritative card registry backed by ``data/cards/Cards.json``.

Cards.json is the single source of truth for numerical card stats (already at
tournament level, so no level scaling is applied on top). ``CardSpawns.json``
describes which sub-units a multi-unit card deploys.

The registry wraps the raw JSON (string ranges such as ``"3.5-6"``, speed
tiers) in ``CardStats`` records and exposes
variant metadata so Evolutions and Heroes can be layered on later without
changing the data format:

* ``variant`` is ``"base"``, ``"evolution"`` or ``"hero"`` (derived from the
  ``Evo``/``Hero`` name prefix),
* ``baseName`` points an Evo/Hero entry at the card it modifies,
* ``variantsOf(base)`` lists every registered variant of a card.

The simulator currently only builds ``base`` variants; see
``clasher/factory/stats_overlay.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cards"
DEFAULT_CARDS_FILE = _DATA_DIR / "Cards.json"
DEFAULT_SPAWNS_FILE = _DATA_DIR / "CardSpawns.json"

VARIANT_BASE = "base"
VARIANT_EVOLUTION = "evolution"
VARIANT_HERO = "hero"

# Speed tiers used by Cards.json -> tiles per minute (the unit the simulator
# moves in). Tier 2-5 match gamedata's TID_SPEED_2..5 exactly (45/60/90/120).
SPEED_TIER_TILES_PER_MIN: Dict[int, float] = {1: 30.0, 2: 45.0, 3: 60.0, 4: 90.0, 5: 120.0}

_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def parseStatRange(value: Any) -> Optional[Tuple[float, float]]:
    """Parse ``"3.5-6"`` style values into ``(3.5, 6.0)``; scalars become ``(v, v)``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    match = _RANGE_RE.match(str(value))
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return None


def _variant_of(name: str, known: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    for prefix, variant in (("Evo", VARIANT_EVOLUTION), ("Hero", VARIANT_HERO)):
        if name.startswith(prefix) and len(name) > len(prefix) and name[len(prefix)].isupper():
            remainder = name[len(prefix):]
            return variant, (remainder if remainder in known else None)
    return VARIANT_BASE, None


@dataclass(frozen=True)
class CardStats:
    """Normalised view of one Cards.json entry. Units: tiles, seconds, tiles/min."""

    name: str
    id: int
    elixir: Optional[int]
    type: str  # troop | air | building | spell | projectile
    variant: str = VARIANT_BASE
    base_name: Optional[str] = None
    values: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # ---- classification -------------------------------------------------
    @property
    def kind(self) -> str:
        """Simulator entity kind: air units are troops that fly."""
        return "troop" if self.type == "air" else self.type

    @property
    def is_air(self) -> bool:
        return self.type == "air"

    @property
    def is_spell(self) -> bool:
        return self.type == "spell"

    @property
    def is_building(self) -> bool:
        return self.type == "building"

    @property
    def is_playable(self) -> bool:
        """True for deckable cards (sub-units carry ``elixir: null``)."""
        return self.elixir is not None or self.name == "Mirror"

    @property
    def is_evolution(self) -> bool:
        return self.variant == VARIANT_EVOLUTION

    @property
    def is_hero(self) -> bool:
        return self.variant == VARIANT_HERO

    # ---- generic accessors ----------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        value = self.values.get(key)
        return default if value is None else value

    def number(self, key: str, default: Optional[float] = None) -> Optional[float]:
        value = self.values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)

    def range_pair(self, key: str) -> Optional[Tuple[float, float]]:
        return parseStatRange(self.values.get(key))

    # ---- common stats -----------------------------------------------------
    @property
    def hitpoints(self) -> Optional[float]:
        return self.number("hp")

    @property
    def damage(self) -> Optional[float]:
        return self.number("damage")

    @property
    def hit_speed(self) -> Optional[float]:
        return self.number("hitSpeed")

    @property
    def range(self) -> Optional[float]:
        return self.number("range")

    @property
    def speed_tier(self) -> Optional[int]:
        value = self.number("speed")
        return int(value) if value is not None else None

    @property
    def speed_tiles_per_min(self) -> Optional[float]:
        tier = self.speed_tier
        return SPEED_TIER_TILES_PER_MIN.get(tier) if tier is not None else None

    @property
    def count(self) -> Optional[int]:
        value = self.number("count")
        return int(value) if value is not None else None

    @property
    def radius(self) -> Optional[float]:
        return self.number("radius")

    @property
    def duration(self) -> Optional[float]:
        return self.number("duration")

    @property
    def lifetime(self) -> Optional[float]:
        return self.number("lifetime")

    @property
    def deploy_time(self) -> Optional[float]:
        return self.number("deployTime")

    @property
    def crown_damage(self) -> Optional[float]:
        return self.number("crownDamage")

    @property
    def death_damage(self) -> Optional[float]:
        return self.number("deathDamage")


class CardDatabase:
    def __init__(self, path=None, spawnsPath=None):
        path = Path(path) if path is not None else DEFAULT_CARDS_FILE
        if spawnsPath is None:
            candidate = path.parent / DEFAULT_SPAWNS_FILE.name
            spawnsPath = candidate if candidate.exists() else None

        with open(path, "r") as f:
            self._cards = json.load(f)

        self._spawns: Dict[str, dict] = {}
        if spawnsPath is not None and Path(spawnsPath).exists():
            with open(spawnsPath, "r") as f:
                self._spawns = json.load(f)

        self._nameToId = {name: data["id"] for name, data in self._cards.items()}
        self._idToName = {data["id"]: name for name, data in self._cards.items()}
        self._idToCard = {data["id"]: data for data in self._cards.values()}

        self._stats: Dict[str, CardStats] = {}
        self._variants: Dict[str, List[str]] = {}
        for name, data in self._cards.items():
            values = dict(data)
            variant, baseName = _variant_of(name, self._cards)
            stats = CardStats(
                name=name,
                id=int(data["id"]),
                elixir=data.get("elixir"),
                type=str(data.get("type", "troop")),
                variant=variant,
                base_name=baseName,
                values=MappingProxyType(values),
            )
            self._stats[name] = stats
            if baseName is not None:
                self._variants.setdefault(baseName, []).append(name)

    # ---- original API (used by perception / RL observation code) --------
    @property
    def vocabSize(self):
        return max(self._idToName, default=0) + 1

    def id(self, name):
        return self._nameToId.get(name, 0)

    def name(self, cardId):
        return self._idToName.get(cardId)

    def contains(self, name):
        return name in self._cards

    def _get(self, card):
        return self._idToCard.get(card, {}) if isinstance(card, int) else self._cards.get(card, {})

    def card(self, name):
        return self._cards.get(name)

    def cardById(self, cardId):
        return self._idToCard.get(cardId)

    def elixir(self, card):
        return self._get(card).get("elixir")

    def hp(self, card):
        return self._get(card).get("hp")

    def damage(self, card):
        return self._get(card).get("damage")

    def _statsFor(self, card) -> Optional[CardStats]:
        name = self._idToName.get(card) if isinstance(card, int) else card
        return self._stats.get(name) if name is not None else None

    def isBuilding(self, card):
        stats = self._statsFor(card)
        return bool(stats and stats.is_building)

    def isSpell(self, card):
        stats = self._statsFor(card)
        return bool(stats and stats.is_spell)

    def isAir(self, card):
        stats = self._statsFor(card)
        return bool(stats and stats.is_air)

    def isEvolution(self, card):
        stats = self._statsFor(card)
        return bool(stats and stats.is_evolution)

    def isHero(self, card):
        stats = self._statsFor(card)
        return bool(stats and stats.is_hero)

    def abilityElixir(self, card):
        return self._get(card).get("abilityElixir")

    # ---- normalised stats API ---------------------------------------------
    def stats(self, card) -> Optional[CardStats]:
        """Normalised ``CardStats`` for a Cards.json name or id."""
        return self._statsFor(card)

    def allStats(self) -> List[CardStats]:
        return list(self._stats.values())

    def names(self) -> List[str]:
        return list(self._cards.keys())

    def variant(self, card) -> Optional[str]:
        stats = self._statsFor(card)
        return stats.variant if stats else None

    def baseName(self, card) -> Optional[str]:
        stats = self._statsFor(card)
        return stats.base_name if stats else None

    def variantsOf(self, baseName: str) -> List[str]:
        return list(self._variants.get(baseName, []))

    def spawnLayout(self, name: str) -> Optional[dict]:
        """Raw CardSpawns.json entry (``units`` + ``maxDistanceX/Y``) or None."""
        return self._spawns.get(name)

    def cardsDeploying(self, unitName: str) -> List[str]:
        """Base cards whose CardSpawns.json layout deploys ``unitName``."""
        return [
            card for card, layout in self._spawns.items()
            if unitName in layout.get("units", {}) and self.variant(card) == VARIANT_BASE
        ]

    def spawnUnits(self, name: str) -> List[Tuple[str, int]]:
        """Sub-units deployed by a multi-unit card as ``[(unitName, count), ...]``.

        Composition comes from CardSpawns.json, but counts stated in Cards.json
        win (it is the authoritative stats file): ``count`` for single-unit
        swarms and ``<unit>Count`` style keys for mixed swarms.
        """
        layout = self._spawns.get(name)
        stats = self._stats.get(name)
        if not layout:
            return []
        units = [(unit, int(count)) for unit, count in layout.get("units", {}).items()]
        if stats is None:
            return units

        if len(units) == 1 and stats.count is not None:
            return [(units[0][0], stats.count)]

        overrides = {
            "Goblin": ("goblinCount",),
            "SpearGoblin": ("spearGoblinCount",),
            "BanditRascal": ("banditRascalCount",),
            "ArcherRascal": ("archerRascalCount",),
        }
        resolved = []
        for unit, count in units:
            for key in overrides.get(unit, ()):
                value = stats.number(key)
                if value is not None:
                    count = int(value)
                    break
            resolved.append((unit, count))
        return resolved
