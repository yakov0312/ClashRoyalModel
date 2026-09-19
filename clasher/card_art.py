"""Pick the card art to show for any simulator entity name.

Art files in ``shared/cards`` are named exactly like Cards.json *cards*
(``Skeletons.png``, ``Golem.png``). Sub-units (``Larry``, ``MiniGolem``,
``LavaPup``) have no art of their own, so they show the card that brings
them into play, found from the data rather than a hand-written table:

1. ``<name>.png`` itself,
2. a card whose CardSpawns.json layout deploys the unit (Larry -> Skeletons),
3. a card whose behaviour template spawns it (MiniGolem -> Golem on death).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .factory.stats_overlay import get_card_database, load_behaviors
from .path import PROJECT_ROOT

CARD_IMAGE_DIR = PROJECT_ROOT / "shared" / "cards"


def _nested_unit_parents() -> Dict[str, List[str]]:
    db = get_card_database()
    parents: Dict[str, List[str]] = {}

    def walk(node, owner: str, is_root: bool) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if not is_root and isinstance(name, str) and name != owner and db.contains(name):
                parents.setdefault(name, [])
                if owner not in parents[name]:
                    parents[name].append(owner)
            for value in node.values():
                walk(value, owner, False)
        elif isinstance(node, list):
            for value in node:
                walk(value, owner, False)

    for card, entry in load_behaviors().items():
        stats = db.stats(card)
        if stats is not None and stats.is_playable:
            walk(entry, card, True)
    return parents


@lru_cache(maxsize=None)
def art_name(name: str, image_dir: str = str(CARD_IMAGE_DIR)) -> Optional[str]:
    """Name of the art file (without ``.png``) to draw for ``name``, or None."""
    directory = Path(image_dir)
    if (directory / f"{name}.png").exists():
        return name
    candidates = get_card_database().cardsDeploying(name) + _nested_unit_parents_cached().get(name, [])
    for card in candidates:
        if (directory / f"{card}.png").exists():
            return card
    return None


@lru_cache(maxsize=1)
def _nested_unit_parents_cached() -> Dict[str, List[str]]:
    return _nested_unit_parents()
