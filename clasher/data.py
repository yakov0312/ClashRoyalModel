from typing import Dict, Optional
from pathlib import Path

from .card_types import CardDefinition, CardStatsCompat
from .factory.card_factory import card_from_gamedata
from .factory.stats_overlay import build_entries, get_card_database, load_behaviors
from .path import CARD_BEHAVIORS_FILE, CARD_SPAWNS_FILE, CARDS_FILE


class CardDataLoader:
    """Card definitions for the simulator.

    Every card and unit is named exactly as in ``data/cards/Cards.json``, which
    also supplies all numbers. ``CardBehaviors.json`` (same names) supplies
    behaviour templates only (see ``factory/stats_overlay.py``). Lookups are
    exact: there is no alias layer.
    """

    def __init__(
        self,
        behaviors_file: str | Path = CARD_BEHAVIORS_FILE,
        cards_file: str | Path = CARDS_FILE,
        spawns_file: str | Path = CARD_SPAWNS_FILE,
    ):
        self.behaviors_file = behaviors_file
        self.cards_file = cards_file
        self.spawns_file = spawns_file
        self._cards: Dict[str, CardStatsCompat] = {}
        self._card_definitions: Dict[str, CardDefinition] = {}

    @property
    def card_database(self):
        return get_card_database(str(self.cards_file), str(self.spawns_file))

    def load_card_definitions(self) -> Dict[str, CardDefinition]:
        if self._card_definitions:
            return self._card_definitions

        behaviors = load_behaviors(str(self.behaviors_file))
        card_definitions: Dict[str, CardDefinition] = {}
        for card_name, entry in build_entries(behaviors, self.card_database).items():
            try:
                card_definitions[card_name] = card_from_gamedata(entry)
            except Exception as exc:
                print(f"Warning: Could not build Cards.json definition for {card_name}: {exc}")

        self._card_definitions = card_definitions
        return self._card_definitions

    def load_cards(self) -> Dict[str, CardStatsCompat]:
        """Materialize compatibility stats from card definitions."""
        if self._cards:
            return self._cards
        definitions = self.load_card_definitions()
        self._cards = {
            name: CardStatsCompat.from_card_definition(card_def)
            for name, card_def in definitions.items()
        }
        return self._cards

    def get_card(self, name: str) -> Optional[CardStatsCompat]:
        """Card or unit stats by exact Cards.json name."""
        if not self._cards:
            self.load_cards()
        return self._cards.get(name)

    def get_card_definition(self, name: str) -> Optional[CardDefinition]:
        if not self._card_definitions:
            self.load_card_definitions()
        return self._card_definitions.get(name)

    def get_card_compat(self, name: str) -> Optional[CardStatsCompat]:
        """Alias for get_card to preserve API compatibility."""
        return self.get_card(name)

    def print_card_summary(self, name: str) -> None:
        """Print a detailed summary of a card's attributes"""
        card = self.get_card(name)
        if not card:
            print(f"Card '{name}' not found")
            return
            
        print(f"=== {card.name} ===")
        print(f"Type: {card.card_type} | Rarity: {card.rarity} | Cost: {card.mana_cost} elixir")
        if card.tribe:
            print(f"Tribe: {card.tribe}")
        if card.unlock_arena:
            print(f"Unlocks: {card.unlock_arena}")
            
        if card.hitpoints or card.damage:
            print(f"\\nCombat:")
            if card.hitpoints:
                print(f"  HP: {card.hitpoints}")
            if card.damage:
                print(f"  Damage: {card.damage}")
            if card.hit_speed:
                print(f"  Attack Speed: {card.hit_speed}ms")
                
        if card.range or card.sight_range or card.speed:
            print(f"\\nMovement & Range:")
            if card.range:
                print(f"  Attack Range: {card.range} tiles")
            if card.sight_range:
                print(f"  Sight Range: {card.sight_range} tiles")
            if card.speed:
                print(f"  Speed: {card.speed} tiles/min")
            if card.collision_radius:
                print(f"  Collision Radius: {card.collision_radius} tiles")
                
        if card.summon_count:
            print(f"\\nDeployment:")
            print(f"  Units Spawned: {card.summon_count}")
            if card.summon_radius:
                print(f"  Spawn Radius: {card.summon_radius} tiles")
            if card.summon_deploy_delay:
                print(f"  Spawn Delay: {card.summon_deploy_delay}ms")
                
        if card.attacks_ground is not None or card.attacks_air is not None:
            print(f"\\nTargeting:")
            if card.attacks_ground:
                print(f"  Attacks Ground: Yes")
            if card.attacks_air:
                print(f"  Attacks Air: Yes")
                
        if card.has_evolution:
            print(f"\\nEvolution: Available")
            
        if card.deploy_time or card.load_time:
            print(f"\\nTiming:")
            if card.deploy_time:
                print(f"  Deploy Time: {card.deploy_time}ms")
            if card.load_time:
                print(f"  Load Time: {card.load_time}ms")
