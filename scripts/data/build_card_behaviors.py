#!/usr/bin/env python3
"""One-time migration: legacy ``data/game/gamedata.json`` -> ``data/cards/CardBehaviors.json``.

The simulator names every card and unit exactly as ``data/cards/Cards.json``
does. ``gamedata.json`` is an old Supercell dump that uses internal names
(``Assassin`` for Bandit, ``Golemite`` for MiniGolem ...) and level-1 numbers.
The simulator only needs its *behaviour* data (projectile speeds and splash
radii, collision radii, sight ranges, target flags, nested spawn/area-effect
structure), so this script copies those templates out, keyed by Cards.json
names, with every nested unit reference renamed to its Cards.json name.

The name tables below exist only here, for this conversion; the simulator never
sees a legacy name. Numbers left in the templates are overridden at load time
by Cards.json wherever Cards.json has a value.

Run from the repo root:  python scripts/data/build_card_behaviors.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
GAMEDATA = ROOT / "data" / "game" / "gamedata.json"
CARDS = ROOT / "data" / "cards" / "Cards.json"
OUTPUT = ROOT / "data" / "cards" / "CardBehaviors.json"

# Cards.json card -> gamedata card entry used as behaviour template (None: no
# usable template, the simulator synthesises the card from Cards.json alone).
CARD_TEMPLATES: Dict[str, Optional[str]] = {
    "ElectroSpirit": "ElectroSpirit",
    "FireSpirit": "FireSpirits",
    "IceSpirit": "IceSpirits",
    "HealSpirit": "Heal",
    "Skeletons": "Skeletons",
    "Bats": "Bats",
    "Berserker": "Berserker",
    "Bomber": "Bomber",
    "Snowball": "Snowball",
    "Goblins": "Goblins",
    "SpearGoblins": "SpearGoblins",
    "Zap": "Zap",
    "Archers": "Archer",
    "Arrows": "Arrows",
    "Cannon": "Cannon",
    "Firecracker": "Firecracker",
    "GoblinGang": "GoblinGang",
    "Knight": "Knight",
    "Minions": "Minions",
    "RoyalDelivery": "RoyalDelivery",
    "SkeletonBarrel": "SkeletonBalloon",
    "Mortar": "Mortar",
    "SkeletonDragons": "SkeletonDragons",
    "Tesla": "Tesla",
    "Barbarians": "Barbarians",
    "MinionHorde": "MinionHorde",
    "Rascals": "Rascals",
    "EliteBarbarians": "AngryBarbarians",
    "RoyalGiant": "RoyalGiant",
    "RoyalRecruits": "RoyalRecruits",
    "IceGolem": "IceGolemite",
    "SuspiciousBush": "SuspiciousBush",
    "DartGoblin": "BlowdartGoblin",
    "Earthquake": "Earthquake",
    "ElixirGolem": "ElixirGolem",
    "MegaMinion": "MegaMinion",
    "Tombstone": "Tombstone",
    "BattleHealer": "BattleHealer",
    "BattleRam": "BattleRam",
    "BombTower": "BombTower",
    "Fireball": "Fireball",
    "FlyingMachine": "DartBarrell",
    "Furnace": None,  # now a troop; the gamedata "FirespiritHut" is the old building
    "GoblinCage": "GoblinCage",
    "GoblinDemolisher": "GoblinDemolisher",
    "GoblinHut": "GoblinHut",
    "HogRider": "HogRider",
    "MiniPEKKA": "MiniPekka",
    "Musketeer": "Musketeer",
    "Valkyrie": "Valkyrie",
    "Zappies": "MiniSparkys",
    "Giant": "Giant",
    "InfernoTower": "InfernoTower",
    "RoyalHogs": "RoyalHogs",
    "Wizard": "Wizard",
    "BarbarianHut": "BarbarianHut",
    "ElixirPump": "Elixir Collector",
    "Rocket": "Rocket",
    "ThreeMusketeers": "ThreeMusketeers",
    "Mirror": "Mirror",
    "BarbarianBarrel": "BarbLog",
    "GoblinCurse": "GoblinCurse",
    "Rage": "Rage",
    "WallBreakers": "Wallbreakers",
    "Clone": "Clone",
    "GoblinBarrel": "GoblinBarrel",
    "Guards": "SkeletonWarriors",
    "SkeletonArmy": "SkeletonArmy",
    "Tornado": "Tornado",
    "Vines": None,
    "BabyDragon": "BabyDragon",
    "DarkPrince": "DarkPrince",
    "Freeze": "Freeze",
    "GoblinDrill": None,  # gamedata models it as a digging troop that morphs; Cards.json: building
    "Hunter": "Hunter",
    "Poison": "Poison",
    "RuneGiant": "GiantBuffer",
    "Balloon": "Balloon",
    "Bowler": "Bowler",
    "CannonCart": "MovingCannon",
    "ElectroDragon": "ElectroDragon",
    "Executioner": "AxeMan",
    "Prince": "Prince",
    "Void": "DarkMagic",
    "Witch": "Witch",
    "GiantSkeleton": "GiantSkeleton",
    "GoblinGiant": "GoblinGiant",
    "Lightning": "Lightning",
    "XBow": "Xbow",
    "ElectroGiant": "ElectroGiant",
    "P-E-K-K-A": "Pekka",
    "Golem": "Golem",
    "Log": "Log",
    "Bandit": "Assassin",
    "Fisherman": "Fisherman",
    "IceWizard": "IceWizard",
    "Miner": "Miner",
    "Princess": "Princess",
    "RoyalGhost": "Ghost",
    "ElectroWizard": "ElectroWizard",
    "InfernoDragon": "InfernoDragon",
    "Lumberjack": "RageBarbarian",
    "MagicArcher": "EliteArcher",
    "MotherWitch": "WitchMother",
    "NightWitch": "DarkWitch",
    "Phoenix": "Phoenix",
    "GoblinMachine": "GoblinMachine",
    "Graveyard": "Graveyard",
    "RamRider": "RamRider",
    "Ronin": None,
    "Sparky": "ZapMachine",
    "SpiritEmpress": None,
    "SpiritEmpressWalk": None,
    "LavaHound": "LavaHound",
    "MegaKnight": "MegaKnight",
    "LittlePrince": "LittlePrince",
    "GoldenKnight": "GoldenKnight",
    "MightyMiner": "MightyMiner",
    "SkeletonKing": "SkeletonKing",
    "ArcherQueen": "ArcherQueen",
    "Goblinstein": "Goblinstein",
    "Monk": "Monk",
    "BossBandit": "BossBandit",
    "GiantMinion": None,
}


UNIT_TEMPLATES: Dict[str, str] = {
    "Larry": "Skeleton",
    "Bat": "Bat",
    "Goblin": "Goblin",
    "SpearGoblin": "SpearGoblin",
    "Archer": "Archer",
    "Minion": "Minion",
    "RoyalRecruit": "Recruit",
    "SkeletonDragon": "SkeletonDragon",
    "Barbarian": "Barbarian",
    "BanditRascal": "RascalBoy",
    "ArcherRascal": "RascalGirl",
    "EliteBarbarian": "AngryBarbarian",
    "Zappy": "MiniZapMachine",
    "RoyalHog": "RoyalHog",
    "WallBreaker": "Wallbreaker",
    "Guard": "SkeletonWarrior",
    "BushGoblin": "BushGoblin",
    "Brawler": "GoblinBrawler",
    "LavaPup": "LavaPups",
    "PhoenixEgg": "PhoenixEgg",
    "MiniGolem": "Golemite",
    "MiniElixirGolem": "ElixirGolem2",
    "ElixirGolemSmall": "ElixirGolem4",
    "CursedHog": "VoodooHog",
    "Doctor": "Goblinstein_doctor",
    "Monster": "Goblinstein",
    "PrinceGuardian": "ChampionGuard",
    "ThreeMusketeer": "ThreeMusketeer",
}


EXTRA_UNIT_ALIASES: Dict[str, str] = {
    "Goblin_Stab": "Goblin",
    "DeliveryRecruit": "RoyalRecruit",
    "Recruit_Chess": "RoyalRecruit",
    "SpearGoblinGiant": "SpearGoblin",
    "SkeletonKingSkeleton": "Larry",
    "Ram": "RamRider",
}


# Nested gamedata objects that are Cards.json units too.
EXTRA_UNIT_ALIASES["GiantSkeletonBomb"] = "GiantBomb"


def index_characters(entries) -> Dict[str, dict]:
    """gamedata character name -> first character dict found anywhere."""
    index: Dict[str, dict] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if isinstance(name, str) and name not in index and (
                "hitpoints" in node or "speed" in node and "sightRange" in node
            ):
                index[name] = node
            for key, value in node.items():
                if key != "evolvedSpellsData":
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for entry in entries:
        char = entry.get("summonCharacterData")
        if isinstance(char, dict) and char.get("name"):
            index.setdefault(char["name"], char)
    for entry in entries:
        walk(entry)
    return index


def rename_nested(node: Any, unit_names: Dict[str, str]) -> Any:
    """Rename nested unit references to Cards.json names (in place)."""
    if isinstance(node, dict):
        name = node.get("name")
        if isinstance(name, str) and name in unit_names:
            node["name"] = unit_names[name]
        for key in ("deathSpawnCharacter", "spawnCharacter", "summonCharacter"):
            value = node.get(key)
            if isinstance(value, str) and value in unit_names:
                node[key] = unit_names[value]
        for value in node.values():
            rename_nested(value, unit_names)
    elif isinstance(node, list):
        for value in node:
            rename_nested(value, unit_names)
    return node


def relabel_internal(node: Any, owner: str, known: Dict[str, Any], key: str = "") -> None:
    """Give nested non-card objects (projectiles, buffs, bombs...) owner-based
    labels such as ``ExecutionerProjectile`` instead of Supercell's internal names."""
    if isinstance(node, dict):
        name = node.get("name")
        if key and isinstance(name, str) and name not in known:
            role = key[0].upper() + key[1:]
            for suffix in ("DataData", "Data"):
                if role.endswith(suffix):
                    role = role[: -len(suffix)]
                    break
            node["name"] = f"{owner}{role}"
        for child_key, value in node.items():
            relabel_internal(value, owner, known, child_key if isinstance(value, (dict, list)) else key)
    elif isinstance(node, list):
        for value in node:
            relabel_internal(value, owner, known, key)


# Localisation ids / art paths from the dump; the simulator never uses them.
COSMETIC_KEYS = {"tid", "tidInfo", "iconFile", "highresImageFilename", "englishName", "source"}


def strip_keys(node: Any, keys) -> None:
    if isinstance(node, dict):
        for key in keys & node.keys():
            del node[key]
        for value in node.values():
            strip_keys(value, keys)
    elif isinstance(node, list):
        for value in node:
            strip_keys(value, keys)


def main() -> int:
    cards = json.loads(CARDS.read_text())
    raw = json.loads(GAMEDATA.read_text())["items"]["spells"]
    entries = {e["name"]: e for e in raw if e.get("name") and "manaCost" in e}
    characters = index_characters(raw)

    unit_names = {template: unit for unit, template in UNIT_TEMPLATES.items()}
    unit_names.update(EXTRA_UNIT_ALIASES)

    behaviors: Dict[str, dict] = {}
    for name, data in cards.items():
        if name.startswith(("Evo", "Hero")) and name[len("Evo" if name.startswith("Evo") else "Hero"):] in cards:
            continue  # variants are layered on their base card later
        if name in CARD_TEMPLATES:
            template = CARD_TEMPLATES[name]
            entry: Optional[dict] = entries.get(template) if template else None
            if entry is None:
                continue
            entry = copy.deepcopy(entry)
            entry.pop("evolvedSpellsData", None)
            char = entry.get("summonCharacterData")
            rename_nested(entry, unit_names)
            entry["name"] = name
            if isinstance(char, dict):
                char["name"] = name
                if name == "Lumberjack":
                    # His death bottle is the Rage card, cast by a card mechanic.
                    char.pop("deathSpawnCharacterData", None)
                    char.pop("deathSpawnCount", None)
            behaviors[name] = entry
            continue
        char = characters.get(UNIT_TEMPLATES.get(name, name))
        if char is None:
            continue
        char = rename_nested(copy.deepcopy(char), unit_names)
        char["name"] = name
        behaviors[name] = {"name": name, "summonCharacterData": char}

    # Princess Tower troop (tower behaviour; numbers live in Towers.json).
    for entry in raw:
        if entry.get("name") == "King_PrincessTowers":
            behaviors["PrincessTower"] = {
                "name": "PrincessTower",
                "summonCharacterData": dict(entry["statCharacterData"], name="PrincessTower"),
            }

    for name, entry in behaviors.items():
        relabel_internal(entry, name, cards)
        strip_keys(entry, COSMETIC_KEYS)

    OUTPUT.write_text(json.dumps(behaviors, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(behaviors)} behaviour templates -> {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
