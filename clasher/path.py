from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CARDS_DIR = DATA_DIR / "cards"
GAME_DATA_DIR = DATA_DIR / "game"

CARDS_FILE = CARDS_DIR / "Cards.json"
CARD_SPAWNS_FILE = CARDS_DIR / "CardSpawns.json"
DECKS_FILE = GAME_DATA_DIR / "decks.json"
CARD_BEHAVIORS_FILE = CARDS_DIR / "CardBehaviors.json"
TOWERS_FILE = CARDS_DIR / "Towers.json"
