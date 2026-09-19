from .archer_queen import ArcherQueenCloak
from .bandit import BanditDash
from .electro_dragon import ElectroDragonChainLightning
from .electro_spirit import ElectroSpiritChain
from .electro_wizard import ElectroWizardSpawnZap, ElectroWizardStunAttack
from .firecracker import FirecrackerRecoil
from .fisherman import FishermanHook
from .hog_rider import HogRiderJump
from .ice_golem import IceGolemChill
from .ice_spirit import IceSpiritFreeze
from .lumberjack import LumberjackRage
from .magic_archer import MagicArcherPierce
from .mega_knight import MegaKnightSlam
from .royal_ghost import RoyalGhostFade
from .sparky import SparkyChargeUp
from .valkyrie import ValkyrieSpin
from .wallbreakers import WallBreakersDemolition

# Keyed by Cards.json unit names (the name each spawned entity carries).
# Battle Ram needs no entry: its Cards.json ``chargeDamage`` is applied by the
# generic charge logic in ``entities.Troop``.
CARD_MECHANICS = {
    'ArcherQueen': [ArcherQueenCloak],
    'Bandit': [BanditDash],
    'ElectroDragon': [ElectroDragonChainLightning],
    'ElectroSpirit': [ElectroSpiritChain],
    'ElectroWizard': [ElectroWizardSpawnZap, ElectroWizardStunAttack],
    'Firecracker': [FirecrackerRecoil],
    'Fisherman': [FishermanHook],
    'HogRider': [HogRiderJump],
    'IceGolem': [IceGolemChill],
    'IceSpirit': [IceSpiritFreeze],
    'Lumberjack': [LumberjackRage],
    'MagicArcher': [MagicArcherPierce],
    'MegaKnight': [MegaKnightSlam],
    'RoyalGhost': [RoyalGhostFade],
    'Sparky': [SparkyChargeUp],
    'Valkyrie': [ValkyrieSpin],
    'WallBreaker': [WallBreakersDemolition],
}

__all__ = [
    'ArcherQueenCloak', 'BanditDash', 'ElectroDragonChainLightning',
    'ElectroSpiritChain', 'ElectroWizardSpawnZap', 'ElectroWizardStunAttack', 'FirecrackerRecoil', 'FishermanHook',
    'HogRiderJump', 'IceGolemChill', 'IceSpiritFreeze', 'LumberjackRage',
    'MagicArcherPierce', 'MegaKnightSlam', 'RoyalGhostFade',
    'SparkyChargeUp', 'ValkyrieSpin', 'WallBreakersDemolition', 'CARD_MECHANICS'
]
