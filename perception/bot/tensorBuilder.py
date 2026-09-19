from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from perception.arena.ArenaCalibration import footPoint
from shared.cardRegistry import CardDatabase
from shared.observationSpec import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    TOWER_POSITIONS,
    staticUnwalkableMask,
    timeFeature,
    towerHpFeature,
    towerSize,
)


SPEC_VERSION = 2

N_MAX, NUM_HAND_SLOTS = 40, 4
CARD_SLOTS = NUM_HAND_SLOTS + 1

DOUBLE_ELIXIR_AT_REMAINING, TRIPLE_ELIXIR_AT_REMAINING = 120, 60


@dataclass
class Observation:
    board: torch.Tensor
    entities: torch.Tensor
    entityMask: torch.Tensor
    globalFeatures: torch.Tensor
    vocabSize: int
    warnings: list[str]


class TensorBuilder:
    def __init__(self, cards: CardDatabase):
        self.cards = cards
        self._staticUnwalkable = self._loadStaticUnwalkableMask()

    @staticmethod
    def _loadStaticUnwalkableMask():
        return staticUnwalkableMask()

    def _buildBoard(self, detections, calibration, hudState):
        board = np.empty((3, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        board[0] = self._staticUnwalkable
        board[1:] = 0.0

        for side in ("own", "enemy"):
            state = getattr(hudState, side)

            for tower in ("left", "right", "king"):
                if getattr(state, f"{tower}TowerHp") != -1:
                    continue

                x, y = TOWER_POSITIONS[side][tower]
                size = towerSize(tower)
                board[0, y:y + size, x:x + size] = 0.0

        for detection in detections:
            px, py = footPoint(detection.boxXyxy)
            tileX, tileY = calibration.pixelToTile(px, py)

            if 0 <= tileX < BOARD_WIDTH and 0 <= tileY < BOARD_HEIGHT:
                board[1 if detection.ownUnit else 2, tileY, tileX] = 1.0

        return board

    def _buildEntities(self, detections, calibration, warnings):
        entities = np.zeros((N_MAX, 6), dtype=np.float32)
        count = 0

        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if count >= N_MAX:
                warnings.append(f"entity overflow: dropped {len(detections) - count} detections")
                break

            px, py = footPoint(detection.boxXyxy)
            tileX, tileY = calibration.pixelToTile(px, py)

            if not 0 <= tileX < BOARD_WIDTH or not 0 <= tileY < BOARD_HEIGHT:
                continue

            entities[count] = [
                self.cards.id(detection.className),
                0.0 if detection.ownUnit else 1.0,
                np.clip(detection.hpPct, 0.0, 1.0),
                tileX / BOARD_WIDTH,
                tileY / BOARD_HEIGHT,
                1.0,
            ]
            count += 1

        return entities, entities[:, 5].astype(bool)

    @staticmethod
    def _towerHp(hp):
        return towerHpFeature(hp)

    @staticmethod
    def _crownsWon(opponentState):
        """Crowns earned against ``opponentState`` (its towers read -1 once destroyed)."""
        if opponentState.kingTowerHp == -1:
            return 3
        return int(opponentState.leftTowerHp == -1) + int(opponentState.rightTowerHp == -1)

    def _buildGlobalFeatures(self, hudState, warnings):
        features = np.zeros(13 + CARD_SLOTS, dtype=np.float32)
        time = hudState.time or 0
        V = self.cards.vocabSize

        features[:5] = (
            timeFeature(time),
            float(time <= DOUBLE_ELIXIR_AT_REMAINING),
            float(time <= TRIPLE_ELIXIR_AT_REMAINING),
            float(time <= 0),
            np.clip(hudState.elixir / 10.0, 0.0, 1.0),
        )

        features[5] = self._crownsWon(hudState.enemy) / 3.0
        features[6] = self._crownsWon(hudState.own) / 3.0

        features[7:13] = (
            self._towerHp(hudState.own.kingTowerHp),
            self._towerHp(hudState.own.leftTowerHp),
            self._towerHp(hudState.own.rightTowerHp),
            self._towerHp(hudState.enemy.kingTowerHp),
            self._towerHp(hudState.enemy.leftTowerHp),
            self._towerHp(hudState.enemy.rightTowerHp),
        )

        cards = (*hudState.hand.handCards, hudState.hand.nextCard)

        for i, cardId in enumerate(cards):
            if 0 <= cardId < V:
                features[13 + i] = cardId
            else:
                if cardId >= V:
                    warnings.append(f"card slot {i}: cardId {cardId} >= vocab {V}")

        return features

    def build(self, detections, calibration, hudState):
        warnings = []

        board = self._buildBoard(detections, calibration, hudState)
        entities, entityMask = self._buildEntities(detections, calibration, warnings)
        globalFeatures = self._buildGlobalFeatures(hudState, warnings)

        return Observation(
            board=torch.from_numpy(board),
            entities=torch.from_numpy(entities),
            entityMask=torch.from_numpy(entityMask),
            globalFeatures=torch.from_numpy(globalFeatures),
            vocabSize=self.cards.vocabSize,
            warnings=warnings,
        )