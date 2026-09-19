import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from perception.arena.ArenaCalibration import ArenaCalibration
from shared.cardRegistry import CardDatabase
from perception.digits.DigitNet import DigitNet


@dataclass
class Hand:
    handCards: list[int] = field(default_factory=lambda: [0] * 4)
    nextCard: int = 0


@dataclass
class SideState:
    leftTowerHp: int = 0
    rightTowerHp: int = 0
    kingTowerHp: int = 0


@dataclass
class State:
    own: SideState
    enemy: SideState
    elixir: int
    time: int
    hand: Hand
    gameAlive: bool
    won: bool | None


class HUDReader:
    redLower = np.array([80, 25, 210], dtype=np.uint8)
    redUpper = np.array([165, 130, 255], dtype=np.uint8)
    blueLower = np.array([105, 50, 0], dtype=np.uint8)
    blueUpper = np.array([255, 210, 112], dtype=np.uint8)
    whiteLower = np.array([231, 208, 182], dtype=np.uint8)
    whiteUpper = np.array([254, 255, 255], dtype=np.uint8)

    cardTemplateExtensions = (".png", ".jpg", ".jpeg")
    cardMinConfidence = 0.53
    cardScale = 0.5
    handChangeThreshold = 2.0

    towerPairs = {
        "allyLeft": "allyPair", "allyRight": "allyPair",
        "enemyLeft": "enemyPair", "enemyRight": "enemyPair"
    }

    towerStateNames = {
        "allyLeft": ("own", "leftTowerHp"),
        "allyRight": ("own", "rightTowerHp"),
        "allyKing": ("own", "kingTowerHp"),
        "enemyLeft": ("enemy", "leftTowerHp"),
        "enemyRight": ("enemy", "rightTowerHp"),
        "enemyKing": ("enemy", "kingTowerHp"),
    }

    def __init__(self, calibration: ArenaCalibration, config, cards: CardDatabase):

        self.calib, self._cards = calibration, cards
        self.cardTemplates = config.get("cardTemplates", "")
        self.cardMinConfidence = config.get("cardMinConfidence", 0.53)
        self.cardScale = config.get("cardScale", 0.5)
        self.handChangeThreshold = config.get("handChangeThreshold", 2.0)
        self.digitMinConfidence = config.get("digitMinConfidence", 0.85)

        self._towerPosCache, self._pairYCache = {}, {}
        self._towerState = {name: 0 for name in self.towerStateNames}
        self._kingMissingFrames = {"allyKing": 0, "enemyKing": 0}
        self.gameAlive, self.won = True, None

        self._cardTemplates, self._preparedCardTemplates = self._loadCardTemplates(), {}
        self.hand = Hand()
        self._previousHandCrops, self._previousNextCrop = [None] * 4, None

        self._digitDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._digitModel = DigitNet()
        self._digitModel.load_state_dict(torch.load(config.get("digitModel", ""), map_location=self._digitDevice, weights_only=True))
        self._digitModel.to(self._digitDevice).eval()

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 28, 28, device=self._digitDevice)
            for _ in range(3): self._digitModel(dummy)

    def readFrame(self, frame: np.ndarray) -> State:
        height, width = frame.shape[:2]
        if self.gameAlive:
            self._readTowers(frame, width, height)

        return State(
            own=SideState(self._towerState["allyLeft"], self._towerState["allyRight"], self._towerState["allyKing"]),
            enemy=SideState(self._towerState["enemyLeft"], self._towerState["enemyRight"], self._towerState["enemyKing"]),
            elixir=self._readElixir(frame, width, height) or 0,
            time=self._readTime(frame, width, height) or 0,
            hand=self.processHand(frame),
            gameAlive=self.gameAlive,
            won=self.won,
        )

    def _readTowers(self, frame, width, height):
        for name in self.towerStateNames:
            if name in self.calib.towerBoxes:
                self._towerState[name] = self._readTower(frame, width, height, name)

    def _isKingTower(self, name):
        return name.endswith("King")

    def _readTower(self, frame, width, height, name):
        king = self._isKingTower(name)
        if not king and self._towerState[name] == -1:
            return -1

        lower, upper, walkDir = self._fillParamsFor(name)

        if name in self._towerPosCache:
            box = self._towerPosCache[name]
            crop = self._crop(frame, box)
            if self._hasFill(crop, lower, upper):
                self._kingMissingFrames[name] = 0
                return self._readHp(crop)
            if not king:
                return -1

        box = self._tryPairedPosition(name, width, height)
        if box is not None and self._hasFill(self._crop(frame, box), lower, upper):
            self._towerPosCache[name] = box
            self._kingMissingFrames[name] = 0
            return self._readHp(self._crop(frame, box))

        box = self.calib.scaleHudBox(self.calib.towerBoxes[name], width, height)
        if self._hasFill(self._crop(frame, box), lower, upper):
            self._towerPosCache[name] = box
            self._rememberPairY(name, box)
            self._kingMissingFrames[name] = 0
            return self._readHp(self._crop(frame, box))

        box = self._locateBar(frame, box, height, lower, upper, walkDir)
        if box is not None:
            self._towerPosCache[name] = box
            self._rememberPairY(name, box)
            self._kingMissingFrames[name] = 0
            return self._readHp(self._crop(frame, box))

        return self._handleMissingKing(name) if king else -1

    def _handleMissingKing(self, name):
        self._kingMissingFrames[name] += 1
        if self._kingMissingFrames[name] < 2:
            return 0

        self.gameAlive = False
        self.won = name == "enemyKing"
        return 0

    def _readHp(self, crop):
        return self._ocrDigits(crop, 4) or 0

    def _fillParamsFor(self, name):
        return (self.blueLower, self.blueUpper, 1) if name.startswith("ally") else (self.redLower, self.redUpper, -1)

    def _tryPairedPosition(self, name, width, height):
        group = self.towerPairs.get(name)
        if group not in self._pairYCache:
            return None
        x, _, w, h = self.calib.scaleHudBox(self.calib.towerBoxes[name], width, height)
        return x, self._pairYCache[group], w, h

    def _rememberPairY(self, name, box):
        group = self.towerPairs.get(name)
        if group is not None:
            self._pairYCache[group] = box[1]

    def _locateBar(self, frame, box, frameHeight, lower, upper, walkDir):
        x, y, width, height = box
        centerX = x + width // 2
        startY = y + height
        searchRange = int(self.calib.towerSearchRange * frameHeight / self.calib.hudReferenceHeight)

        barY = None
        for offset in range(searchRange + 1):
            for searchY in (startY + offset, startY - offset):
                if self._isColorRun(frame, centerX, searchY, lower, upper):
                    barY = searchY
                    break
            if barY is not None:
                break

        if barY is None:
            return None

        yCursor = barY
        for _ in range(searchRange):
            if self._isColorRun(frame, centerX, yCursor, self.whiteLower, self.whiteUpper):
                break
            yCursor += walkDir

        numberY = yCursor - 2 * walkDir
        return centerX - width // 2, numberY - height // 2, width, height

    def _crop(self, frame, box):
        x, y, width, height = box
        return frame[max(0, y):max(0, y) + height, max(0, x):max(0, x) + width]

    def _isColorRun(self, frame, centerX, y, lower, upper, width=5, minPixels=3):
        if y < 0 or y >= frame.shape[0]:
            return False
        xStart = max(0, centerX - width // 2)
        row = frame[y:y + 1, xStart:min(frame.shape[1], xStart + width)]
        return cv2.countNonZero(cv2.inRange(row, lower, upper)) >= minPixels

    def _hasFill(self, crop, lower, upper, minPixels=15):
        return crop.size > 0 and cv2.countNonZero(cv2.inRange(crop, lower, upper)) >= minPixels

    def _getDigitComponents(self, binary):
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components = [(x, y, w, h) for x, y, w, h, area in stats[1:] if area >= 8]
        components.sort(key=lambda c: c[0])
        return components

    def _glyphToTensor(self, glyph):
        rgb = cv2.cvtColor(glyph, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (28, 28), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)

    def _predictDigits(self, glyphs):
        if not glyphs:
            return []

        batch = torch.stack([self._glyphToTensor(g) for g in glyphs]).to(self._digitDevice)

        with torch.inference_mode():
            probabilities = torch.softmax(self._digitModel(batch), dim=1)
            confidences, predictions = probabilities.max(dim=1)

        return [(int(p) if float(c) >= self.digitMinConfidence else None, float(c))
                for c, p in zip(confidences, predictions)]

    def _ocrDigits(self, crop, maxDigits, upscale=3, threshold=228):
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        value = cv2.resize(hsv[:, :, 2], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        saturation = cv2.resize(hsv[:, :, 1], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        binary = cv2.inRange(
            cv2.merge([saturation, value]),
            np.array([0, threshold], dtype=np.uint8),
            np.array([100, 255], dtype=np.uint8),
        )

        components = self._getDigitComponents(binary)
        if not components:
            return None

        glyphs = []
        for x, y, width, height in components:
            ox, oy = max(0, int(x / upscale)), max(0, int(y / upscale))
            ow, oh = max(1, int(np.ceil(width / upscale))), max(1, int(np.ceil(height / upscale)))
            glyph = crop[oy:oy + oh, ox:ox + ow]
            if glyph.size:
                glyphs.append(glyph)

        digits = [str(d) for d, _ in self._predictDigits(glyphs) if d is not None][:maxDigits]
        return int("".join(digits)) if digits else None

    def _ocrTime(self, crop):
        value = self._ocrDigits(crop, 3)
        if value is None:
            return None
        return (value // 100) * 60 + value % 100 if value > 99 else value

    def _readTime(self, frame, width, height):
        box = self.calib.scaleHudBox(self.calib.timeBox, width, height)
        return self._ocrTime(self._crop(frame, box))

    def _readElixir(self, frame, width, height):
        box = self.calib.scaleHudBox(self.calib.elixirBox, width, height)
        return self._ocrDigits(self._crop(frame, box), 2)

    def _loadCardTemplates(self):
        templates = {}
        if not os.path.isdir(self.cardTemplates):
            return templates

        for fileName in sorted(os.listdir(self.cardTemplates)):
            if not fileName.lower().endswith(self.cardTemplateExtensions):
                continue

            name = os.path.splitext(fileName)[0]
            if self._cards.id(name) == 0:
                continue

            image = cv2.imread(os.path.join(self.cardTemplates, fileName), cv2.IMREAD_COLOR)
            if image is not None:
                templates[name] = image

        return templates

    def _prepareCardTemplates(self, cropWidth, cropHeight):
        key = cropWidth, cropHeight
        if key in self._preparedCardTemplates:
            return self._preparedCardTemplates[key]

        targetWidth = max(1, int(cropWidth * self.cardScale))
        targetHeight = max(1, int(cropHeight * self.cardScale))
        names, images = [], []

        for name, template in self._cardTemplates.items():
            resized = cv2.resize(template, (targetWidth, targetHeight), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
            centered = gray.reshape(-1) - gray.mean()
            norm = np.linalg.norm(centered)

            if norm <= 1e-6:
                continue

            names.append(name)
            images.append(centered / norm)

        result = names, np.stack(images) if images else np.empty((0, targetWidth * targetHeight), dtype=np.float32)
        self._preparedCardTemplates[key] = result
        return result

    def _matchCard(self, crop):
        if crop.size == 0 or not self._cardTemplates:
            return None, 0.0

        height, width = crop.shape[:2]
        names, templates = self._prepareCardTemplates(width, height)
        if not names:
            return None, 0.0

        targetWidth = max(1, int(width * self.cardScale))
        targetHeight = max(1, int(height * self.cardScale))
        small = cv2.resize(crop, (targetWidth, targetHeight), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        centered = gray.reshape(-1) - gray.mean()
        norm = np.linalg.norm(centered)

        if norm <= 1e-6:
            return None, 0.0

        scores = templates @ (centered / norm)
        index = int(np.argmax(scores))
        score = float(scores[index])

        return (names[index], score) if score >= self.cardMinConfidence else (None, score)

    def _cardCropChanged(self, crop, previous):
        if previous is None or crop.shape != previous.shape:
            return True

        current = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        return float(cv2.absdiff(current, previous).mean()) >= self.handChangeThreshold

    def _matchCardBoxWithLiftFallback(self, frame, box, frameHeight):
        crop = self._crop(frame, box)
        name, score = self._matchCard(crop)
        if name is not None:
            return name, score

        move = self.calib.scaleCardMoveBy(frameHeight)
        if move <= 0:
            return None, score

        x, y, width, height = box
        lifted = self._crop(frame, (x, y - move, width, height))
        liftedName, liftedScore = self._matchCard(lifted)

        return (liftedName, liftedScore) if liftedName is not None else (None, max(score, liftedScore))

    def getHand(self):
        return self.hand

    def processHand(self, frame):
        frameHeight, frameWidth = frame.shape[:2]
        handCards = []

        for index, slot in enumerate(("slot1", "slot2", "slot3", "slot4")):
            box = self.calib.cardBoxes.get(slot)

            if box is None:
                handCards.append(0)
                continue

            boxPx = self.calib.scaleHudBox(box, frameWidth, frameHeight)
            crop = self._crop(frame, boxPx)

            if not self._cardCropChanged(crop, self._previousHandCrops[index]):
                handCards.append(self.hand.handCards[index])
                continue

            name, _ = self._matchCardBoxWithLiftFallback(frame, boxPx, frameHeight)
            handCards.append(self._cards.id(name))
            self._previousHandCrops[index] = crop.copy()

        nextCard = self.hand.nextCard

        if self.calib.nextCardBox is not None:
            box = self.calib.scaleHudBox(self.calib.nextCardBox, frameWidth, frameHeight)
            crop = self._crop(frame, box)

            if self._cardCropChanged(crop, self._previousNextCrop):
                name, _ = self._matchCard(crop)
                nextCard = self._cards.id(name)
                self._previousNextCrop = crop.copy()

        self.hand = Hand(handCards, nextCard)
        return self.hand

    def updateHand(self, playedCardSlot):
        if 0 < playedCardSlot < len(self.hand.handCards):
            self.hand.handCards[playedCardSlot] = self.hand.nextCard
            self.hand.nextCard = -1
            self._previousHandCrops[playedCardSlot] = None