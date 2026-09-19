from __future__ import annotations
from dataclasses import dataclass
import os
import cv2
import numpy as np
import yaml


BOARD_WIDTH = 18
BOARD_HEIGHT = 32

@dataclass(frozen=True)
class TowerHP:
    value: int | None
    alive: bool


class ArenaCalibration:
    def __init__(self, config):

        arenaTemplates = config.get("arenaTemplates")
        calibrations = config.get("calibrations")

        with open(calibrations, "r") as f:
            data = yaml.safe_load(f)

        self.arenas = {}
        for name, arena in data["arenas"].items():
            imagePath = os.path.join(arenaTemplates, arena["Image"])
            image = cv2.imread(imagePath)
            if image is None:
                print(f"Could not load arena image: {imagePath}")
                continue

            calibration = arena["calibration"]
            src = np.array([
                calibration["topLeft"], calibration["topRight"],
                calibration["bottomLeft"], calibration["bottomRight"]
            ], dtype=np.float32)
            dst = np.array([[0, 0], [BOARD_WIDTH, 0], [0, BOARD_HEIGHT], [BOARD_WIDTH, BOARD_HEIGHT]], dtype=np.float32)

            self.arenas[name] = {
                "image": image,
                "resolution": tuple(arena["resolution"]),
                "matrix": cv2.getPerspectiveTransform(src, dst),
                "inverseMatrix": cv2.getPerspectiveTransform(dst, src)
            }

        # --- HUD: multiple hand-calibrated references, closest one is auto-selected per resolution ---
        hud = data["HUD"]
        self.towerSearchRange = hud.get("towerSearchRange", 40)

        self.hudReferences = []
        for ref in hud["references"]:
            width, height = ref["resolution"]
            cards = ref.get("cards", {})
            cardBoxes = {name: tuple(box) for name, box in cards.items() if name.startswith("slot")}
            nextCardBox = tuple(cards["nextCard"]) if "nextCard" in cards else None
            cardMoveBy = int(cards.get("cardMoveBy", 0))

            self.hudReferences.append({
                "resolution": (width, height),
                "aspect": width / height,
                "timeBox": tuple(ref["timeBox"]),
                "elixirBox": tuple(ref["elixirBox"]),
                "towerBoxes": {name: tuple(box) for name, box in ref["towerBoxes"].items()},
                "cardBoxes": cardBoxes,
                "nextCardBox": nextCardBox,
                "cardMoveBy": cardMoveBy,
            })

        if not self.hudReferences:
            raise ValueError("No HUD references found in calibration file")

        self._activeHudRef = self.hudReferences[0]
        self._lastTargetRes = None
        self._scaledBoxCache = {}
        self.arenaName = None

    @staticmethod
    def _resizeToSmaller(image1, image2):
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        size = (min(w1, w2), min(h1, h2))

        if (w1, h1) != size:
            image1 = cv2.resize(image1, size)
        if (w2, h2) != size:
            image2 = cv2.resize(image2, size)

        return image1, image2

    def match(self, image, excellentThreshold=0.95):
        bestName, bestScore = None, -1.0

        for name, arena in self.arenas.items():
            template, current = self._resizeToSmaller(arena["image"], image)
            h, w = template.shape[:2]
            rectW, rectH = int(w * 0.65), int(w * 0.65 * 0.45)
            x1, y1 = (w - rectW) // 2, (h - rectH) // 2

            templateCrop = template[y1:y1 + rectH, x1:x1 + rectW]
            currentCrop = current[y1:y1 + rectH, x1:x1 + rectW]
            score = cv2.matchTemplate(currentCrop, templateCrop, cv2.TM_CCOEFF_NORMED)[0][0]

            if score > bestScore:
                bestName, bestScore = name, score
            if score >= excellentThreshold:
                self.arenaName = name
                print(f"Excellent match: {name} ({score:.4f})")
                return name

        self.arenaName = bestName
        return bestName

    def _arena(self):
        if self.arenaName is None:
            raise RuntimeError("No arena selected. Call match() first.")
        return self.arenas[self.arenaName]

    def pixelToTile(self, px: float, py: float) -> tuple[int, int]:
        point = np.array([[[px, py]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(point, self._arena()["matrix"])[0][0]
        return max(0, min(BOARD_WIDTH - 1, int(x))), max(0, min(BOARD_HEIGHT - 1, int(y)))

    def tileToPixel(self, tileX: int, tileY: int, center=True) -> tuple[float, float]:
        tx, ty = tileX + 0.5 if center else float(tileX), tileY + 0.5 if center else float(tileY)
        point = np.array([[[tx, ty]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(point, self._arena()["inverseMatrix"])[0][0]
        return float(x), float(y)

    # --- HUD reference selection (transparent to callers) ---

    def _ensureReference(self, targetWidth: int, targetHeight: int):
        """Auto-select the closest-aspect-ratio HUD reference, cached per (width, height)."""
        if self._lastTargetRes == (targetWidth, targetHeight):
            return
        targetAspect = targetWidth / targetHeight
        self._activeHudRef = min(self.hudReferences, key=lambda r: abs(r["aspect"] - targetAspect))
        self._lastTargetRes = (targetWidth, targetHeight)

    # --- Top-level properties, same names/shape as before, backed by the active reference ---

    @property
    def hudReferenceWidth(self):
        return self._activeHudRef["resolution"][0]

    @property
    def hudReferenceHeight(self):
        return self._activeHudRef["resolution"][1]

    @property
    def timeBox(self):
        return self._activeHudRef["timeBox"]

    @property
    def elixirBox(self):
        return self._activeHudRef["elixirBox"]

    @property
    def towerBoxes(self):
        return self._activeHudRef["towerBoxes"]

    @property
    def cardBoxes(self):
        return self._activeHudRef["cardBoxes"]

    @property
    def nextCardBox(self):
        return self._activeHudRef["nextCardBox"]

    @property
    def cardMoveBy(self):
        return self._activeHudRef["cardMoveBy"]

    # --- Scaling (same signatures as before) ---

    def scaleHudBox(self, box, targetWidth, targetHeight):
        key = (box, targetWidth, targetHeight)
        cached = self._scaledBoxCache.get(key)
        if cached is not None:
            return cached

        self._ensureReference(targetWidth, targetHeight)
        scale = targetHeight / self.hudReferenceHeight
        offsetX = (targetWidth - self.hudReferenceWidth * scale) / 2
        x, y, w, h = box
        result = round(x * scale + offsetX), round(y * scale), round(w * scale), round(h * scale)
        self._scaledBoxCache[key] = result
        return result

    def scaleCardMoveBy(self, targetHeight: int) -> int:
        return int(self.cardMoveBy * targetHeight / self.hudReferenceHeight)


def footPoint(boxXyxy):
    x1, y1, x2, y2 = boxXyxy
    return (x1 + x2) / 2, y2