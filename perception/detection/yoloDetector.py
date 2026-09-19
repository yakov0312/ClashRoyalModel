from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

IMG_SIZE = 896

HP_BAR_CLASS = "HpBar"
BADGE_CLASS = "BadgeLevel"


@dataclass(frozen=True)
class UnitDetection:
    className: str
    confidence: float
    boxXyxy: tuple[float, float, float, float]
    hpPct: float
    ownUnit: bool
    trackId: int = -1


@dataclass(frozen=True)
class _RawDetection:
    className: str
    confidence: float
    boxXyxy: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _clipBox(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a box to valid image bounds. Prevents negative indices from
    silently wrapping around to the other end of the array during slicing,
    which was the source of most of the bogus 'enemy' calls before."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    interX1, interY1 = max(ax1, bx1), max(ay1, by1)
    interX2, interY2 = min(ax2, bx2), min(ay2, by2)
    interW, interH = max(0.0, interX2 - interX1), max(0.0, interY2 - interY1)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


def _medianFilter1d(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(mask.astype(np.uint8), pad, mode="edge")
    out = np.empty_like(mask, dtype=np.uint8)
    for i in range(mask.shape[0]):
        window = padded[i:i + kernel]
        out[i] = 1 if window.sum() > kernel // 2 else 0
    return out.astype(bool)


def _isInEnemyZone(box: tuple[float, float, float, float], frameHeight: int, enemyZoneYFrac: float) -> bool:
    unitTopY = box[1]
    return unitTopY <= frameHeight * enemyZoneYFrac


def _readHpBar(barCropBgr: np.ndarray) -> float:
    """Returns hp fraction only.

    HP bars are colored by remaining health (green -> yellow -> red), not
    by team -- there is no reliable side signal in this crop. Reading side
    from its hue means a full-health own unit (green, hue ~60) and a
    nearly-dead own unit (red, hue ~0) can get opposite, both-wrong side
    answers purely from how much health they have left. Side must come
    from the badge instead.
    """
    h, w = barCropBgr.shape[:2]
    if h == 0 or w == 0:
        return 1.0

    hsv = cv2.cvtColor(barCropBgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]

    marginY = max(1, int(h * 0.25))
    marginX = max(1, int(w * 0.03))

    bandSat = sat[marginY:h - marginY, marginX:w - marginX]
    if bandSat.size == 0:
        bandSat = sat

    SAT_THRESHOLD = 130.0
    colSaturation = bandSat.astype(np.float32).mean(axis=0)
    filledMask = _medianFilter1d(colSaturation > SAT_THRESHOLD, kernel=5)

    totalWidth = filledMask.shape[0]
    return 1.0 if totalWidth == 0 else max(0.0, min(1.0, filledMask.sum() / totalWidth))


def _readBadgeSide(badgeCropBgr: np.ndarray) -> Optional[bool]:
    if badgeCropBgr.size == 0:
        return None

    hsv = cv2.cvtColor(badgeCropBgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Ignore white/gray/dark pixels.
    valid = (sat > 60) & (val > 60)

    # OpenCV hue: red is near 0 OR 179, blue is around 100-130.
    blueMask = valid & (hue >= 100) & (hue <= 130)
    redMask = valid & ((hue <= 10) | (hue >= 170))

    blueCount = int(np.count_nonzero(blueMask))
    redCount = int(np.count_nonzero(redMask))

    total = blueCount + redCount
    if total == 0:
        return None

    if blueCount > redCount:
        return True

    if redCount > blueCount:
        return False

    return None


def _matchByProximity(
    units: list[_RawDetection],
    annotations: list[_RawDetection],
    maxDistScale: float,
    overlapTolerance: float,
) -> dict[int, _RawDetection]:
    """Match each badge/HP-bar annotation to the unit it most likely
    belongs to, using global best-pair-first assignment (build every
    candidate pair, sort by distance, greedily claim) -- the same pattern
    as the tracker's frame-to-frame matching.

    A naive 'for each annotation, take the nearest unit, no matter what
    other annotations decided' loop can cross-wire assignments in crowded
    scenes (Skeleton Army, Minion Horde, etc.): a later, worse-matching
    annotation can silently overwrite the correct assignment an earlier
    one already made for the same unit, or steal the nearest unit out
    from under a badge that actually belonged to it. Sorting all pairs
    globally and claiming best-distance-first avoids that.

    Both the search radius and how far the annotation is allowed to sit
    below the unit's top edge scale with that unit's OWN box size, not a
    frame-wide constant -- unit sprites (and their badges) shrink and grow
    with on-screen perspective.
    """
    candidates: list[tuple[float, int, int]] = []

    for unitIdx, unit in enumerate(units):
        unitX = (unit.boxXyxy[0] + unit.boxXyxy[2]) / 2.0
        unitTopY = unit.boxXyxy[1]
        unitHeight = max(1.0, unit.boxXyxy[3] - unit.boxXyxy[1])
        unitWidth = max(1.0, unit.boxXyxy[2] - unit.boxXyxy[0])
        maxDist = maxDistScale * max(unitHeight, unitWidth)

        for annIdx, ann in enumerate(annotations):
            annX = (ann.boxXyxy[0] + ann.boxXyxy[2]) / 2.0
            annY = (ann.boxXyxy[1] + ann.boxXyxy[3]) / 2.0

            # Allow the annotation to overlap the top of the unit box a
            # bit -- don't require pixel-exact separation, boxes are fuzzy.
            if annY > unitTopY + overlapTolerance * unitHeight:
                continue

            dist = ((unitX - annX) ** 2 + (unitTopY - annY) ** 2) ** 0.5
            if dist > maxDist:
                continue

            candidates.append((dist, unitIdx, annIdx))

    candidates.sort(key=lambda c: c[0])

    matched: dict[int, _RawDetection] = {}
    claimedUnits: set[int] = set()
    claimedAnns: set[int] = set()

    for dist, unitIdx, annIdx in candidates:
        if unitIdx in claimedUnits or annIdx in claimedAnns:
            continue
        claimedUnits.add(unitIdx)
        claimedAnns.add(annIdx)
        matched[unitIdx] = annotations[annIdx]

    return matched


# ---------------------------------------------------------------------------
# side resolution
# ---------------------------------------------------------------------------

class SideResolver:
    """Decides hp% and reads the current frame's badge color per unit.

    hp% comes from the matched HP bar's fill width.

    Side, for THIS frame only, comes from the matched BadgeLevel's color:
    blue -> own, red -> enemy. If no badge matches or its color can't be
    read, `ownUnit` on the returned detection is a meaningless placeholder
    and hasSignal is False -- the caller (UnitTracker) is responsible for
    filling in the real side in that case using the unit's first-seen
    position, since "no badge visible right now" says nothing about
    which side the unit is on.
    """

    def __init__(self, config):
        # How far (in multiples of the unit's own box size) a badge/hp-bar
        # can sit from a unit and still count as a match.
        self.maxMatchDistScale = config.get("maxMatchDistScale", 1.5)
        # How far (in multiples of unit height) an annotation is allowed
        # to overlap down into the unit's box and still count as "above" it.
        self.badgeOverlapTolerance = config.get("badgeOverlapTolerance", 0.4)

    def resolve(
        self,
        image: np.ndarray,
        units: list[_RawDetection],
        hpBars: list[_RawDetection],
        badges: list[_RawDetection],
    ) -> list[tuple[UnitDetection, bool]]:
        height, width = image.shape[:2]

        hpMatch = _matchByProximity(units, hpBars, self.maxMatchDistScale, self.badgeOverlapTolerance)
        badgeMatch = _matchByProximity(units, badges, self.maxMatchDistScale, self.badgeOverlapTolerance)

        results: list[tuple[UnitDetection, bool]] = []
        for i, unit in enumerate(units):
            hpPct = 1.0

            hpAnn = hpMatch.get(i)
            if hpAnn is not None:
                x1, y1, x2, y2 = _clipBox(hpAnn.boxXyxy, width, height)
                crop = image[y1:y2, x1:x2]
                if crop.size:
                    hpPct = _readHpBar(crop)

            side: Optional[bool] = None

            badgeAnn = badgeMatch.get(i)
            if badgeAnn is not None:
                x1, y1, x2, y2 = _clipBox(badgeAnn.boxXyxy, width, height)
                crop = image[y1:y2, x1:x2]
                side = _readBadgeSide(crop)

            hasSignal = side is not None

            detection = UnitDetection(
                className=unit.className,
                confidence=unit.confidence,
                boxXyxy=unit.boxXyxy,
                hpPct=hpPct,
                ownUnit=side if hasSignal else True,  # placeholder; UnitTracker fills the real value when hasSignal is False
            )
            results.append((detection, hasSignal))

        return results


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    trackId: int
    className: str
    box: tuple[float, float, float, float]
    # Fixed once, the frame this track was first created: was it on our
    # side of the board at that moment? This -- not the badge, not the
    # unit's current position -- is what a badge-less frame falls back to.
    firstSeenOwnSide: bool
    framesSinceSeen: int = 0


class UnitTracker:
    """Greedy IoU tracker that decides the final side per unit per frame.

    Rule, applied every frame:
      - A blue badge this frame -> own.
      - A red badge this frame -> enemy. (A badge, when present, is
        authoritative in both directions -- it overrides everything else.)
      - No badge this frame -> whichever side the unit was first seen on
        when its track was created. NOT its current position -- a unit
        that has walked across the river is still whatever side it
        started on unless a badge says otherwise.
    """

    def __init__(self, config):
        self.maxFramesLost = config.get("trackerMaxFramesLost", 10)
        self.iouThreshold = config.get("trackerIouThreshold", 0.25)
        # Fraction of frame height (from the top) that is exclusively
        # enemy territory -- used only to decide a brand-new track's
        # first-seen side when its very first frame has no badge either.
        # Tune/flip to match your capture orientation.
        self.enemyZoneYFrac = config.get("enemyZoneYFrac", 0.2)
        self._tracks: dict[int, _Track] = {}
        self._nextId = 0

    def reset(self) -> None:
        """Call at the start of each new match -- stale tracks/sides from
        the previous game should never bleed into the next one."""
        self._tracks = {}
        self._nextId = 0

    def update(
        self,
        detections: list[UnitDetection],
        hasSignal: list[bool],
        frameHeight: int,
    ) -> list[UnitDetection]:
        results: list[Optional[UnitDetection]] = [None] * len(detections)

        candidates = []
        for trackId, track in self._tracks.items():
            for i, det in enumerate(detections):
                if det.className != track.className:
                    continue
                iou = _iou(track.box, det.boxXyxy)
                if iou >= self.iouThreshold:
                    candidates.append((iou, trackId, i))
        candidates.sort(key=lambda c: c[0], reverse=True)

        claimedTracks: set[int] = set()
        claimedDets: set[int] = set()

        for iou, trackId, i in candidates:
            if trackId in claimedTracks or i in claimedDets:
                continue
            claimedTracks.add(trackId)
            claimedDets.add(i)

            track = self._tracks[trackId]
            det = detections[i]

            side = det.ownUnit if hasSignal[i] else track.firstSeenOwnSide

            track.box = det.boxXyxy
            track.framesSinceSeen = 0

            results[i] = replace(det, ownUnit=side, trackId=trackId)

        for trackId in self._tracks:
            if trackId not in claimedTracks:
                self._tracks[trackId].framesSinceSeen += 1
        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.framesSinceSeen <= self.maxFramesLost
        }

        for i, det in enumerate(detections):
            if i in claimedDets:
                continue
            trackId = self._nextId
            self._nextId += 1

            firstSeenOwnSide = not _isInEnemyZone(det.boxXyxy, frameHeight, self.enemyZoneYFrac)
            side = det.ownUnit if hasSignal[i] else firstSeenOwnSide

            self._tracks[trackId] = _Track(
                trackId=trackId,
                className=det.className,
                box=det.boxXyxy,
                firstSeenOwnSide=firstSeenOwnSide,
                framesSinceSeen=0,
            )
            results[i] = replace(det, ownUnit=side, trackId=trackId)

        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------

class UnitDetector:
    def __init__(self, config):
        self.confThreshold = config.get("confidence", 0.35)
        self.model = YOLO(config.get("weights"))
        self.sideResolver = SideResolver(config)
        self.tracker = UnitTracker(config)

        dummy = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(source=dummy, conf=self.confThreshold, device=0, imgsz=IMG_SIZE, verbose=False)

    def resetTracking(self) -> None:
        """Call at the start of each new match."""
        self.tracker.reset()

    def _rawDetect(self, image: np.ndarray) -> list[_RawDetection]:
        results = self.model.predict(source=image, conf=self.confThreshold, device=0, imgsz=IMG_SIZE, verbose=False)

        out = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                clsId = int(box.cls.item())
                className = names[clsId]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                out.append(_RawDetection(className, conf, (x1, y1, x2, y2)))
        return out

    def process(self, image: np.ndarray) -> list[UnitDetection]:
        detections = self._rawDetect(image)

        units = [d for d in detections if d.className not in (HP_BAR_CLASS, BADGE_CLASS)]
        hpBars = [d for d in detections if d.className == HP_BAR_CLASS]
        badges = [d for d in detections if d.className == BADGE_CLASS]

        resolved = self.sideResolver.resolve(image, units, hpBars, badges)
        if not resolved:
            return []

        resolvedDetections, signalFlags = zip(*resolved)
        return list(resolvedDetections)

    def debugRaw(self, image: np.ndarray, detections: list[UnitDetection]) -> np.ndarray:
        debugImage = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.boxXyxy)
            color = (255, 0, 0) if detection.ownUnit else (0, 0, 255)
            cv2.rectangle(debugImage, (x1, y1), (x2, y2), color, 2)
        return debugImage