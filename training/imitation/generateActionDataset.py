from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from shared.cardRegistry import CardDatabase
from perception.bot.Config import Config
from perception.bot.tensorBuilder import TensorBuilder
from perception.arena.ArenaCalibration import ArenaCalibration, footPoint
from perception.hud.HUDReader import HUDReader
from perception.detection.yoloDetector import UnitDetector


BOARD_WIDTH = 18
BOARD_HEIGHT = 32
HAND_SIZE = 4
PENDING_TIMEOUT = 80

# Last action index is reserved for "do nothing this decision".
NOOP_ACTION = HAND_SIZE * BOARD_WIDTH * BOARD_HEIGHT

DEFAULT_GROUP_MAX_DISTANCE = float(max(BOARD_WIDTH, BOARD_HEIGHT))
DEFAULT_VIDEO_FPS = 30.0
PROGRESS_EVERY_N_FRAMES = 100

DEPLOY_ZONE = (16, 31)


class ConsoleLogger:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self._progressActive = False

    def event(self, msg: str) -> None:
        self._freshLine()
        print(msg)

    def debug(self, msg: str) -> None:
        if self.verbose:
            self._freshLine()
            print(msg)

    def progress(self, msg: str) -> None:
        if not self.verbose:
            return
        print(f"\r{msg}", end="", flush=True)
        self._progressActive = True

    def _freshLine(self) -> None:
        if self._progressActive:
            print()
            self._progressActive = False


def encodeAction(slot, tileX, tileY):
    return slot * BOARD_WIDTH * BOARD_HEIGHT + tileY * BOARD_WIDTH + tileX


def tileOf(detection, calibration):
    px, py = footPoint(detection.boxXyxy)
    return calibration.pixelToTile(px, py)


def isWithinOwnDeployZone(tileY, deployZone):
    return deployZone[0] <= tileY <= deployZone[1]


def loadSpawnGroups(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def loadExistingSamples(path):
    if not os.path.exists(path):
        return []

    print(f"Appending to existing dataset: {path}")
    existing = torch.load(path, weights_only=False)

    if not isinstance(existing, dict):
        raise RuntimeError(f"Invalid actions.pt format: {path}")

    if existing.get("spec_version") != 1:
        raise RuntimeError(f"Unsupported spec_version: {existing.get('spec_version')}")

    if existing.get("board_width") != BOARD_WIDTH:
        raise RuntimeError(f"Board width mismatch: existing={existing.get('board_width')} current={BOARD_WIDTH}")

    if existing.get("board_height") != BOARD_HEIGHT:
        raise RuntimeError(f"Board height mismatch: existing={existing.get('board_height')} current={BOARD_HEIGHT}")

    if existing.get("hand_size") != HAND_SIZE:
        raise RuntimeError(f"Hand size mismatch: existing={existing.get('hand_size')} current={HAND_SIZE}")

    samples = existing.get("samples", [])

    if not isinstance(samples, list):
        raise RuntimeError("Existing 'samples' field is not a list")

    print(f"Existing samples : {len(samples)}")
    return samples


def spawnGroupFor(cardName, spawnGroups):
    return spawnGroups.get(cardName, {
        "units": {cardName: 1},
        "maxDistanceX": DEFAULT_GROUP_MAX_DISTANCE,
        "maxDistanceY": DEFAULT_GROUP_MAX_DISTANCE,
    })


def matchSpawnGroup(groupDef, detections, calibration, claimed, deployZone):
    unitsNeeded = groupDef["units"]
    maxDistanceX = groupDef.get("maxDistanceX", DEFAULT_GROUP_MAX_DISTANCE)
    maxDistanceY = groupDef.get("maxDistanceY", DEFAULT_GROUP_MAX_DISTANCE)

    pools = {}

    for unitName in unitsNeeded:
        pools[unitName] = [
            d for d in detections
            if d.ownUnit
            and d.className == unitName
            and id(d) not in claimed
            and isWithinOwnDeployZone(tileOf(d, calibration)[1], deployZone)
        ]

    for name, count in unitsNeeded.items():
        if len(pools[name]) < count:
            return None

    anchorName = next(iter(unitsNeeded))
    bestCluster = None
    bestSpread = None

    for anchor in pools[anchorName]:
        anchorTile = tileOf(anchor, calibration)
        cluster = []
        ok = True

        for name, count in unitsNeeded.items():
            scored = []

            for d in pools[name]:
                tx, ty = tileOf(d, calibration)
                dx = abs(tx - anchorTile[0])
                dy = abs(ty - anchorTile[1])

                if dx <= maxDistanceX and dy <= maxDistanceY:
                    scored.append((dx + dy, d))

            if len(scored) < count:
                ok = False
                break

            scored.sort(key=lambda x: x[0])
            cluster.extend(d for _, d in scored[:count])

        if not ok:
            continue

        tiles = [tileOf(d, calibration) for d in cluster]
        xs, ys = zip(*tiles)

        spread = max(xs) - min(xs) + max(ys) - min(ys)

        if bestSpread is None or spread < bestSpread:
            bestSpread = spread
            bestCluster = cluster

    return bestCluster


def groupCenterTile(group, calibration):
    points = [footPoint(d.boxXyxy) for d in group]
    px = sum(p[0] for p in points) / len(points)
    py = sum(p[1] for p in points) / len(points)
    return calibration.pixelToTile(px, py)


def resolveSpawnGroup(pendingPlay, frame, detections, calibration, claimed, deployZone):
    if pendingPlay["resolvedGroup"] is not None:
        return

    matched = matchSpawnGroup(
        pendingPlay["groupDef"],
        detections,
        calibration,
        claimed,
        deployZone,
    )

    if not matched:
        return

    tileX, tileY = groupCenterTile(matched, calibration)

    if not isWithinOwnDeployZone(tileY, deployZone):
        return

    claimed.update(id(d) for d in matched)

    pendingPlay["resolvedGroup"] = matched
    pendingPlay["resolvedTile"] = (tileX, tileY)
    pendingPlay["resolvedFrame"] = frame.copy()
    pendingPlay["resolvedDetections"] = list(detections)


def drawDetections(image, detections, calibration, highlight=None):
    image = image.copy()
    highlight = highlight or []

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.boxXyxy)

        color = (
            (0, 255, 255)
            if detection in highlight
            else ((0, 255, 0) if detection.ownUnit else (0, 0, 255))
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"{detection.className} {detection.confidence:.2f} {'OWN' if detection.ownUnit else 'ENEMY'}"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        px, py = footPoint(detection.boxXyxy)
        tileX, tileY = calibration.pixelToTile(px, py)

        cv2.circle(image, (int(px), int(py)), 6, color, -1)

        cv2.putText(
            image,
            f"tile=({tileX},{tileY})",
            (int(px) + 8, int(py)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


def drawTile(image, calibration, tileX, tileY):
    corners = [
        calibration.tileToPixel(tileX, tileY, center=False),
        calibration.tileToPixel(tileX + 1, tileY, center=False),
        calibration.tileToPixel(tileX + 1, tileY + 1, center=False),
        calibration.tileToPixel(tileX, tileY + 1, center=False),
    ]

    poly = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()

    cv2.fillPoly(overlay, [poly], (0, 255, 255))
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    cv2.polylines(image, [poly], True, (0, 255, 255), 3)

    center = calibration.tileToPixel(tileX, tileY, center=True)

    cv2.putText(
        image,
        f"TARGET ({tileX},{tileY})",
        (int(center[0]) - 70, int(center[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def saveDebugImage(path, image, detections, calibration, title, highlight=None, tile=None):
    image = drawDetections(image, detections, calibration, highlight)

    if tile is not None:
        drawTile(image, calibration, tile[0], tile[1])

    cv2.putText(
        image,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(path, image)


def defaultAnnotatedVideoPath(outputPath: str) -> str:
    root, _ = os.path.splitext(outputPath)
    return f"{root}_annotated.mp4"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--spawn-groups", default="data/cards/CardSpawns.json")
    parser.add_argument("--annotated-video", default=None)
    parser.add_argument("--skip-annotated-video", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--noop-every",
        type=int,
        default=30,
        help="Sample a noop (do-nothing) example every N frames while idle. "
             "Default ~1/sec at 30fps. Set to 0 to disable noop sampling.",
    )
    parser.add_argument(
        "--noop-cooldown",
        type=int,
        default=45,
        help="Frames to skip noop sampling for after a card leaves the hand, "
             "so frames adjacent to a real play aren't mislabeled as 'wait'.",
    )

    args = parser.parse_args()

    log = ConsoleLogger(args.verbose)

    cards = CardDatabase("data/cards/Cards.json")
    config = Config("perception/bot/Config.yml")
    spawnGroups = loadSpawnGroups(args.spawn_groups)
    calibration = ArenaCalibration(config.section("calibration"))
    tensorBuilder = TensorBuilder(cards)
    detector = UnitDetector(config.section("yolo"))
    hudReader = HUDReader(calibration, config.section("hudReader"), cards)

    video = cv2.VideoCapture(args.video)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS

    print(f"Video  : {args.video}")
    print(f"Frames : {totalFrames}")
    print(f"FPS    : {fps:.2f}")
    print(f"Deploy : Y={DEPLOY_ZONE[0]}..{DEPLOY_ZONE[1]}")
    print(f"Noop   : every={args.noop_every} cooldown={args.noop_cooldown} action_id={NOOP_ACTION}")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    annotatedVideoPath = None
    annotatedWriter = None

    if not args.skip_annotated_video:
        annotatedVideoPath = args.annotated_video or defaultAnnotatedVideoPath(args.output)

        outDir = os.path.dirname(annotatedVideoPath)
        if outDir:
            os.makedirs(outDir, exist_ok=True)

        print(f"Video output : {annotatedVideoPath}")

    previousHand = None
    previousDetections = []
    previousObservation = None
    previousFrame = None
    previousElixir = None

    pendingPlays = []
    debugIndex = 0

    # Frame index of the most recent card-leaves-hand event, used to gate
    # noop sampling so we don't label frames right around a real play as
    # "the correct move was to wait".
    lastPlayFrame = -10 ** 9
    noopSampleCount = 0

    # Load existing samples only when --append is used.
    samples = loadExistingSamples(args.output) if args.append else []
    originalSampleCount = len(samples)

    if args.append:
        print(f"Append mode enabled")
    else:
        print("Starting new dataset")

    frameIndex = 0

    while True:
        if args.max_frames and frameIndex >= args.max_frames:
            break

        ok, frame = video.read()

        if not ok:
            break

        if frameIndex == 0:
            log.debug(f"Arena match: {calibration.match(frame)}")

            if annotatedVideoPath is not None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotatedWriter = cv2.VideoWriter(
                    annotatedVideoPath,
                    fourcc,
                    fps,
                    (width, height),
                )

        detections = detector.process(frame)
        hudState = hudReader.readFrame(frame)
        observation = tensorBuilder.build(detections, calibration, hudState)

        if annotatedWriter is not None:
            annotatedWriter.write(drawDetections(frame, detections, calibration))

        hand = list(hudState.hand.handCards)
        currentElixir = hudState.elixir

        # --- Noop sampling ------------------------------------------------
        # Every video frame is otherwise only ever a source of "play a card"
        # labels (from the pending-play resolution below). Without explicit
        # "do nothing" examples the model never learns to wait and will
        # always emit some play action. Sample idle frames at a fixed
        # cadence (default ~1/sec) so the wait/act ratio in the dataset
        # roughly matches real decision frequency, and skip a cooldown
        # window after any real play so we don't mislabel the transition.
        if args.noop_every > 0:
            sinceLastPlay = frameIndex - lastPlayFrame

            if sinceLastPlay > args.noop_cooldown and frameIndex % args.noop_every == 0:
                samples.append({
                    "frame": frameIndex,
                    "action": NOOP_ACTION,
                    "global_features": observation.globalFeatures.cpu(),
                    "board": observation.board.cpu(),
                    "entities": observation.entities.cpu(),
                    "entity_mask": observation.entityMask.cpu(),
                })

                noopSampleCount += 1

                log.debug(
                    f"[NOOP] frame={frameIndex} "
                    f"since_last_play={sinceLastPlay} "
                    f"elixir={currentElixir}"
                )
        # -------------------------------------------------------------------

        stillPending = []
        claimed = set()

        for pendingPlay in pendingPlays:
            pendingPlay["age"] += 1

            if not pendingPlay["confirmed"] and pendingPlay["cardId"] in hand:
                log.debug(
                    f"[CANCELLED] frame={pendingPlay['frame']} "
                    f"card={pendingPlay['cardName']} slot={pendingPlay['slot']}"
                )
                continue

            resolveSpawnGroup(
                pendingPlay,
                frame,
                detections,
                calibration,
                claimed,
                DEPLOY_ZONE,
            )

            cumulativeDrop = pendingPlay["elixirBaseline"] - currentElixir

            hasSpawnGroup = pendingPlay["resolvedGroup"] is not None

            if not pendingPlay["confirmed"] and hasSpawnGroup:
                if cumulativeDrop >= 1.0:
                    pendingPlay["confirmed"] = True

                    log.debug(
                        f"[ELIXIR CONFIRMED] frame={pendingPlay['frame']} "
                        f"card={pendingPlay['cardName']} "
                        f"drop={cumulativeDrop:.2f} "
                        f"age={pendingPlay['age']}"
                    )

                elif args.verbose and pendingPlay["age"] % 10 == 0:
                    log.debug(
                        f"[ELIXIR WAIT] frame={pendingPlay['frame']} "
                        f"card={pendingPlay['cardName']} "
                        f"drop={cumulativeDrop:.2f} "
                        f"age={pendingPlay['age']} "
                        f"spawn_found=True"
                    )

            elif not pendingPlay["confirmed"] and args.verbose and pendingPlay["age"] % 10 == 0:
                log.debug(
                    f"[PLAY WAIT] frame={pendingPlay['frame']} "
                    f"card={pendingPlay['cardName']} "
                    f"drop={cumulativeDrop:.2f} "
                    f"age={pendingPlay['age']} "
                    f"spawn_found=False"
                )

            if pendingPlay["confirmed"] and pendingPlay["resolvedGroup"] is not None:
                group = pendingPlay["resolvedGroup"]
                tileX, tileY = pendingPlay["resolvedTile"]

                if 0 <= tileX < BOARD_WIDTH and isWithinOwnDeployZone(tileY, DEPLOY_ZONE):
                    action = encodeAction(pendingPlay["slot"], tileX, tileY)

                    samples.append({
                        "frame": pendingPlay["frame"],
                        "action": action,
                        "global_features": pendingPlay["observation"].globalFeatures.cpu(),
                        "board": pendingPlay["observation"].board.cpu(),
                        "entities": pendingPlay["observation"].entities.cpu(),
                        "entity_mask": pendingPlay["observation"].entityMask.cpu(),
                    })

                    if args.debug_dir:
                        debugIndex += 1
                        debugDir = os.path.join(args.debug_dir, f"play_{debugIndex:04d}")
                        os.makedirs(debugDir, exist_ok=True)

                        saveDebugImage(
                            os.path.join(debugDir, "start.jpg"),
                            pendingPlay["frameImage"],
                            pendingPlay["frameDetections"],
                            calibration,
                            f"START | frame={pendingPlay['frame']} | {pendingPlay['cardName']} | slot={pendingPlay['slot']}",
                        )

                        saveDebugImage(
                            os.path.join(debugDir, "end.jpg"),
                            pendingPlay["resolvedFrame"],
                            pendingPlay["resolvedDetections"],
                            calibration,
                            f"END | tile=({tileX},{tileY}) | action={action}",
                            highlight=group,
                            tile=(tileX, tileY),
                        )

                    log.event(
                        f"[PLAY] frame={pendingPlay['frame']} "
                        f"card={pendingPlay['cardName']} "
                        f"slot={pendingPlay['slot']} "
                        f"units={len(group)} "
                        f"tile=({tileX},{tileY}) "
                        f"action={action}"
                    )

                    continue

            if pendingPlay["age"] >= PENDING_TIMEOUT:
                finalDrop = pendingPlay["elixirBaseline"] - currentElixir

                log.event(
                    f"[UNKNOWN] frame={pendingPlay['frame']} "
                    f"card={pendingPlay['cardName']} "
                    f"confirmed={pendingPlay['confirmed']} "
                    f"has_spawn_group={pendingPlay['resolvedGroup'] is not None} "
                    f"cost={pendingPlay['cost']} "
                    f"final_cumulative_drop={finalDrop:.2f}"
                )

                continue

            stillPending.append(pendingPlay)

        pendingPlays = stillPending

        if previousHand is not None:
            pendingSlots = {p["slot"] for p in pendingPlays}

            for slot in range(HAND_SIZE):
                oldCard = previousHand[slot]
                newCard = hand[slot]

                if oldCard != 0 and newCard == 0 and slot not in pendingSlots:
                    cardName = cards.name(oldCard)
                    cost = cards.elixir(oldCard)

                    pendingPlays.append({
                        "frame": frameIndex - 1,
                        "slot": slot,
                        "cardId": oldCard,
                        "cardName": cardName,
                        "cost": cost,
                        "elixirBaseline": previousElixir,
                        "observation": previousObservation,
                        "frameImage": previousFrame.copy(),
                        "frameDetections": previousDetections.copy(),
                        "age": 0,
                        "confirmed": False,
                        "groupDef": spawnGroupFor(cardName, spawnGroups),
                        "resolvedGroup": None,
                        "resolvedTile": None,
                        "resolvedFrame": None,
                        "resolvedDetections": None,
                    })

                    pendingSlots.add(slot)
                    lastPlayFrame = frameIndex - 1

                    log.debug(
                        f"[PENDING] frame={frameIndex - 1} "
                        f"card={cardName} slot={slot} cost={cost} "
                        f"elixir_baseline={previousElixir}"
                    )

        previousHand = hand
        previousDetections = detections
        previousObservation = observation
        previousFrame = frame.copy()
        previousElixir = currentElixir

        frameIndex += 1

        if frameIndex % PROGRESS_EVERY_N_FRAMES == 0:
            log.progress(f"Processed {frameIndex}/{totalFrames} samples={len(samples)}")

    video.release()

    if annotatedWriter is not None:
        annotatedWriter.release()

    output = {
        "spec_version": 1,
        "board_width": BOARD_WIDTH,
        "board_height": BOARD_HEIGHT,
        "hand_size": HAND_SIZE,
        "num_actions": HAND_SIZE * BOARD_WIDTH * BOARD_HEIGHT + 1,
        "samples": samples,
    }

    torch.save(output, args.output)

    newSamples = len(samples) - originalSampleCount

    log._freshLine()

    print()
    print("Finished")
    print(f"Frames processed : {frameIndex}")
    print(f"Previous samples : {originalSampleCount}")
    print(f"New samples      : {newSamples}")
    print(f"  of which noop  : {noopSampleCount}")
    print(f"Total samples    : {len(samples)}")
    print(f"Saved to         : {args.output}")

    if args.debug_dir:
        print(f"Debug images     : {args.debug_dir}")

    if annotatedVideoPath is not None:
        print(f"Annotated video  : {annotatedVideoPath}")


if __name__ == "__main__":
    main()
