from __future__ import annotations

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import argparse
import time
import cv2
import numpy as np
import torch

from shared.cardRegistry import CardDatabase
from Configs.Config import Config
from perception.bot.tensorBuilder import TensorBuilder
from Model.policy import ClashRLModel
from perception.detection.yoloDetector import UnitDetector
from perception.arena.ArenaCalibration import ArenaCalibration, footPoint
from perception.hud.HUDReader import HUDReader


def fitToScreen(image, maxWidth=800, maxHeight=900):
    height, width = image.shape[:2]
    scale = min(maxWidth / width, maxHeight / height, 1.0)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image


def drawTileMarker(image, calibration, tileX, tileY, color=(0, 255, 255), alpha=0.35):
    corners = [
        calibration.tileToPixel(tileX, tileY, center=False),
        calibration.tileToPixel(tileX + 1, tileY, center=False),
        calibration.tileToPixel(tileX + 1, tileY + 1, center=False),
        calibration.tileToPixel(tileX, tileY + 1, center=False),
    ]
    poly = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [poly], True, color, 2)


def drawDetection(image, detection, calibration):
    px, py = footPoint(detection.box_xyxy)
    tileX, tileY = calibration.pixelToTile(px, py)
    x1, y1, x2, y2 = map(int, detection.box_xyxy)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(image, (int(px), int(py)), 7, (0, 255, 0), -1)
    drawTileMarker(image, calibration, tileX, tileY)

    label = f"{detection.class_name} {detection.hp_pct * 100:.0f}% {'OWN' if detection.ownUnit else 'ENEMY'} T:{tileX},{tileY}"
    (textW, textH), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    textX, textY = x1, max(textH + 5, y1 - 5)

    cv2.rectangle(image, (textX, textY - textH - baseline), (textX + textW, textY + baseline), (0, 0, 0), -1)
    cv2.putText(image, label, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return px, py, tileX, tileY


def drawHudBox(image, box_xywh, label, color=(0, 200, 255)):
    x, y, w, h = box_xywh
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def drawHudState(image, hudReader, hudState, w, h):
    drawHudBox(image, hudReader.calib.scaleHudBox(hudReader.calib.timeBox, w, h), f"time={formatTime(hudState.time)}")
    drawHudBox(image, hudReader.calib.scaleHudBox(hudReader.calib.elixirBox, w, h), f"elixir={hudState.elixir}")

    towers = {
        "allyLeft": hudState.own.leftTowerHp, "allyRight": hudState.own.rightTowerHp,
        "allyKing": hudState.own.kingTowerHp, "enemyLeft": hudState.enemy.leftTowerHp,
        "enemyRight": hudState.enemy.rightTowerHp, "enemyKing": hudState.enemy.kingTowerHp,
    }

    for name, hp in towers.items():
        boxPx = hudReader._towerPosCache.get(name)
        if boxPx is None:
            continue
        drawHudBox(image, boxPx, f"{name}={hp}" if hp != -1 else f"{name}=DEAD", (0, 0, 255) if hp == -1 else (0, 200, 255))

    gameText = "GAME ACTIVE" if hudState.gameAlive else ("YOU WON" if hudState.won else "YOU LOST")
    gameColor = (0, 200, 0) if hudState.gameAlive or hudState.won else (0, 0, 255)
    cv2.putText(image, gameText, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gameColor, 2, cv2.LINE_AA)


def drawHand(image, cards, hudReader, hand, w, h):
    for slot, cardId in zip(("slot1", "slot2", "slot3", "slot4"), hand.handCards):
        box = hudReader.calib.cardBoxes.get(slot)
        if box is None:
            continue
        drawHudBox(image, hudReader.calib.scaleHudBox(box, w, h), f"{slot}={cards.name(cardId) or '?'}", (255, 180, 0))

    if hudReader.calib.nextCardBox is not None:
        boxPx = hudReader.calib.scaleHudBox(hudReader.calib.nextCardBox, w, h)
        drawHudBox(image, boxPx, f"next={cards.name(hand.nextCard) or '?'}", (255, 180, 0))


def formatTime(seconds):
    return "--:--" if seconds is None else f"{seconds // 60}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    cards = CardDatabase("data/cards/Cards.json")
    config = Config("perception/bot/Config.yml")
    calibration = ArenaCalibration(config.section("calibration"))
    arena = calibration.match(image)
    print(f"Arena: {arena}")

    tensorBuilder = TensorBuilder(cards)
    detector = UnitDetector(config.section("yolo"))
    hudReader = HUDReader(calibration, config.section("hudReader"), cards)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClashRLModel().to(device).eval()

    start = time.perf_counter()
    detections = detector.process(image)
    detectionTime = (time.perf_counter() - start) * 1000
    print(f"Time: {detectionTime:.2f} ms")

    start = time.perf_counter()
    hudState = hudReader.readFrame(image)
    hudTime = (time.perf_counter() - start) * 1000
    print(f"HUD time: {hudTime:.2f} ms")

    start = time.perf_counter()
    observation = tensorBuilder.build(detections, calibration, hudState)
    tensorTime = (time.perf_counter() - start) * 1000
    print(f"Tensor time: {tensorTime:.2f} ms")

    globalFeatures = observation.global_features.unsqueeze(0).float().to(device)
    board = observation.board.unsqueeze(0).float().to(device)
    entities = observation.entities.unsqueeze(0).float().to(device)
    entityMask = observation.entity_mask.unsqueeze(0).bool().to(device)

    with torch.no_grad():
        for _ in range(5):
            model(globalFeatures, board, entities, entityMask)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        policyLogits, value = model(globalFeatures, board, entities, entityMask)

    torch.cuda.synchronize()
    rlTime = (time.perf_counter() - start) * 1000

    action = policyLogits.argmax(dim=-1).item()
    value = value.item()

    print(f"RL time: {rlTime:.2f} ms")
    print(f"\n[main] RL:\n  action : {action}\n  value  : {value:.4f}")

    print("\n[main] HUD state:")
    print(f"  time remaining : {formatTime(hudState.time)}")
    print(f"  elixir         : {hudState.elixir}")
    print(f"  game alive     : {hudState.gameAlive}")
    print(f"  result         : {'IN PROGRESS' if hudState.gameAlive else ('WON' if hudState.won else 'LOST')}")

    print("\n  Own:")
    print(f"    left tower   : {hudState.own.leftTowerHp}")
    print(f"    right tower  : {hudState.own.rightTowerHp}")
    print(f"    king tower   : {hudState.own.kingTowerHp}")

    print("\n  Enemy:")
    print(f"    left tower   : {hudState.enemy.leftTowerHp}")
    print(f"    right tower  : {hudState.enemy.rightTowerHp}")
    print(f"    king tower   : {hudState.enemy.kingTowerHp}")

    print("\n[main] Hand:")
    for slot, cardId in zip(("slot1", "slot2", "slot3", "slot4"), hudState.hand.handCards):
        print(f"  {slot:6s}: id={cardId:<3d} name={cards.name(cardId) or 'unmatched'}")
    print(f"  {'next':6s}: id={hudState.hand.nextCard:<3d} name={cards.name(hudState.hand.nextCard) or 'unmatched'}")

    print(f"\n[main] {len(detections)} actual detections:")
    for detection in detections:
        px, py, tileX, tileY = drawDetection(image, detection, calibration)
        print(f"  {detection.class_name:20s} conf={detection.confidence:.2f} hp={detection.hp_pct * 100:5.1f}% side={'OWN' if detection.ownUnit else 'ENEMY'} box={detection.box_xyxy}")
        print(f"    foot=({px:.1f}, {py:.1f}) -> tile=({tileX}, {tileY})")

    print("\n[main] Observation:")
    print(f"  board          : {observation.board.shape}")
    print(f"  entities       : {observation.entities.shape}")
    print(f"  entity mask    : {observation.entity_mask.sum()} active")
    print(f"  global features: {observation.global_features.shape}")
    print(f"  vocab size     : {observation.vocab_size}")

    for warning in observation.warnings:
        print(f"  WARNING: {warning}")

    print("\n[main] Timing:")
    print(f"  detection : {detectionTime:.2f} ms")
    print(f"  HUD       : {hudTime:.2f} ms")
    print(f"  tensor    : {tensorTime:.2f} ms")
    print(f"  RL        : {rlTime:.2f} ms")
    print(f"  total     : {detectionTime + hudTime + tensorTime + rlTime:.2f} ms")

    if not args.display:
        return

    h, w = image.shape[:2]
    drawHudState(image, hudReader, hudState, w, h)
    drawHand(image, cards, hudReader, hudState.hand, w, h)

    cv2.imshow("Processed", fitToScreen(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
