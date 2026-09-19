"""
Usage:
    python calibrateCorners.py --image frame.png

Click 8 points, IN THIS ORDER:
    1. board top-left       (wide part of the floor, left edge, top row)
    2. board top-right      (wide part of the floor, right edge, top row)
    3. king top-left        (top-left tile behind the king, i.e. where the floor
                              actually starts vertically at the top)
    4. king top-right       (top-right tile behind the king)
    5. board bottom-left    (wide part of the floor, left edge, bottom row)
    6. board bottom-right   (wide part of the floor, right edge, bottom row)
    7. king bottom-left     (bottom-left tile behind the king)
    8. king bottom-right    (bottom-right tile behind the king)

The tool combines the X from the "board" points with the Y from the "king"
points to build the 4 corners actually used for calibration:
    topLeft     = (board_top_left.x,     king_top_left.y)
    topRight    = (board_top_right.x,    king_top_right.y)
    bottomLeft  = (board_bottom_left.x,  king_bottom_left.y)
    bottomRight = (board_bottom_right.x, king_bottom_right.y)

Press 'r' to reset and re-pick, 'q' or Enter once all 8 are placed.
"""
from __future__ import annotations
import argparse
import cv2

LABELS = [
    "board-top-left", "board-top-right",
    "king-top-left", "king-top-right",
    "board-bottom-left", "board-bottom-right",
    "king-bottom-left", "king-bottom-right",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    points: list[tuple[int, int]] = []

    def _redraw() -> None:
        display = image.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, LABELS[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Click 8 corners", display)

    def _on_click(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
            points.append((x, y))
            print(f"{LABELS[len(points) - 1]}: ({x}, {y})")
            _redraw()

    cv2.namedWindow("Click 8 corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click 8 corners", _on_click)
    _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
            print("--- reset ---")
            _redraw()
        elif key in (ord("q"), 13) and len(points) == 8:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    boardTL, boardTR, kingTL, kingTR, boardBL, boardBR, kingBL, kingBR = points

    topLeft = (boardTL[0], kingTL[1])
    topRight = (boardTR[0], kingTR[1])
    bottomLeft = (boardBL[0], kingBL[1])
    bottomRight = (boardBR[0], kingBR[1])

    print("\nPaste into ArenaMapping.yml:\n")
    print("    calibration:")
    print(f"      topLeft: [{topLeft[0]}, {topLeft[1]}]")
    print(f"      topRight: [{topRight[0]}, {topRight[1]}]")
    print(f"      bottomLeft: [{bottomLeft[0]}, {bottomLeft[1]}]")
    print(f"      bottomRight: [{bottomRight[0]}, {bottomRight[1]}]")


if __name__ == "__main__":
    main()