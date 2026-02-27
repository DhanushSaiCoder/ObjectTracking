import os
os.environ["MPLBACKEND"] = "Agg"

import time
import cv2
import numpy as np
from pathlib import Path

from ostrack_tracker import OSTrackTracker, BBox


def select_init_bbox(window_name: str, frame: np.ndarray):
    """
    Use OpenCV ROI selector.
    Returns BBox in xyxy or None if cancelled.
    """
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    return BBox(float(x), float(y), float(x + w), float(y + h))


def main():
    repo_root = Path(__file__).resolve().parents[1]

    tracker = OSTrackTracker(
        repo_root=str(repo_root),
        tracker_name="ostrack",
        param_name="vitb_384_mae_ce_32x4_ep300",
        dataset_name="demo",
        verbose=False,
    )

    cap = cv2.VideoCapture(0)   # change to video path if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    window_name = "OSTrack Only Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nInstructions:")
    print("  1) Press 's' to select target ROI")
    print("  2) Drag a box around the target and press ENTER")
    print("  3) Press 'r' to reset/reselect")
    print("  4) Press 'q' to quit\n")

    tracking = False
    fps_smooth = 0.0
    prev_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR: Failed to read frame")
            break

        draw = frame.copy()

        # FPS
        now_t = time.perf_counter()
        dt = max(now_t - prev_t, 1e-6)
        prev_t = now_t
        fps = 1.0 / dt
        fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)

        if tracking:
            result = tracker.update(frame)
            if result.ok and result.bbox is not None:
                b = result.bbox
                x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])

                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    draw,
                    "TRACKING",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                tracking = False
                tracker.reset()
                cv2.putText(
                    draw,
                    "LOST - Press 's' to select again",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                draw,
                "Press 's' to select target ROI",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            draw,
            f"FPS: {fps_smooth:.1f}",
            (10, draw.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, draw)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            tracking = False
            tracker.reset()
            print("Tracker reset")

        elif key == ord("s"):
            paused = frame.copy()
            init_bbox = select_init_bbox(window_name, paused)
            if init_bbox is None:
                print("ROI selection cancelled")
                continue

            ok = tracker.initialize(frame, init_bbox)
            print(f"Tracker initialize: {ok}")
            tracking = bool(ok)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()