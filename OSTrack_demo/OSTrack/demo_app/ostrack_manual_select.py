import os
os.environ["MPLBACKEND"] = "Agg"

import time
import cv2
from pathlib import Path

from ostrack_tracker import OSTrackTracker, BBox


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


class AppState:
    def __init__(self):
        self.mode = "SELECT"   # SELECT / TRACK
        self.paused = False
        self.last_frame = None

    def reset_to_select(self):
        self.mode = "SELECT"
        self.paused = False
        self.last_frame = None


def draw_track(frame, bbox, label="TRACK MODE"):
    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10 if y1 > 14 else y1 + 22),
        FONT,
        0.7,
        GREEN,
        2,
        cv2.LINE_AA,
    )


def select_roi_on_frame(window_name: str, frame):
    """
    Pause and let user drag ROI.
    Returns BBox or None.
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
        verbose=True,
    )

    cap = cv2.VideoCapture("cars.webm")
    
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1e-3:
        src_fps = 25.0  # fallback if video metadata is bad

    frame_delay_ms = max(1, int(1000 / src_fps))
    print(f"Source FPS: {src_fps:.2f}, frame delay: {frame_delay_ms} ms")

    if not cap.isOpened():
        print("ERROR: Could not open source")
        return

    state = AppState()

    window_name = "OSTrack Manual ROI Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nInstructions:")
    print("  - Press 'p' to pause and select ROI")
    print("  - Drag ROI and press ENTER / SPACE")
    print("  - Press 'c' inside ROI tool to cancel")
    print("  - Press 'r' to reset to SELECT mode")
    print("  - Press 'q' to quit\n")

    prev_time = time.perf_counter()
    fps_smooth = 0.0

    while True:
        # In paused mode, keep showing same frame
        if not state.paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of stream / failed to read frame")
                break
            state.last_frame = frame
        else:
            frame = state.last_frame.copy()

        frame_h = frame.shape[0]

        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        inst_fps = 1.0 / dt
        fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

        if state.mode == "SELECT":
            cv2.putText(
                frame,
                "SELECT MODE: Press 'p' to pause and drag ROI",
                (10, 25),
                FONT,
                0.7,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            if state.paused:
                cv2.putText(
                    frame,
                    "PAUSED: Draw ROI or press 'r' / 'q'",
                    (10, 55),
                    FONT,
                    0.7,
                    YELLOW,
                    2,
                    cv2.LINE_AA,
                )

        else:   # TRACK mode
            result = tracker.update(frame)

            if result.ok and result.bbox is not None:
                draw_track(frame, result.bbox, "TRACK MODE")
            else:
                tracker.reset()
                state.reset_to_select()
                cv2.putText(
                    frame,
                    "LOST - Back to SELECT mode",
                    (10, 30),
                    FONT,
                    0.7,
                    RED,
                    2,
                    cv2.LINE_AA,
                )

        # common overlays
        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f}",
            (10, frame_h - 10),
            FONT,
            0.7,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(10) & 0xFF
        
        # key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"):
            tracker.reset()
            state.reset_to_select()
            print("Reset to SELECT mode")

        elif key == ord("p") and state.mode == "SELECT" and state.last_frame is not None:
            state.paused = True
            frozen = state.last_frame.copy()

            roi_bbox = select_roi_on_frame(window_name, frozen)

            if roi_bbox is None:
                print("ROI selection cancelled")
                state.paused = False
                continue

            ok = tracker.initialize(frozen, roi_bbox)
            print(f"Tracker initialize: {ok}")
            print(f"Selected bbox: {roi_bbox}")

            if ok:
                state.mode = "TRACK"
                state.paused = False
            else:
                print("OSTrack initialization failed")
                state.paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()