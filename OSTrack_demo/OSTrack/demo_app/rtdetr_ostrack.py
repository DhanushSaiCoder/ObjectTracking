import os
os.environ["MPLBACKEND"] = "Agg"

import time
import cv2
from pathlib import Path

from rtdetr_detector import RTDETRDetector
from ostrack_tracker import OSTrackTracker, BBox as TrkBBox


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class AppState:
    def __init__(self):
        self.mode = "SELECT"
        self.last_detections = []
        self.last_select_frame = None   # exact frame shown in SELECT mode
        self.pending_click = None

    def reset_to_select(self):
        self.mode = "SELECT"
        self.pending_click = None
        self.last_detections = []
        self.last_select_frame = None


def pick_detection_by_click(detections, x: float, y: float):
    """
    Single-pass version:
    pick the smallest-area bbox containing the click,
    tie-break with higher confidence.
    """
    best_det = None
    best_area = float("inf")
    best_score = -1.0

    for det in detections:
        b = det.bbox
        if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
            area = b.area
            score = float(getattr(det, "score", 0.0))

            if area < best_area or (area == best_area and score > best_score):
                best_det = det
                best_area = area
                best_score = score

    return best_det


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.pending_click = (x, y)


def draw_detection(frame, det):
    b = det.bbox
    x1 = int(b.x1)
    y1 = int(b.y1)
    x2 = int(b.x2)
    y2 = int(b.y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
    cv2.putText(
        frame,
        f"{det.class_name} {det.score:.2f}",
        (x1, y1 - 8 if y1 > 12 else y1 + 18),
        FONT,
        0.5,
        YELLOW,
        1,
        cv2.LINE_AA,
    )


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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "rtdetr-l.pt"

    detector = RTDETRDetector(
        model_path=str(model_path),
        device="cuda:0",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        classes=(4,),     
        max_det=50,
        half=True,
        verbose=False,
    )
    
    # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}   

    tracker = OSTrackTracker(
        repo_root=str(repo_root),
        tracker_name="ostrack",
        param_name="vitb_384_mae_ce_32x4_ep300",
        dataset_name="demo",
        verbose=True,
    )

    cap = cv2.VideoCapture(
        "plane2.webm"
    )

    if not cap.isOpened():
        print("ERROR: Could not open source")
        return

    state = AppState()

    window_name = "RTDETR + OSTrack Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, state)

    print("\nInstructions:")
    print("  - SELECT mode: click a detected person bbox")
    print("  - TRACK mode : OSTrack tracks selected target")
    print("  - Press 'r' to reset back to SELECT mode")
    print("  - Press 'q' to quit\n")

    prev_time = time.perf_counter()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("End of stream / failed to read frame")
            break

        # no frame.copy() here — draw directly on frame after inference
        frame_h = frame.shape[0]

        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        inst_fps = 1.0 / dt
        fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

        if state.mode == "SELECT":
            detections = detector.detect(frame)
            state.last_detections = detections
            state.last_select_frame = frame.copy()   # freeze the exact frame user sees

            # draw detections on current frame
            for det in detections:
                draw_detection(frame, det)

            cv2.putText(
                frame,
                "SELECT MODE: Click a person bbox",
                (10, 25),
                FONT,
                0.7,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            # IMPORTANT: use the frozen displayed frame + detections
            if state.pending_click is not None:
                click_x, click_y = state.pending_click
                state.pending_click = None

                if state.last_select_frame is None or not state.last_detections:
                    print("No frozen frame/detections available for click processing")
                else:
                    selected = pick_detection_by_click(state.last_detections, click_x, click_y)

                    if selected is not None:
                        b = selected.bbox
                        init_bbox = TrkBBox(b.x1, b.y1, b.x2, b.y2)

                        ok = tracker.initialize(state.last_select_frame, init_bbox)
                        print(f"Tracker initialize: {ok}")
                        print(f"Selected bbox: {b}")

                        if ok:
                            state.mode = "TRACK"
                        else:
                            print("OSTrack initialization failed")
                    else:
                        print("Click was not inside any detected bbox")

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
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()
            state.reset_to_select()
            print("Reset to SELECT mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


