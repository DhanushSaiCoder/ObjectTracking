import os
os.environ["MPLBACKEND"] = "Agg"

import time
import cv2
from pathlib import Path

from ostrack_tracker import OSTrackTracker, BBox, TrackResult


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
PALETTE = [
    (0, 255, 0),
    (255, 200, 0),
    (255, 0, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 255, 255),
    (200, 80, 255),
    (80, 255, 160),
]

TRACK_UPDATE_INTERVAL = 2  # Update each tracker every N frames to reduce load.
DRAW_SEARCH_HINTS = False  # Disable to save draw cost when many objects.
MAX_STATUS_LINES = 6

TRACKER_KWARGS = {
    "tracker_name": "ostrack",
    "param_name": "vitb_384_mae_ce_32x4_ep300",
    "dataset_name": "demo",
    "verbose": True,
    "min_confidence": 0.3,
    "max_center_distance_factor": 1.5,
    "min_area_ratio": 0.20,
    "max_area_ratio": 1.5,
    "consistency_relax_score": 0.55,
    "consistency_relax_factor": 4.5,
    "consistency_relax_area_margin": 0.30,
    "max_uncertain_frames": 15,
    "freeze_backend_on_uncertain": True,
    "max_lost_frames": 220,
    "verify_interval_frames": 20,
    "verify_search_frames": 4,
    "verify_score_threshold": 0.35,
    "verify_score_margin": 0.03,
    "min_identity_similarity": 0.28,
    "anchor_min_similarity": 0.25,
    "appearance_update_interval_frames": 7,
    "appearance_update_trust_frames": 7,
    "appearance_update_min_score": 0.5,
    "appearance_update_max_motion_ratio": 0.65,
    "appearance_update_min_similarity": 0.3,
    "appearance_update_alpha": 0.3,
    "appearance_update_min_sharpness": 0.0,
    "update_backend_template_on_appearance": False,
    "long_update_interval_frames": 7,
    "long_update_min_score": 0.55,
    "long_update_min_similarity": 0.37,
    "long_update_alpha_base": 0.06,
    "low_similarity_grace_frames": 3,
    "search_grid_step_factor": 1.0,
    "search_box_scale": 2.0,
    "search_max_probes": 4,
    "search_min_similarity": 0.3,
    "search_min_score": 0.3,
    "search_interval_frames": 6,
    "search_backoff_enabled": True,
    "search_backoff_scale_factor": 1.5,
    "search_backoff_max_scale": 4.5,
    "search_backoff_max_interval": 12,
}


def make_tracker(repo_root: Path) -> OSTrackTracker:
    return OSTrackTracker(repo_root=str(repo_root), **TRACKER_KWARGS)


class AppState:
    def __init__(self):
        self.mode = "SELECT"   # SELECT / TRACK
        self.paused = False
        self.last_frame = None
        self.tracks = {}
        self.next_track_id = 1
        self.frame_index = 0

    def reset_to_select(self):
        self.mode = "SELECT"
        self.paused = False
        self.last_frame = None
        self.tracks = {}
        self.next_track_id = 1
        self.frame_index = 0


def draw_track(frame, bbox, label="TRACK MODE", color=GREEN):
    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10 if y1 > 14 else y1 + 22),
        FONT,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, text: str, line_idx: int, color=WHITE):
    y = 30 + line_idx * 24
    cv2.putText(frame, text, (10, y), FONT, 0.6, color, 2, cv2.LINE_AA)


def draw_search_area(frame, bbox, label="SEARCH AREA", color=BLUE):
    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10 if y1 > 14 else y1 + 22),
        FONT,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def select_rois_on_frame(window_name: str, frame):
    rois = cv2.selectROIs(window_name, frame, fromCenter=False, showCrosshair=True)
    boxes = []
    for roi in rois:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            continue
        boxes.append(BBox(float(x), float(y), float(x + w), float(y + h)))
    return boxes


def main():
    repo_root = Path(__file__).resolve().parents[1]

    cap = cv2.VideoCapture("./assets/cars.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1e-3:
        src_fps = 30.0  # fallback if video metadata is bad

    output = cv2.VideoWriter(
        f"outputs/{time.perf_counter()}.mp4",
        cv2.VideoWriter_fourcc(*'MP4V'),
        src_fps,
        (width,height)
    )

    frame_delay_ms = max(1, int(1000 / src_fps))
    print(f"Source FPS: {src_fps:.2f}, frame delay: {frame_delay_ms} ms")

    if not cap.isOpened():
        print("ERROR: Could not open source")
        return

    state = AppState()

    window_name = "OSTrack Multi ROI Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nInstructions:")
    print("  - Press 'p' to pause and choose multiple ROIs")
    print("  - Drag ROIs, press ENTER to confirm all")
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
            state.frame_index += 1
        else:
            if state.last_frame is None:
                continue
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
                "SELECT MODE: Press 'p' to pause and choose ROIs",
                (10, 25),
                FONT,
                0.7,
                WHITE,
                2,
                cv2.LINE_AA,
            )
            if state.paused:
                draw_status(frame, "PAUSED: Draw ROIs or press 'c'", 1, YELLOW)
        else:   # TRACK mode (multi-object)
            if state.paused:
                draw_status(frame, "PAUSED: Draw ROIs to add", 0, YELLOW)
            active_count = 0
            lost_ids = []
            status_lines = []
            do_update = (state.frame_index % TRACK_UPDATE_INTERVAL) == 0
            for tid in sorted(state.tracks.keys()):
                track = state.tracks[tid]
                if not track["active"]:
                    continue
                active_count += 1
                tracker = track["tracker"]
                color = track["color"]
                needs_update = do_update or track.get("needs_first_update", False)
                if needs_update:
                    result = tracker.update(frame)
                    track["last_result"] = result
                    track["needs_first_update"] = False
                else:
                    result = track.get("last_result")
                    if result is None:
                        continue

                if result.ok and result.bbox is not None:
                    label = f"ID {tid}"
                    if result.score is not None:
                        label = f"ID {tid} {result.score:.2f}"
                    verifying = tracker.is_verifying()
                    appearance_updated = tracker.consume_appearance_update_flag() if needs_update else False
                    box_color = color
                    if verifying:
                        box_color = BLUE
                    elif appearance_updated:
                        box_color = YELLOW
                    draw_track(frame, result.bbox, label, box_color)
                elif result.state in ("UNCERTAIN", "SEARCHING"):
                    msg = result.message if result.message else "Target uncertain"
                    status_lines.append((f"ID {tid} {result.state}: {msg}", WHITE if result.state == "SEARCHING" else YELLOW))
                    if DRAW_SEARCH_HINTS and result.state == "SEARCHING":
                        hint = tracker.get_search_hint_bbox()
                        if hint is not None:
                            draw_search_area(frame, hint, label=f"SEARCH ID {tid}", color=color)
                else:
                    tracker.reset()
                    track["active"] = False
                    lost_ids.append(tid)

            for i, (text, color) in enumerate(status_lines[:MAX_STATUS_LINES]):
                draw_status(frame, text, i, color)

            if lost_ids:
                print(f"Lost tracks: {lost_ids}")

            if active_count == 0:
                state.reset_to_select()
                cv2.putText(
                    frame,
                    "All tracks lost - Back to SELECT mode",
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
        
        output.write(frame)

        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"):
            for track in state.tracks.values():
                track["tracker"].reset()
            state.reset_to_select()
            print("Reset to SELECT mode")

        elif key == ord("p") and state.last_frame is not None:
            state.paused = True
            frozen = state.last_frame.copy()
            rois = select_rois_on_frame(window_name, frozen)

            if not rois:
                print("ROI selection cancelled/empty")
                state.paused = False
                continue

            added = 0
            for roi_bbox in rois:
                tracker = make_tracker(repo_root)
                ok = tracker.initialize(frozen, roi_bbox)
                print(f"Tracker initialize [{state.next_track_id}]: {ok} | bbox={roi_bbox}")
                if not ok:
                    tracker.reset()
                    continue
                color = PALETTE[(state.next_track_id - 1) % len(PALETTE)]
                state.tracks[state.next_track_id] = {
                    "tracker": tracker,
                    "color": color,
                    "active": True,
                    "last_result": TrackResult(True, roi_bbox, None, "TRACKING", "INIT"),
                    "needs_first_update": True,
                }
                state.next_track_id += 1
                added += 1

            if added > 0:
                state.mode = "TRACK"
                print(f"Added tracks: {added} | Active: {sum(1 for t in state.tracks.values() if t['active'])}")
            else:
                print("All OSTrack initializations failed")
            state.paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
