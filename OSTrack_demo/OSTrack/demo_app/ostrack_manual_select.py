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
BLUE = (255, 0, 0)


class AppState:
    def __init__(self):
        self.mode = "SELECT"   # SELECT / TRACK
        self.paused = False
        self.last_frame = None

    def reset_to_select(self):
        self.mode = "SELECT"
        self.paused = False
        self.last_frame = None


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
        min_confidence=0.3,  # Tracker score gate; increase to reject weak detections, decrease to be more permissive.
        max_center_distance_factor=1.5,  # Motion gate (center jump vs prev diag); increase for fast motion/shaky cam, decrease to curb drift.
        min_area_ratio=0.20,  # Scale gate lower bound (new/prev area); increase to reject shrink jumps, decrease for rapid scale-down.
        max_area_ratio=1.5,  # Scale gate upper bound; decrease to reject sudden growth, increase for rapid scale-up/zoom.
        consistency_relax_score=0.55,  # If score >= this, relax motion/scale gates; increase to relax less often, decrease to relax more often.
        consistency_relax_factor=4.5,  # Multiplier for allowed center jump when relaxed; increase for handheld motion, decrease to tighten.
        consistency_relax_area_margin=0.30,  # Extra tolerance on area ratio when relaxed; increase for scale volatility, decrease to tighten.
        max_uncertain_frames=15,  # Frames allowed in UNCERTAIN before SEARCHING; increase to wait longer, decrease to search sooner.
        freeze_backend_on_uncertain=True,  # Hold tracker state when uncertain; True stabilizes, False allows more motion.
        max_lost_frames=600,  # Frames allowed in SEARCHING before LOST; increase to keep trying, decrease to give up sooner.
        verify_interval_frames=20,  # Periodic verification interval; decrease to verify more often, increase to reduce verify overhead.
        verify_search_frames=4,  # Verification attempts per trigger; increase to probe longer, decrease to be lighter.
        verify_score_threshold=0.35,  # Trigger verify when score <= this; increase to verify more often, decrease to verify less.
        verify_score_margin=0.03,  # Probe must beat current score by this margin; increase to avoid switching, decrease to switch easier.
        min_identity_similarity=0.28,  # Identity gate vs stored appearance; increase to be stricter, decrease to allow more changes.
        anchor_min_similarity=0.25,  # Anchor similarity floor for updating memory; increase to prevent drift, decrease to adapt faster.
        appearance_update_interval_frames=7,  # Min frames between appearance updates; increase to update slower, decrease to update faster.
        appearance_update_trust_frames=7,  # Consecutive trusted frames before update; increase to be conservative, decrease to update sooner.
        appearance_update_min_score=0.5,  # Min score to update appearance; increase to update only high confidence, decrease to update more.
        appearance_update_max_motion_ratio=0.65,  # Max center motion ratio to allow update; decrease to avoid motion blur, increase to allow motion.
        appearance_update_min_similarity=0.3,  # Similarity to recent memory required to update; increase to prevent drift, decrease to adapt.
        appearance_update_alpha=0.3,  # EMA rate for recent appearance; increase to adapt faster, decrease to be stable.
        appearance_update_min_sharpness=0.0,  # Min sharpness to update; increase to avoid blur updates, decrease to allow more updates.
        update_backend_template_on_appearance=False,  # If True, refresh tracker template when memory updates; True adapts, False keeps stable.
        long_update_interval_frames=7,  # Min frames between long-term updates; increase to slow drift, decrease to adapt faster.
        long_update_min_score=0.55,  # Min score to update long-term memory; increase to be strict, decrease to adapt more.
        long_update_min_similarity=0.37,  # Similarity floor for long-term update; increase to prevent drift, decrease to adapt.
        long_update_alpha_base=0.06,  # Base EMA rate for long-term memory; increase to adapt faster, decrease to be stable.
        low_similarity_grace_frames=3,  # Low-sim frames before forcing SEARCHING; increase to reduce flicker.
        search_grid_step_factor=1.0,  # Grid probe step as fraction of bbox size; lower to reduce jumps.
        search_box_scale=2.0,  # Scale probe bbox size to expand search crop.
        search_max_probes=4,  # Max probes per SEARCHING frame; reduce if too slow.
        search_min_similarity=0.3,  # Min identity similarity to accept a probe.
        search_min_score=0.4,  # Min tracker score to accept a probe.
        search_interval_frames=6,  # Run SEARCHING probes every N frames (1 = every frame).
        search_backoff_enabled=True,  # Expand search area and reduce search frequency on probe misses.
        search_backoff_scale_factor=1.5,  # Scale multiplier per backoff level.
        search_backoff_max_scale=4.5,  # Max scale for search area expansion.
        search_backoff_max_interval=12,  # Max interval between search probes.
    )

    cap = cv2.VideoCapture("./assets/drone.mp4")

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
                label = "TRACK MODE"
                if result.score is not None:
                    label = f"TRACK MODE {result.score:.2f}"
                verifying = tracker.is_verifying()
                appearance_updated = tracker.consume_appearance_update_flag()
                if verifying:
                    box_color = BLUE
                elif appearance_updated:
                    box_color = YELLOW
                else:
                    box_color = GREEN
                draw_track(frame, result.bbox, label, box_color)
            elif result.state in ("UNCERTAIN", "SEARCHING"):
                msg = result.message if result.message else "Target uncertain"
                color = YELLOW if result.state == "UNCERTAIN" else WHITE
                cv2.putText(
                    frame,
                    f"{result.state}: {msg}",
                    (10, 30),
                    FONT,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                if result.state == "SEARCHING":
                    hint = tracker.get_search_hint_bbox()
                    if hint is not None:
                        draw_search_area(frame, hint)
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
        
        output.write(frame)

        key = cv2.waitKey(frame_delay_ms) & 0xFF
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
