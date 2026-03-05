import os
os.environ["MPLBACKEND"] = "Agg"

import time
from pathlib import Path

import cv2

from OSTrackMOT import OSTrackMOT


FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
}

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
DRAW_SEARCH_HINTS = True  # Disable to save draw cost when many objects.
MAX_STATUS_LINES = 6
SHOW_HELP_DEFAULT = True
ENABLE_CONCURRENCY = True
MAX_WORKERS = 6

HELP_LINES = [
    "Controls:",
    "  P: pause/resume",
    "  Mouse drag: draw ROIs (paused)",
    "  Enter: add drawn ROIs",
    "  C: clear drawn ROIs",
    "  R: reset all",
    "  H: toggle help",
    "  Q: quit",
]

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

SOURCE_PATH = "./assets/air_show.mp4"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    cap = cv2.VideoCapture(SOURCE_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {SOURCE_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1e-3:
        src_fps = 30.0  # fallback if video metadata is bad

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.perf_counter():.0f}.mp4"
    output = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MP4V"),
        src_fps,
        (width, height),
    )

    frame_delay_ms = max(1, int(1000 / src_fps))
    print(f"Source FPS: {src_fps:.2f}, frame delay: {frame_delay_ms} ms")

    mot = OSTrackMOT(
        repo_root=repo_root,
        tracker_kwargs=TRACKER_KWARGS,
        palette=PALETTE,
        update_interval=TRACK_UPDATE_INTERVAL,
        draw_search_hints=DRAW_SEARCH_HINTS,
        max_status_lines=MAX_STATUS_LINES,
        show_help_default=SHOW_HELP_DEFAULT,
        enable_concurrency=ENABLE_CONCURRENCY,
        max_workers=MAX_WORKERS,
        help_lines=HELP_LINES,
    )

    window_name = "OSTrack Multi ROI Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mot.on_mouse)

    print("\nInstructions:")
    print("  - Press 'p' to pause/resume")
    print("  - While paused, draw ROIs with mouse")
    print("  - Press ENTER to add drawn ROIs")
    print("  - Press 'c' to clear drawn ROIs")
    print("  - Press 'r' to reset all")
    print("  - Press 'h' to toggle on-screen help")
    print("  - Press 'q' to quit\n")

    prev_time = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            if not mot.paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("End of stream / failed to read frame")
                    break
                mot.on_new_frame(frame)
            else:
                if mot.last_frame is None:
                    key = cv2.waitKey(frame_delay_ms) & 0xFF
                    if key == ord("q"):
                        break
                    mot.handle_key(key, None)
                    continue
                frame = mot.last_frame.copy()

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            prev_time = now
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            mot.update_and_draw(frame, FONT, COLORS, fps_smooth)

            cv2.imshow(window_name, frame)
            output.write(frame)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == ord("q"):
                break
            mot.handle_key(key, frame)
    finally:
        mot.shutdown()
        output.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
