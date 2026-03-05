import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict

from ostrack_tracker import OSTrackTracker, BBox, TrackResult


class OSTrackMOT:
    def __init__(
        self,
        repo_root: Path,
        tracker_kwargs: Dict,
        palette: List[Tuple[int, int, int]],
        update_interval: int = 2,
        draw_search_hints: bool = False,
        max_status_lines: int = 6,
        show_help_default: bool = True,
        enable_concurrency: bool = True,
        max_workers: int = 4,
        help_lines: Optional[List[str]] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.tracker_kwargs = dict(tracker_kwargs)
        self.palette = list(palette)
        self.update_interval = max(int(update_interval), 1)
        self.draw_search_hints = bool(draw_search_hints)
        self.max_status_lines = max(int(max_status_lines), 1)
        self.show_help = bool(show_help_default)
        self.enable_concurrency = bool(enable_concurrency)
        self.max_workers = max(int(max_workers), 1)
        self.help_lines = list(help_lines) if help_lines else []

        self.mode = "SELECT"  # SELECT / TRACK
        self.paused = False
        self.last_frame = None
        self.frame_index = 0

        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1

        self.pending_rois: List[BBox] = []
        self.is_drawing = False
        self.drag_start = None
        self.drag_current = None

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.enable_concurrency else None

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def reset(self) -> None:
        for track in self.tracks.values():
            track["tracker"].reset()
        self.mode = "SELECT"
        self.paused = False
        self.tracks = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.pending_rois = []
        self.is_drawing = False
        self.drag_start = None
        self.drag_current = None

    def make_tracker(self) -> OSTrackTracker:
        return OSTrackTracker(repo_root=str(self.repo_root), **self.tracker_kwargs)

    def on_new_frame(self, frame) -> None:
        self.last_frame = frame
        self.frame_index += 1

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused:
            self.clear_pending_rois()

    def clear_pending_rois(self) -> None:
        self.pending_rois = []
        self.is_drawing = False
        self.drag_start = None
        self.drag_current = None

    def add_rois(self, frame, rois: List[BBox]) -> int:
        if not rois:
            return 0
        added = 0
        for roi_bbox in rois:
            tracker = self.make_tracker()
            ok = tracker.initialize(frame, roi_bbox)
            if not ok:
                tracker.reset()
                continue
            color = self.palette[(self.next_track_id - 1) % len(self.palette)]
            self.tracks[self.next_track_id] = {
                "tracker": tracker,
                "color": color,
                "active": True,
                "last_result": TrackResult(True, roi_bbox, None, "TRACKING", "INIT"),
                "needs_first_update": True,
            }
            self.next_track_id += 1
            added += 1
        if added > 0:
            self.mode = "TRACK"
        return added

    def on_mouse(self, event, x, y, flags, param=None) -> None:
        if not self.paused:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            self.drag_current = (x, y)
            if self.drag_start is None or self.drag_current is None:
                self.drag_start = None
                self.drag_current = None
                return
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            bx1, bx2 = (x1, x2) if x1 <= x2 else (x2, x1)
            by1, by2 = (y1, y2) if y1 <= y2 else (y2, y1)
            if abs(bx2 - bx1) > 1 and abs(by2 - by1) > 1:
                self.pending_rois.append(BBox(float(bx1), float(by1), float(bx2), float(by2)))
            self.drag_start = None
            self.drag_current = None

    def handle_key(self, key: int, frame) -> None:
        if key == ord("h"):
            self.show_help = not self.show_help
            return
        if key == ord("r"):
            self.reset()
            return
        if key == ord("p"):
            self.toggle_pause()
            return
        if key in (13, 10) and self.paused:
            if frame is not None:
                self.add_rois(frame, list(self.pending_rois))
            self.clear_pending_rois()
            return
        if key == ord("c") and self.paused:
            self.clear_pending_rois()

    def _draw_track(self, frame, bbox, label, color, font, thickness=3):
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 10 if y1 > 14 else y1 + 22), font, 0.7, color, 2, cv2.LINE_AA)

    def _draw_status(self, frame, text, line_idx, color, font):
        y = 30 + line_idx * 24
        cv2.putText(frame, text, (10, y), font, 0.6, color, 2, cv2.LINE_AA)

    def _draw_search_area(self, frame, bbox, label, color, font):
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10 if y1 > 14 else y1 + 22), font, 0.7, color, 2, cv2.LINE_AA)

    def _draw_help(self, frame, lines, start_y, color, font):
        y = start_y
        for line in lines:
            cv2.putText(frame, line, (10, y), font, 0.55, color, 2, cv2.LINE_AA)
            y += 22

    def _draw_pending_rois(self, frame, font, color):
        for bbox in self.pending_rois:
            self._draw_search_area(frame, bbox, "PENDING", color, font)
        if self.drag_start is not None and self.drag_current is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            bx1, bx2 = (x1, x2) if x1 <= x2 else (x2, x1)
            by1, by2 = (y1, y2) if y1 <= y2 else (y2, y1)
            bbox = BBox(float(bx1), float(by1), float(bx2), float(by2))
            self._draw_search_area(frame, bbox, "DRAW", color, font)

    def update_and_draw(
        self,
        frame,
        font,
        colors: dict,
        fps_smooth: float,
    ) -> None:
        if frame is None:
            return

        active_tracks = sum(1 for t in self.tracks.values() if t["active"])
        mode_line = f"MODE: {self.mode} | Active: {active_tracks} | {'PAUSED' if self.paused else 'LIVE'}"
        self._draw_status(frame, mode_line, 0, colors["white"], font)

        if self.mode == "SELECT":
            self._draw_status(frame, "Press 'p' to pause and draw ROIs", 1, colors["white"], font)
            if self.paused:
                self._draw_status(frame, "PAUSED: Draw ROIs, Enter to add, P to resume", 2, colors["yellow"], font)
        else:
            if self.paused:
                self._draw_status(frame, "PAUSED: Draw ROIs, Enter to add, P to resume", 1, colors["yellow"], font)

            do_update = (not self.paused) and (self.frame_index % self.update_interval == 0)
            active_ids = [tid for tid in sorted(self.tracks.keys()) if self.tracks[tid]["active"]]

            results_by_id = {}
            if do_update and active_ids:
                if self.enable_concurrency and self.executor is not None:
                    futures = {}
                    for tid in active_ids:
                        track = self.tracks[tid]
                        if track.get("needs_first_update", False) or do_update:
                            futures[self.executor.submit(track["tracker"].update, frame)] = tid
                    for fut in as_completed(futures):
                        tid = futures[fut]
                        try:
                            results_by_id[tid] = fut.result()
                        except Exception as exc:
                            results_by_id[tid] = TrackResult(False, None, None, "LOST", f"Update exception: {exc}")
                else:
                    for tid in active_ids:
                        track = self.tracks[tid]
                        if track.get("needs_first_update", False) or do_update:
                            try:
                                results_by_id[tid] = track["tracker"].update(frame)
                            except Exception as exc:
                                results_by_id[tid] = TrackResult(False, None, None, "LOST", f"Update exception: {exc}")

            status_lines = []
            lost_ids = []
            for tid in active_ids:
                track = self.tracks[tid]
                color = track["color"]
                tracker = track["tracker"]
                needs_update = do_update or track.get("needs_first_update", False)
                if needs_update and tid in results_by_id:
                    result = results_by_id[tid]
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
                        box_color = colors["blue"]
                    elif appearance_updated:
                        box_color = colors["yellow"]
                    self._draw_track(frame, result.bbox, label, box_color, font)
                elif result.state in ("UNCERTAIN", "SEARCHING"):
                    msg = result.message if result.message else "Target uncertain"
                    status_lines.append((f"ID {tid} {result.state}: {msg}", colors["white"] if result.state == "SEARCHING" else colors["yellow"]))
                    if self.draw_search_hints and result.state == "SEARCHING":
                        hint = tracker.get_search_hint_bbox()
                        if hint is not None:
                            self._draw_search_area(frame, hint, f"SEARCH ID {tid}", color, font)
                else:
                    tracker.reset()
                    track["active"] = False
                    lost_ids.append(tid)

            for i, (text, color) in enumerate(status_lines[: self.max_status_lines]):
                self._draw_status(frame, text, i + 1, color, font)

            for tid in lost_ids:
                self.tracks.pop(tid, None)

            if self.mode == "TRACK" and not any(track["active"] for track in self.tracks.values()):
                self.mode = "SELECT"

        if self.paused:
            self._draw_pending_rois(frame, font, colors["cyan"])

        if self.show_help and self.help_lines:
            help_height = 22 * len(self.help_lines)
            frame_h = frame.shape[0]
            self._draw_help(frame, self.help_lines, frame_h - help_height - 20, colors["white"], font)

        frame_h = frame.shape[0]
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, frame_h - 10), font, 0.7, colors["white"], 2, cv2.LINE_AA)
