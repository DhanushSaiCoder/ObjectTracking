from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import sys
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return w * h if (w > 0.0 and h > 0.0) else 0.0

    def clip(self, width: int, height: int) -> "BBox":
        max_x = width - 1
        max_y = height - 1

        x1 = float(0 if self.x1 < 0 else max_x if self.x1 > max_x else self.x1)
        y1 = float(0 if self.y1 < 0 else max_y if self.y1 > max_y else self.y1)
        x2 = float(0 if self.x2 < 0 else max_x if self.x2 > max_x else self.x2)
        y2 = float(0 if self.y2 < 0 else max_y if self.y2 > max_y else self.y2)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return BBox(x1, y1, x2, y2)

    def to_xywh(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2 - self.x1), float(self.y2 - self.y1)]

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(float(x), float(y), float(x + w), float(y + h))


@dataclass
class TrackResult:
    ok: bool
    bbox: Optional[BBox]
    score: Optional[float] = None
    state: str = "IDLE"
    message: str = ""


class OSTrackTracker:
    """
    Optimized lean wrapper around OSTrack for single-target tracking.

    Runtime optimizations:
    - cache backend methods once
    - reduce repeated branching in output parsing
    - avoid unnecessary property calls/allocations
    """

    def __init__(
        self,
        repo_root: str,
        tracker_name: str = "ostrack",
        param_name: str = "vitb_384_mae_ce_32x4_ep300",
        dataset_name: str = "demo",
        min_box_area: float = 25.0,
        pad_ratio: float = 0.03,
        verbose: bool = False,
        min_confidence: float = 0.30,
        max_center_distance_factor: float = 2.0,
        min_area_ratio: float = 0.25,
        max_area_ratio: float = 4.0,
        max_uncertain_frames: int = 30,
        freeze_backend_on_uncertain: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.tracker_name = tracker_name
        self.param_name = param_name
        self.dataset_name = dataset_name
        self.min_box_area = float(min_box_area)
        self.pad_ratio = float(pad_ratio)
        self.verbose = bool(verbose)
        self.min_confidence = float(min_confidence)
        self.max_center_distance_factor = float(max_center_distance_factor)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.max_uncertain_frames = int(max_uncertain_frames)
        self.freeze_backend_on_uncertain = bool(freeze_backend_on_uncertain)

        if self.min_confidence < 0.0:
            self.min_confidence = 0.0
        elif self.min_confidence > 1.0:
            self.min_confidence = 1.0

        if self.max_center_distance_factor <= 0.0:
            self.max_center_distance_factor = 1.0

        if self.min_area_ratio <= 0.0:
            self.min_area_ratio = 0.01
        if self.max_area_ratio < self.min_area_ratio:
            self.max_area_ratio = self.min_area_ratio

        if self.max_uncertain_frames < 0:
            self.max_uncertain_frames = 0

        self._backend = None
        self._backend_initialize_fn = None
        self._backend_track_fn = None

        self._initialized = False
        self._last_bbox: Optional[BBox] = None
        self._params = None
        self._uncertain_frames = 0

        self._build_backend()

    # ---------------------------
    # Public API
    # ---------------------------
    def initialize(self, frame_bgr: np.ndarray, init_bbox_xyxy: BBox) -> bool:
        if frame_bgr is None or frame_bgr.size == 0:
            self._initialized = False
            return False

        h, w = frame_bgr.shape[:2]
        bbox = self._pad_bbox(init_bbox_xyxy, w, h, self.pad_ratio)

        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        if bw <= 1.0 or bh <= 1.0 or (bw * bh) < self.min_box_area:
            self._initialized = False
            return False

        init_info = {"init_bbox": bbox.to_xywh()}

        try:
            self._backend_initialize_fn(frame_bgr, init_info)
        except Exception as e:
            self._initialized = False
            if self.verbose:
                print(f"[OSTrackTracker] initialize failed: {e}")
            return False

        self._initialized = True
        self._last_bbox = bbox
        self._uncertain_frames = 0
        return True

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        if not self._initialized:
            return TrackResult(
                ok=False,
                bbox=None,
                score=None,
                state="IDLE",
                message="Tracker not initialized",
            )

        if frame_bgr is None or frame_bgr.size == 0:
            self._initialized = False
            return TrackResult(
                ok=False,
                bbox=None,
                score=None,
                state="LOST",
                message="Empty frame",
            )

        h, w = frame_bgr.shape[:2]

        if self.freeze_backend_on_uncertain and self._uncertain_frames > 0 and self._last_bbox is not None:
            self._set_backend_state(self._last_bbox)

        try:
            out = self._backend_track_fn(frame_bgr)
        except Exception as e:
            self._initialized = False
            return TrackResult(
                ok=False,
                bbox=None,
                score=None,
                state="LOST",
                message=f"OSTrack update exception: {e}",
            )

        bbox_xywh, score = self._parse_backend_output(out)
        if bbox_xywh is None:
            return self._mark_uncertain(score, "No bbox from tracker")

        x, y, bw, bh = bbox_xywh
        if bw <= 1.0 or bh <= 1.0 or (bw * bh) < self.min_box_area:
            return self._mark_uncertain(score, "Degenerate bbox")

        bbox = BBox(x, y, x + bw, y + bh).clip(w, h)

        if score is not None and score < self.min_confidence:
            return self._mark_uncertain(score, f"Low confidence: {score:.3f} < {self.min_confidence:.3f}")

        if self._last_bbox is not None and not self._passes_consistency_gate(self._last_bbox, bbox):
            return self._mark_uncertain(score, "Inconsistent jump (possible drift)")

        self._last_bbox = bbox
        self._uncertain_frames = 0

        return TrackResult(
            ok=True,
            bbox=bbox,
            score=score,
            state="TRACKING",
            message="OK",
        )

    def reset(self) -> None:
        self._initialized = False
        self._last_bbox = None
        self._uncertain_frames = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _build_backend(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"OSTrack repo_root not found: {self.repo_root}")

        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from lib.test.evaluation.tracker import Tracker  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import OSTrack Tracker. "
                "Check repo_root and OSTrack dependencies. "
                "Expected import: from lib.test.evaluation.tracker import Tracker"
            ) from e

        tracker_info = Tracker(
            self.tracker_name,
            self.param_name,
            self.dataset_name,
        )

        params = tracker_info.get_parameters()

        # Patch required fields once
        if not hasattr(params, "debug"):
            params.debug = 0
        if not hasattr(params, "save_all_boxes"):
            params.save_all_boxes = False
        if not hasattr(params, "tracker_name"):
            params.tracker_name = self.tracker_name
        if not hasattr(params, "param_name"):
            params.param_name = self.param_name

        backend = tracker_info.create_tracker(params)

        self._backend = backend
        self._backend_initialize_fn = backend.initialize
        self._backend_track_fn = backend.track
        self._params = params

        if self.verbose:
            print("[OSTrackTracker] Backend created successfully")


    def _mark_uncertain(self, score: Optional[float], reason: str) -> TrackResult:
        self._uncertain_frames += 1
        if self.freeze_backend_on_uncertain and self._last_bbox is not None:
            self._set_backend_state(self._last_bbox)

        if self._uncertain_frames >= self.max_uncertain_frames:
            self._initialized = False
            return TrackResult(
                ok=False,
                bbox=None,
                score=score,
                state="LOST",
                message=f"{reason} (uncertain timeout)",
            )

        return TrackResult(
            ok=False,
            bbox=None,
            score=score,
            state="UNCERTAIN",
            message=f"{reason} ({self._uncertain_frames}/{self.max_uncertain_frames})",
        )


    def _set_backend_state(self, bbox_xyxy: BBox) -> None:
        if self._backend is None:
            return

        try:
            self._backend.state = bbox_xyxy.to_xywh()
        except Exception:
            if self.verbose:
                print("[OSTrackTracker] warning: failed to reset backend state")


    def _passes_consistency_gate(self, prev_bbox: BBox, new_bbox: BBox) -> bool:
        prev_area = prev_bbox.area
        new_area = new_bbox.area
        if prev_area <= 0.0 or new_area <= 0.0:
            return False

        area_ratio = new_area / prev_area
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        prev_cx = 0.5 * (prev_bbox.x1 + prev_bbox.x2)
        prev_cy = 0.5 * (prev_bbox.y1 + prev_bbox.y2)
        new_cx = 0.5 * (new_bbox.x1 + new_bbox.x2)
        new_cy = 0.5 * (new_bbox.y1 + new_bbox.y2)

        dx = new_cx - prev_cx
        dy = new_cy - prev_cy
        center_dist = float(np.hypot(dx, dy))

        prev_diag = float(np.hypot(prev_bbox.w, prev_bbox.h))
        max_center_dist = self.max_center_distance_factor * max(prev_diag, 1.0)
        return center_dist <= max_center_dist

    def _parse_backend_output(self, out: Any) -> tuple[Optional[tuple[float, float, float, float]], Optional[float]]:
        """
        Fast-path parser for common OSTrack outputs.
        Expected common forms:
          - {'target_bbox': [x, y, w, h], 'best_score': ...}
          - {'bbox': [x, y, w, h], 'score': ...}
          - {'pred_bbox': [x, y, w, h], ...}
          - [x, y, w, h]
        """
        if out is None:
            return None, None

        if isinstance(out, dict):
            v = out.get("target_bbox", None)
            if v is None:
                v = out.get("bbox", None)
            if v is None:
                v = out.get("pred_bbox", None)

            bbox_xywh = None
            if v is not None and len(v) >= 4:
                bbox_xywh = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))

            score = out.get("best_score", None)
            if score is None:
                score = out.get("score", None)
            if score is None:
                score = out.get("conf", None)
            if score is None:
                score = out.get("confidence", None)

            return bbox_xywh, (None if score is None else float(score))

        if isinstance(out, (list, tuple, np.ndarray)) and len(out) >= 4:
            return (float(out[0]), float(out[1]), float(out[2]), float(out[3])), None

        return None, None

    @staticmethod
    def _pad_bbox(bbox: BBox, frame_w: int, frame_h: int, pad_ratio: float) -> BBox:
        if pad_ratio <= 0.0:
            return bbox.clip(frame_w, frame_h)

        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        pad_x = bw * pad_ratio
        pad_y = bh * pad_ratio

        return BBox(
            bbox.x1 - pad_x,
            bbox.y1 - pad_y,
            bbox.x2 + pad_x,
            bbox.y2 + pad_y,
        ).clip(frame_w, frame_h)