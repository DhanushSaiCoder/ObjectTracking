from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import sys

import cv2
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
    def __init__(
        self,
        repo_root: str,
        tracker_name: str = "ostrack",
        param_name: str = "vitb_384_mae_ce_32x4_ep300",
        dataset_name: str = "demo",
        min_box_area: float = 25.0,  # Min bbox area; increase to ignore tiny blobs, decrease for small targets.
        pad_ratio: float = 0.03,  # ROI padding around bbox; increase for context, decrease for tighter crop.
        verbose: bool = False,  # Enable debug prints; True for troubleshooting, False for quiet.
        min_confidence: float = 0.25,  # Score gate; increase to reject weak detections, decrease to be permissive.
        max_center_distance_factor: float = 3.5,  # Motion gate; increase for fast motion, decrease to curb drift.
        min_area_ratio: float = 0.20,  # Scale lower bound; increase to reject shrink jumps, decrease for rapid scale-down.
        max_area_ratio: float =1.5,  # Scale upper bound; decrease to reject growth jumps, increase for rapid scale-up.
        consistency_relax_score: float = 0.6,  # If score >= this, relax motion/scale gates; increase to relax less.
        consistency_relax_factor: float = 1.8,  # Extra motion allowance when relaxed; increase for shaky cam, decrease to tighten.
        consistency_relax_area_margin: float = 0.15,  # Extra scale tolerance when relaxed; increase for scale volatility.
        max_uncertain_frames: int = 30,  # UNCERTAIN frames before SEARCHING; increase to wait longer, decrease to search sooner.
        freeze_backend_on_uncertain: bool = True,  # Hold backend state while uncertain; True stabilizes, False allows motion.
        max_lost_frames: int = 360,  # SEARCHING frames before LOST; increase to keep trying, decrease to give up sooner.
        verify_interval_frames: int = 40,  # Periodic verification interval; decrease to verify more often.
        verify_search_frames: int = 4,  # Verification attempts per trigger; increase to probe longer.
        verify_score_threshold: float = 0.35,  # Trigger verify when score <= this; increase to verify more often.
        verify_score_margin: float = 0.03,  # Probe must beat current score by this margin; increase to avoid switching.
        min_identity_similarity: float = 0.1,  # Identity gate vs memory; increase to be strict, decrease to allow change.
        anchor_min_similarity: float = 0.25,  # Anchor similarity floor for updates; increase to prevent drift.
        appearance_update_interval_frames: int = 10,  # Min frames between updates; increase to update slower.
        appearance_update_trust_frames: int = 8,  # Trusted consecutive frames before update; increase to be conservative.
        appearance_update_min_score: float = 0.55,  # Min score to update memory; increase to update only high confidence.
        appearance_update_max_motion_ratio: float = 0.35,  # Max motion ratio to allow update; decrease to avoid blur.
        appearance_update_min_similarity: float = 0.35,  # Similarity to recent memory required; increase to prevent drift.
        appearance_update_alpha: float = 0.2,  # EMA rate for recent memory; increase to adapt faster.
        appearance_update_min_sharpness: float = 0.0,  # Min sharpness to update; increase to avoid blur updates.
        update_backend_template_on_appearance: bool = False,  # If True, refresh template on update; True adapts more.
        long_update_interval_frames: int = 20,  # Min frames between long-term updates; increase to slow drift.
        long_update_min_score: float = 0.6,  # Min score to update long-term memory; increase to be strict.
        long_update_min_similarity: float = 0.25,  # Similarity floor for long-term update; increase to prevent drift.
        long_update_alpha_base: float = 0.1,  # Base EMA rate for long-term memory; increase to adapt faster.
        low_similarity_grace_frames: int = 3,  # Frames to tolerate low similarity before forcing SEARCHING.
        search_grid_step_factor: float = 1.0,  # Grid probe step as fraction of bbox size during SEARCHING.
        search_box_scale: float = 1.5,  # Scale factor for probe bbox size during SEARCHING.
        search_max_probes: int = 7,  # Max probe attempts per SEARCHING frame.
        search_min_similarity: float = 0.30,  # Min identity similarity to accept a probe.
        search_min_score: float = 0.3,  # Min tracker score to accept a probe.
        search_interval_frames: int = 2,  # Run SEARCHING probes every N frames (1 = every frame).
        search_backoff_enabled: bool = False,  # If True, expand search area and reduce search frequency on misses.
        search_backoff_scale_factor: float = 1.25,  # Multiplicative scale per backoff level.
        search_backoff_max_scale: float = 3.0,  # Cap on search box scale during backoff.
        search_backoff_max_interval: int = 8,  # Cap on search interval frames during backoff.
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
        self.consistency_relax_score = float(consistency_relax_score)
        self.consistency_relax_factor = float(consistency_relax_factor)
        self.consistency_relax_area_margin = float(consistency_relax_area_margin)
        self.max_uncertain_frames = int(max_uncertain_frames)
        self.freeze_backend_on_uncertain = bool(freeze_backend_on_uncertain)
        self.max_lost_frames = int(max_lost_frames)

        self.verify_interval_frames = int(verify_interval_frames)
        self.verify_search_frames = int(verify_search_frames)
        self.verify_score_threshold = float(verify_score_threshold)
        self.verify_score_margin = float(verify_score_margin)
        self.min_identity_similarity = float(min_identity_similarity)
        self.anchor_min_similarity = float(anchor_min_similarity)
        self.appearance_update_interval_frames = int(appearance_update_interval_frames)
        self.appearance_update_trust_frames = int(appearance_update_trust_frames)
        self.appearance_update_min_score = float(appearance_update_min_score)
        self.appearance_update_max_motion_ratio = float(appearance_update_max_motion_ratio)
        self.appearance_update_min_similarity = float(appearance_update_min_similarity)
        self.appearance_update_alpha = float(appearance_update_alpha)
        self.appearance_update_min_sharpness = float(appearance_update_min_sharpness)
        self.update_backend_template_on_appearance = bool(update_backend_template_on_appearance)
        self.long_update_interval_frames = int(long_update_interval_frames)
        self.long_update_min_score = float(long_update_min_score)
        self.long_update_min_similarity = float(long_update_min_similarity)
        self.long_update_alpha_base = float(long_update_alpha_base)
        self.low_similarity_grace_frames = int(low_similarity_grace_frames)
        self.search_grid_step_factor = float(search_grid_step_factor)
        self.search_box_scale = float(search_box_scale)
        self.search_max_probes = int(search_max_probes)
        self.search_min_similarity = float(search_min_similarity)
        self.search_min_score = float(search_min_score)
        self.search_interval_frames = int(search_interval_frames)
        self.search_backoff_enabled = bool(search_backoff_enabled)
        self.search_backoff_scale_factor = float(search_backoff_scale_factor)
        self.search_backoff_max_scale = float(search_backoff_max_scale)
        self.search_backoff_max_interval = int(search_backoff_max_interval)

        self.min_confidence = min(max(self.min_confidence, 0.0), 1.0)
        self.verify_score_threshold = min(max(self.verify_score_threshold, 0.0), 1.0)
        self.verify_score_margin = max(self.verify_score_margin, 0.0)
        self.min_identity_similarity = min(max(self.min_identity_similarity, 0.0), 1.0)
        self.anchor_min_similarity = min(max(self.anchor_min_similarity, 0.0), 1.0)
        self.consistency_relax_score = min(max(self.consistency_relax_score, 0.0), 1.0)
        self.consistency_relax_factor = max(self.consistency_relax_factor, 1.0)
        self.consistency_relax_area_margin = min(max(self.consistency_relax_area_margin, 0.0), 0.9)
        self.appearance_update_min_score = min(max(self.appearance_update_min_score, 0.0), 1.0)
        self.appearance_update_min_similarity = min(max(self.appearance_update_min_similarity, 0.0), 1.0)
        self.appearance_update_alpha = min(max(self.appearance_update_alpha, 0.01), 1.0)
        self.appearance_update_max_motion_ratio = max(self.appearance_update_max_motion_ratio, 0.0)
        self.appearance_update_min_sharpness = max(self.appearance_update_min_sharpness, 0.0)
        self.long_update_min_score = min(max(self.long_update_min_score, 0.0), 1.0)
        self.long_update_min_similarity = min(max(self.long_update_min_similarity, 0.0), 1.0)
        self.long_update_alpha_base = min(max(self.long_update_alpha_base, 0.0), 1.0)
        self.search_grid_step_factor = max(self.search_grid_step_factor, 0.0)
        if self.search_box_scale <= 0.0:
            self.search_box_scale = 1.0
        self.search_min_similarity = min(max(self.search_min_similarity, 0.0), 1.0)
        self.search_min_score = min(max(self.search_min_score, 0.0), 1.0)
        if self.search_interval_frames < 1:
            self.search_interval_frames = 1
        if self.search_backoff_scale_factor < 1.0:
            self.search_backoff_scale_factor = 1.0
        if self.search_backoff_max_scale <= 0.0:
            self.search_backoff_max_scale = self.search_box_scale
        if self.search_backoff_max_scale < self.search_box_scale:
            self.search_backoff_max_scale = self.search_box_scale
        if self.search_backoff_max_interval < 1:
            self.search_backoff_max_interval = self.search_interval_frames
        if self.search_backoff_max_interval < self.search_interval_frames:
            self.search_backoff_max_interval = self.search_interval_frames

        if self.max_center_distance_factor <= 0.0:
            self.max_center_distance_factor = 1.0
        if self.min_area_ratio <= 0.0:
            self.min_area_ratio = 0.01
        if self.max_area_ratio < self.min_area_ratio:
            self.max_area_ratio = self.min_area_ratio
        if self.max_uncertain_frames < 0:
            self.max_uncertain_frames = 0
        if self.max_lost_frames < 0:
            self.max_lost_frames = 0
        if self.low_similarity_grace_frames < 0:
            self.low_similarity_grace_frames = 0
        if self.search_max_probes < 0:
            self.search_max_probes = 0
        if self.verify_interval_frames < 1:
            self.verify_interval_frames = 1
        if self.verify_search_frames < 0:
            self.verify_search_frames = 0
        if self.appearance_update_interval_frames < 0:
            self.appearance_update_interval_frames = 0
        if self.appearance_update_trust_frames < 1:
            self.appearance_update_trust_frames = 1
        if self.long_update_interval_frames < 0:
            self.long_update_interval_frames = 0

        self._backend = None
        self._backend_initialize_fn = None
        self._backend_track_fn = None

        self._initialized = False
        self._last_bbox: Optional[BBox] = None
        self._params = None
        self._uncertain_frames = 0
        self._lost_frames = 0
        self._low_similarity_frames = 0
        self._search_frame_count = 0
        self._search_backoff_level = 0
        self._frame_count = 0
        self._verify_remaining = 0
        self._verification_active = False

        self._init_bbox: Optional[BBox] = None
        self._anchor_hist: Optional[np.ndarray] = None
        self._recent_hist: Optional[np.ndarray] = None
        self._recent_bbox: Optional[BBox] = None
        self._long_hist: Optional[np.ndarray] = None
        self._long_bbox: Optional[BBox] = None
        self._appearance_last_update_frame = -1
        self._long_last_update_frame = -1
        self._appearance_updated = False
        self._trusted_frames = 0
        self._search_hint_bbox: Optional[BBox] = None

        self._build_backend()

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
        self._init_bbox = bbox
        self._anchor_hist = self._compute_bbox_hist(frame_bgr, bbox)
        self._recent_hist = self._anchor_hist
        self._recent_bbox = bbox
        self._long_hist = self._anchor_hist
        self._long_bbox = bbox
        self._appearance_last_update_frame = 0
        self._long_last_update_frame = 0
        self._appearance_updated = False
        self._trusted_frames = 0
        self._search_hint_bbox = None
        self._uncertain_frames = 0
        self._lost_frames = 0
        self._low_similarity_frames = 0
        self._search_frame_count = 0
        self._search_backoff_level = 0
        self._frame_count = 0
        self._verify_remaining = 0
        return True

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        if not self._initialized:
            return TrackResult(False, None, None, "IDLE", "Tracker not initialized")

        if frame_bgr is None or frame_bgr.size == 0:
            self._initialized = False
            return TrackResult(False, None, None, "LOST", "Empty frame")

        self._appearance_updated = False
        self._verification_active = False
        h, w = frame_bgr.shape[:2]

        prev_bbox = self._last_bbox

        is_searching = self._uncertain_frames > 0 and self._uncertain_frames >= self.max_uncertain_frames
        if is_searching:
            self._update_search_hint(frame_bgr)
            if not self._should_run_search():
                return self._mark_uncertain(None, "Searching (cooldown)")
            search_result = self._run_search(frame_bgr)
            if search_result is not None:
                return search_result
        else:
            self._search_frame_count = 0
            self._search_backoff_level = 0
            self._search_hint_bbox = None

        if self.freeze_backend_on_uncertain and self._uncertain_frames > 0 and not is_searching and self._last_bbox is not None:
            self._set_backend_state(self._last_bbox)

        try:
            out = self._backend_track_fn(frame_bgr)
        except Exception as e:
            self._initialized = False
            return TrackResult(False, None, None, "LOST", f"OSTrack update exception: {e}")

        bbox_xywh, score = self._parse_backend_output(out)
        if bbox_xywh is None:
            self._low_similarity_frames = 0
            return self._mark_uncertain(score, "No bbox from tracker")

        x, y, bw, bh = bbox_xywh
        if bw <= 1.0 or bh <= 1.0 or (bw * bh) < self.min_box_area:
            self._low_similarity_frames = 0
            return self._mark_uncertain(score, "Degenerate bbox")

        bbox = BBox(x, y, x + bw, y + bh).clip(w, h)

        if score is not None and score < self.min_confidence:
            self._low_similarity_frames = 0
            return self._mark_uncertain(score, f"Low confidence: {score:.3f} < {self.min_confidence:.3f}")

        if self._last_bbox is not None and not self._passes_consistency_gate(self._last_bbox, bbox, score):
            self._low_similarity_frames = 0
            return self._mark_uncertain(score, "Inconsistent jump (possible drift)")

        similarity = self._compute_identity_similarity(frame_bgr, bbox)
        if similarity is not None and similarity < self.min_identity_similarity:
            self._low_similarity_frames += 1
            if self._low_similarity_frames >= self.low_similarity_grace_frames:
                self._force_searching()
            return self._mark_uncertain(score, f"Low identity similarity: {similarity:.3f}")
        self._low_similarity_frames = 0

        self._frame_count += 1
        if self._should_verify(score):
            self._verification_active = True
            verified = self._run_identity_verification(frame_bgr, bbox, score)
            if verified is not None:
                return verified

        self._maybe_update_appearance(frame_bgr, bbox, score, prev_bbox)
        self._maybe_update_long_appearance(frame_bgr, bbox, score, prev_bbox)
        self._last_bbox = bbox
        self._uncertain_frames = 0
        self._lost_frames = 0
        self._low_similarity_frames = 0
        self._search_frame_count = 0
        self._search_backoff_level = 0
        self._search_hint_bbox = None

        return TrackResult(True, bbox, score, "TRACKING", "OK")

    def reset(self) -> None:
        self._initialized = False
        self._last_bbox = None
        self._uncertain_frames = 0
        self._lost_frames = 0
        self._low_similarity_frames = 0
        self._search_frame_count = 0
        self._search_backoff_level = 0
        self._frame_count = 0
        self._verify_remaining = 0
        self._init_bbox = None
        self._anchor_hist = None
        self._recent_hist = None
        self._recent_bbox = None
        self._appearance_last_update_frame = -1
        self._long_hist = None
        self._long_bbox = None
        self._long_last_update_frame = -1
        self._appearance_updated = False
        self._trusted_frames = 0
        self._verification_active = False
        self._search_hint_bbox = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

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

        tracker_info = Tracker(self.tracker_name, self.param_name, self.dataset_name)
        params = tracker_info.get_parameters()

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
        is_searching = self._uncertain_frames > 0 and self._uncertain_frames >= self.max_uncertain_frames
        if self.freeze_backend_on_uncertain and self._last_bbox is not None and not is_searching:
            self._set_backend_state(self._last_bbox)

        if is_searching:
            self._lost_frames += 1
            if self._lost_frames > self.max_lost_frames:
                self._initialized = False
                return TrackResult(False, None, score, "LOST", f"{reason} (lost timeout)")
            return TrackResult(
                False,
                None,
                score,
                "SEARCHING",
                f"{reason} (searching {self._lost_frames}/{self.max_lost_frames})",
            )

        return TrackResult(
            False,
            None,
            score,
            "UNCERTAIN",
            f"{reason} ({self._uncertain_frames}/{self.max_uncertain_frames})",
        )

    def _set_backend_state(self, bbox_xyxy: BBox) -> None:
        if self._backend is None:
            return
        try:
            self._backend.state = bbox_xyxy.to_xywh()
        except Exception:
            if self.verbose:
                print("[OSTrackTracker] warning: failed to reset backend state")

    def _passes_consistency_gate(
        self,
        prev_bbox: BBox,
        new_bbox: BBox,
        score: Optional[float],
    ) -> bool:
        prev_area = prev_bbox.area
        new_area = new_bbox.area
        if prev_area <= 0.0 or new_area <= 0.0:
            return False
        min_area_ratio = self.min_area_ratio
        max_area_ratio = self.max_area_ratio
        max_center_factor = self.max_center_distance_factor
        if score is not None and score >= self.consistency_relax_score:
            max_center_factor *= self.consistency_relax_factor
            area_margin = self.consistency_relax_area_margin
            min_area_ratio = max(min_area_ratio * (1.0 - area_margin), 0.01)
            max_area_ratio = max_area_ratio * (1.0 + area_margin)

        area_ratio = new_area / prev_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False

        prev_cx = 0.5 * (prev_bbox.x1 + prev_bbox.x2)
        prev_cy = 0.5 * (prev_bbox.y1 + prev_bbox.y2)
        new_cx = 0.5 * (new_bbox.x1 + new_bbox.x2)
        new_cy = 0.5 * (new_bbox.y1 + new_bbox.y2)

        center_dist = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))
        prev_diag = float(np.hypot(prev_bbox.w, prev_bbox.h))
        max_center_dist = max_center_factor * max(prev_diag, 1.0)
        return center_dist <= max_center_dist

    def _parse_backend_output(self, out: Any) -> tuple[Optional[tuple[float, float, float, float]], Optional[float]]:
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

    def _should_verify(self, score: Optional[float]) -> bool:
        if self.verify_search_frames <= 0:
            return False

        if self._verify_remaining > 0:
            return True

        if self._frame_count % self.verify_interval_frames != 0:
            return False

        if score is None:
            self._verify_remaining = self.verify_search_frames
            return True

        if score <= self.verify_score_threshold:
            self._verify_remaining = self.verify_search_frames
            return True

        return False

    def _run_identity_verification(self, frame_bgr: np.ndarray, tracked_bbox: BBox, tracked_score: Optional[float]) -> Optional[TrackResult]:
        probe_seed = self._recent_bbox if self._recent_bbox is not None else self._init_bbox
        if probe_seed is None:
            return None

        probe = self._probe_state(frame_bgr, probe_seed)
        self._verify_remaining = max(self._verify_remaining - 1, 0)

        if probe is None:
            return None

        probe_bbox, probe_score = probe
        if probe_score is None:
            return None

        tracked_score_val = -1.0 if tracked_score is None else float(tracked_score)
        if probe_score <= (tracked_score_val + self.verify_score_margin):
            return None

        if self._bbox_iou(tracked_bbox, probe_bbox) > 0.4:
            return None

        probe_similarity = self._compute_identity_similarity(frame_bgr, probe_bbox)
        tracked_similarity = self._compute_identity_similarity(frame_bgr, tracked_bbox)

        if probe_similarity is None:
            return None

        if tracked_similarity is None:
            tracked_similarity = 0.0

        if probe_similarity > tracked_similarity + 0.05:
            self._set_backend_state(probe_bbox)
            self._last_bbox = probe_bbox
            self._uncertain_frames = 0
            self._lost_frames = 0
            self._low_similarity_frames = 0
            return TrackResult(
                True,
                probe_bbox,
                probe_score,
                "TRACKING",
                "Re-identified target during verification",
            )

        self._low_similarity_frames += 1
        if self._low_similarity_frames >= self.low_similarity_grace_frames:
            self._force_searching()
        return self._mark_uncertain(tracked_score, "Verification mismatch")

    def _force_searching(self) -> None:
        if self.max_uncertain_frames < 0:
            self.max_uncertain_frames = 0
        self._uncertain_frames = max(self._uncertain_frames, self.max_uncertain_frames)
        self._lost_frames = max(self._lost_frames, 1)
        self._search_backoff_level = 0
        self._search_frame_count = 0

    def _run_search(self, frame_bgr: np.ndarray) -> Optional[TrackResult]:
        if self._backend is None:
            return None
        if self.search_max_probes <= 0:
            return self._mark_uncertain(None, "Searching (probe miss)")

        h, w = frame_bgr.shape[:2]
        seeds: list[BBox] = []
        if self._recent_bbox is not None:
            seeds.append(self._recent_bbox)
        if self._long_bbox is not None:
            seeds.append(self._long_bbox)
        if self._init_bbox is not None:
            seeds.append(self._init_bbox)

        if not seeds:
            return self._mark_uncertain(None, "Searching (probe miss)")

        offsets_primary = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        best_bbox: Optional[BBox] = None
        best_score: Optional[float] = None
        best_similarity: Optional[float] = None
        probes = 0

        for seed_index, seed in enumerate(seeds):
            offsets = offsets_primary if seed_index == 0 else [(0.0, 0.0)]
            step_x = seed.w * self.search_grid_step_factor
            step_y = seed.h * self.search_grid_step_factor
            scale = self._effective_search_box_scale()
            bw = seed.w * scale
            bh = seed.h * scale
            if bw <= 1.0 or bh <= 1.0:
                continue

            seed_cx = 0.5 * (seed.x1 + seed.x2)
            seed_cy = 0.5 * (seed.y1 + seed.y2)

            for ox, oy in offsets:
                if probes >= self.search_max_probes:
                    break

                cx = seed_cx + ox * step_x
                cy = seed_cy + oy * step_y
                candidate = BBox(cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh).clip(w, h)
                if candidate.w <= 1.0 or candidate.h <= 1.0 or candidate.area < self.min_box_area:
                    continue

                probes += 1
                probe = self._probe_state(frame_bgr, candidate)
                if probe is None:
                    continue
                probe_bbox, probe_score = probe
                if probe_bbox.area < self.min_box_area:
                    continue
                score_floor = max(self.search_min_score, self.min_confidence)
                if probe_score is None or probe_score < score_floor:
                    continue

                similarity = self._compute_anchor_similarity(frame_bgr, probe_bbox)
                if similarity is None:
                    similarity = self._compute_identity_similarity(frame_bgr, probe_bbox)
                if similarity is None or similarity < self.search_min_similarity:
                    continue

                if best_similarity is None or similarity > best_similarity or (
                    similarity == best_similarity and best_score is not None and probe_score > best_score
                ):
                    best_similarity = similarity
                    best_score = probe_score
                    best_bbox = probe_bbox

            if probes >= self.search_max_probes:
                break

        if best_bbox is not None and best_score is not None:
            self._set_backend_state(best_bbox)
            self._last_bbox = best_bbox
            self._uncertain_frames = 0
            self._lost_frames = 0
            self._low_similarity_frames = 0
            self._search_hint_bbox = None
            self._search_backoff_level = 0
            return TrackResult(True, best_bbox, best_score, "TRACKING", "Reacquired in search")

        self._bump_search_backoff()
        return self._mark_uncertain(None, "Searching (probe miss)")

    def _should_run_search(self) -> bool:
        interval = self._effective_search_interval()
        if interval <= 1:
            return True
        self._search_frame_count += 1
        if self._search_frame_count >= interval:
            self._search_frame_count = 0
            return True
        return False

    def _compute_search_interval_for_level(self, level: int) -> int:
        interval = self.search_interval_frames * (2 ** max(level, 0))
        if self.search_backoff_max_interval > 0:
            interval = min(interval, self.search_backoff_max_interval)
        return max(int(interval), 1)

    def _compute_search_scale_for_level(self, level: int) -> float:
        scale = self.search_box_scale * (self.search_backoff_scale_factor ** max(level, 0))
        if self.search_backoff_max_scale > 0.0:
            scale = min(scale, self.search_backoff_max_scale)
        return max(float(scale), 0.1)

    def _effective_search_interval(self) -> int:
        if not self.search_backoff_enabled:
            return self.search_interval_frames
        return self._compute_search_interval_for_level(self._search_backoff_level)

    def _effective_search_box_scale(self) -> float:
        if not self.search_backoff_enabled:
            return self.search_box_scale
        return self._compute_search_scale_for_level(self._search_backoff_level)

    def _bump_search_backoff(self) -> None:
        if not self.search_backoff_enabled:
            return
        current_interval = self._compute_search_interval_for_level(self._search_backoff_level)
        current_scale = self._compute_search_scale_for_level(self._search_backoff_level)
        next_level = self._search_backoff_level + 1
        next_interval = self._compute_search_interval_for_level(next_level)
        next_scale = self._compute_search_scale_for_level(next_level)
        if next_interval > current_interval or next_scale > current_scale:
            self._search_backoff_level = next_level
            self._search_frame_count = 0

    def _update_search_hint(self, frame_bgr: np.ndarray) -> None:
        if self._recent_bbox is not None:
            seed = self._recent_bbox
        elif self._long_bbox is not None:
            seed = self._long_bbox
        elif self._init_bbox is not None:
            seed = self._init_bbox
        else:
            self._search_hint_bbox = None
            return

        h, w = frame_bgr.shape[:2]
        step_x = seed.w * self.search_grid_step_factor
        step_y = seed.h * self.search_grid_step_factor
        scale = self._effective_search_box_scale()
        bw = seed.w * scale + (2.0 * step_x)
        bh = seed.h * scale + (2.0 * step_y)
        if bw <= 1.0 or bh <= 1.0:
            self._search_hint_bbox = None
            return

        cx = 0.5 * (seed.x1 + seed.x2)
        cy = 0.5 * (seed.y1 + seed.y2)
        hint = BBox(cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh).clip(w, h)
        self._search_hint_bbox = hint

    def _probe_state(self, frame_bgr: np.ndarray, state_bbox: BBox) -> Optional[tuple[BBox, Optional[float]]]:
        if self._backend is None:
            return None

        saved_state = None
        if hasattr(self._backend, "state"):
            st = getattr(self._backend, "state")
            if isinstance(st, (list, tuple)) and len(st) >= 4:
                saved_state = [float(st[0]), float(st[1]), float(st[2]), float(st[3])]

        try:
            self._set_backend_state(state_bbox)
            out = self._backend_track_fn(frame_bgr)
            bbox_xywh, score = self._parse_backend_output(out)
            if bbox_xywh is None:
                return None
            x, y, bw, bh = bbox_xywh
            if bw <= 1.0 or bh <= 1.0:
                return None
            bbox = BBox(x, y, x + bw, y + bh)
            return bbox, score
        except Exception:
            return None
        finally:
            if saved_state is not None:
                try:
                    self._backend.state = saved_state
                except Exception:
                    pass
            if self._last_bbox is not None:
                self._set_backend_state(self._last_bbox)

    def _compute_bbox_hist(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        b = bbox.clip(w, h)
        x1, y1 = int(b.x1), int(b.y1)
        x2, y2 = int(b.x2), int(b.y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_identity_similarity(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[float]:
        if self._anchor_hist is None and self._recent_hist is None and self._long_hist is None:
            return None
        curr_hist = self._compute_bbox_hist(frame_bgr, bbox)
        if curr_hist is None:
            return None
        sim_anchor = None
        sim_recent = None
        if self._anchor_hist is not None:
            sim_anchor = self._hist_similarity(self._anchor_hist, curr_hist)
        if self._recent_hist is not None:
            sim_recent = self._hist_similarity(self._recent_hist, curr_hist)
        sim_long = None
        if self._long_hist is not None:
            sim_long = self._hist_similarity(self._long_hist, curr_hist)
        if sim_anchor is None and sim_recent is None:
            return sim_long
        if sim_anchor is None:
            return max(sim_recent, sim_long) if sim_long is not None else sim_recent
        if sim_recent is None:
            return max(sim_anchor, sim_long) if sim_long is not None else sim_anchor
        if sim_long is None:
            return max(sim_anchor, sim_recent)
        return max(sim_anchor, sim_recent, sim_long)

    def _compute_anchor_similarity(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[float]:
        if self._anchor_hist is None:
            return None
        curr_hist = self._compute_bbox_hist(frame_bgr, bbox)
        if curr_hist is None:
            return None
        return self._hist_similarity(self._anchor_hist, curr_hist)

    def consume_appearance_update_flag(self) -> bool:
        if self._appearance_updated:
            self._appearance_updated = False
            return True
        return False

    def is_verifying(self) -> bool:
        return self._verification_active

    def get_search_hint_bbox(self) -> Optional[BBox]:
        return self._search_hint_bbox

    def _maybe_update_appearance(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox,
        score: Optional[float],
        prev_bbox: Optional[BBox],
    ) -> None:
        if self.appearance_update_interval_frames > 0:
            if (self._frame_count - self._appearance_last_update_frame) < self.appearance_update_interval_frames:
                return

        if score is None or score < self.appearance_update_min_score:
            self._trusted_frames = 0
            return

        if prev_bbox is not None:
            motion_ratio = self._center_motion_ratio(prev_bbox, bbox)
            if motion_ratio > self.appearance_update_max_motion_ratio:
                self._trusted_frames = 0
                return

        if self.appearance_update_min_sharpness > 0.0:
            sharpness = self._compute_roi_sharpness(frame_bgr, bbox)
            if sharpness is None or sharpness < self.appearance_update_min_sharpness:
                self._trusted_frames = 0
                return

        curr_hist = self._compute_bbox_hist(frame_bgr, bbox)
        if curr_hist is None:
            self._trusted_frames = 0
            return

        if self._anchor_hist is not None:
            anchor_similarity = self._hist_similarity(self._anchor_hist, curr_hist)
            if anchor_similarity < self.anchor_min_similarity:
                self._trusted_frames = 0
                return

        if self._recent_hist is not None and self.appearance_update_min_similarity > 0.0:
            recent_similarity = self._hist_similarity(self._recent_hist, curr_hist)
            if recent_similarity < self.appearance_update_min_similarity:
                self._trusted_frames = 0
                return

        self._trusted_frames += 1
        if self._trusted_frames < self.appearance_update_trust_frames:
            return

        if self._recent_hist is None:
            self._recent_hist = curr_hist
        else:
            alpha = self.appearance_update_alpha
            self._recent_hist = (1.0 - alpha) * self._recent_hist + alpha * curr_hist
            self._recent_hist = cv2.normalize(self._recent_hist, self._recent_hist).flatten()

        self._recent_bbox = bbox
        self._appearance_last_update_frame = self._frame_count
        self._appearance_updated = True
        self._trusted_frames = 0

        if self.update_backend_template_on_appearance:
            self._maybe_update_backend_template(frame_bgr, bbox)

    def _maybe_update_long_appearance(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox,
        score: Optional[float],
        prev_bbox: Optional[BBox],
    ) -> None:
        if self.long_update_interval_frames > 0:
            if (self._frame_count - self._long_last_update_frame) < self.long_update_interval_frames:
                return

        if score is None or score < self.long_update_min_score:
            return

        if prev_bbox is not None:
            motion_ratio = self._center_motion_ratio(prev_bbox, bbox)
            if motion_ratio > self.appearance_update_max_motion_ratio:
                return

        if self.appearance_update_min_sharpness > 0.0:
            sharpness = self._compute_roi_sharpness(frame_bgr, bbox)
            if sharpness is None or sharpness < self.appearance_update_min_sharpness:
                return

        curr_hist = self._compute_bbox_hist(frame_bgr, bbox)
        if curr_hist is None:
            return

        if self._anchor_hist is not None:
            anchor_similarity = self._hist_similarity(self._anchor_hist, curr_hist)
            if anchor_similarity < self.anchor_min_similarity:
                return

        if self._long_hist is not None and self.long_update_min_similarity > 0.0:
            long_similarity = self._hist_similarity(self._long_hist, curr_hist)
            if long_similarity < self.long_update_min_similarity:
                return

        if self._long_hist is None:
            self._long_hist = curr_hist
        else:
            alpha = self._score_weighted_long_alpha(float(score))
            if alpha <= 0.0:
                return
            self._long_hist = (1.0 - alpha) * self._long_hist + alpha * curr_hist
            self._long_hist = cv2.normalize(self._long_hist, self._long_hist).flatten()

        self._long_bbox = bbox
        self._long_last_update_frame = self._frame_count
        self._appearance_updated = True

    def _maybe_update_backend_template(self, frame_bgr: np.ndarray, bbox: BBox) -> None:
        if self._backend is None:
            return
        if not hasattr(self._backend, "update_template"):
            return
        try:
            self._backend.update_template(frame_bgr, bbox.to_xywh())
        except Exception as e:
            if self.verbose:
                print(f"[OSTrackTracker] warning: update_template failed: {e}")

    @staticmethod
    def _center_motion_ratio(prev_bbox: BBox, new_bbox: BBox) -> float:
        prev_cx = 0.5 * (prev_bbox.x1 + prev_bbox.x2)
        prev_cy = 0.5 * (prev_bbox.y1 + prev_bbox.y2)
        new_cx = 0.5 * (new_bbox.x1 + new_bbox.x2)
        new_cy = 0.5 * (new_bbox.y1 + new_bbox.y2)
        center_dist = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))
        prev_diag = float(np.hypot(prev_bbox.w, prev_bbox.h))
        return center_dist / max(prev_diag, 1.0)

    def _score_weighted_long_alpha(self, score: float) -> float:
        if self.long_update_alpha_base <= 0.0:
            return 0.0
        if score <= self.long_update_min_score:
            return 0.0
        denom = 1.0 - self.long_update_min_score
        if denom <= 1e-6:
            return 0.0
        weight = (score - self.long_update_min_score) / denom
        weight = min(max(weight, 0.0), 1.0)
        return self.long_update_alpha_base * weight

    @staticmethod
    def _compute_roi_sharpness(frame_bgr: np.ndarray, bbox: BBox) -> Optional[float]:
        h, w = frame_bgr.shape[:2]
        b = bbox.clip(w, h)
        x1, y1 = int(b.x1), int(b.y1)
        x2, y2 = int(b.x2), int(b.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _hist_similarity(ref_hist: np.ndarray, curr_hist: np.ndarray) -> float:
        dist = cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        return float(max(0.0, 1.0 - dist))

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        xx1 = max(a.x1, b.x1)
        yy1 = max(a.y1, b.y1)
        xx2 = min(a.x2, b.x2)
        yy2 = min(a.y2, b.y2)
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        union = a.area + b.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

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
