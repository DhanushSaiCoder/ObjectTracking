from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import inspect

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

    def clipped(self, width: int, height: int) -> "BBox":
        max_x = width - 1
        max_y = height - 1
        x1 = float(0 if self.x1 < 0 else max_x if self.x1 > max_x else self.x1)
        y1 = float(0 if self.y1 < 0 else max_y if self.y1 > max_y else self.y1)
        x2 = float(0 if self.x2 < 0 else max_x if self.x2 > max_x else self.x2)
        y2 = float(0 if self.y2 < 0 else max_y if self.y2 > max_y else self.y2)
        return BBox(x1, y1, x2, y2)


@dataclass
class Detection:
    bbox: BBox
    score: float
    class_id: int
    class_name: str


class RTDETRDetector:
    """
    Optimized RT-DETR detector wrapper for:
    - person-only detection
    - low Python overhead
    - clean output objects

    Key optimizations:
    - cache model names once
    - avoid per-frame try/except for `half`
    - move box data from torch->cpu only once
    - no per-frame sort unless explicitly enabled
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.60,
        iou: float = 0.7,
        classes: Optional[Sequence[int]] = (0,),   # person only
        max_det: int = 100,
        half: bool = True,
        verbose: bool = False,
        sort_by_score: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = tuple(classes) if classes is not None else None
        self.max_det = int(max_det)
        self.half = bool(half)
        self.verbose = bool(verbose)
        self.sort_by_score = bool(sort_by_score)

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"RTDETR model file not found: {model_file}")

        try:
            from ultralytics import RTDETR  # type: ignore
        except Exception as e:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from e

        self.model = RTDETR(self.model_path)

        # cache names once
        names = getattr(self.model, "names", None)
        self._names = names if isinstance(names, dict) else {}

        # determine once whether predict(...) supports `half`
        try:
            self._supports_half = "half" in inspect.signature(self.model.predict).parameters
        except Exception:
            self._supports_half = True

        # base kwargs reused every frame
        self._predict_kwargs = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
            "max_det": self.max_det,
            "device": self.device,
            "verbose": False,
        }

    def warmup(self, shape: tuple[int, int, int] = (640, 640, 3)) -> None:
        dummy = np.zeros(shape, dtype=np.uint8)
        _ = self.detect(dummy)

    def set_conf(self, conf: float) -> None:
        self.conf = float(conf)
        self._predict_kwargs["conf"] = self.conf

    def set_iou(self, iou: float) -> None:
        self.iou = float(iou)
        self._predict_kwargs["iou"] = self.iou

    def set_classes(self, classes: Optional[Sequence[int]]) -> None:
        self.classes = tuple(classes) if classes is not None else None
        self._predict_kwargs["classes"] = self.classes

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        kwargs = self._predict_kwargs.copy()
        kwargs["source"] = frame_bgr

        if self._supports_half:
            kwargs["half"] = self.half

        results = self.model.predict(**kwargs)
        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return []

        # Fast path: move all box info in one go.
        # For Ultralytics detect output, boxes.data is typically Nx6:
        # [x1, y1, x2, y2, conf, cls]
        data = getattr(boxes, "data", None)
        if data is None:
            return []

        arr = self._to_numpy(data)
        if arr is None or len(arr) == 0:
            return []

        names = self._names
        if not names:
            result_names = getattr(res, "names", None)
            names = result_names if isinstance(result_names, dict) else {}

        detections: List[Detection] = []
        append_det = detections.append

        for row in arr:
            if len(row) < 6:
                continue

            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])
            score = float(row[4])
            cid = int(row[5])

            bbox = BBox(x1, y1, x2, y2).clipped(w, h)

            if bbox.w <= 1.0 or bbox.h <= 1.0:
                continue

            append_det(
                Detection(
                    bbox=bbox,
                    score=score,
                    class_id=cid,
                    class_name=names.get(cid, str(cid)),
                )
            )

        if self.sort_by_score and len(detections) > 1:
            detections.sort(key=lambda d: d.score, reverse=True)

        return detections

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)