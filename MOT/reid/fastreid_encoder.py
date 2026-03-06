import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_fastreid_to_path() -> Path:
    fastreid_root = _repo_root() / "third_party" / "bot_sort" / "fast_reid"
    if not fastreid_root.exists():
        raise FileNotFoundError(
            f"FastReID root not found at {fastreid_root}. Did you init the BoT-SORT submodule?"
        )
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(fastreid_root) not in sys.path:
        sys.path.insert(0, str(fastreid_root))
    return fastreid_root


def _resolve_path(path_value: str, default_path: Path) -> Path:
    if not path_value:
        return default_path.resolve()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


class FastReIDEncoder:
    """FastReID encoder compatible with Ultralytics BoTSORT."""

    def __init__(self, config_path: str, weights_path: str, device: str, batch_size: int = 32):
        _add_fastreid_to_path()

        from fast_reid.fastreid.config import get_cfg  # type: ignore
        from fast_reid.fastreid.modeling import build_model  # type: ignore
        from fast_reid.fastreid.utils.checkpoint import Checkpointer  # type: ignore

        self.config_path = _resolve_path(
            config_path,
            _repo_root() / "third_party" / "bot_sort" / "fast_reid" / "configs" / "MOT17" / "sbs_S50.yml",
        )
        self.weights_path = _resolve_path(
            weights_path,
            _repo_root() / "weights" / "reid" / "mot17_sbs_S50.pth",
        )

        if not self.config_path.exists():
            raise FileNotFoundError(f"FastReID config not found: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"FastReID weights not found: {self.weights_path}")

        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.freeze()
        self.cfg = cfg

        self.model = build_model(cfg)
        self.model.to(self.device)
        self.model.eval()

        Checkpointer(self.model).load(str(self.weights_path))

        size_test = cfg.INPUT.SIZE_TEST
        self.size_test = (int(size_test[0]), int(size_test[1]))
        try:
            pixel_mean = cfg.INPUT.PIXEL_MEAN
            pixel_std = cfg.INPUT.PIXEL_STD
        except Exception:
            pixel_mean = cfg.MODEL.PIXEL_MEAN
            pixel_std = cfg.MODEL.PIXEL_STD

        self.pixel_mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.array(pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self.margin = 0.10
        self._logged_first = False

        print(
            f"[FastReID] Initialized encoder | config={self.config_path} | weights={self.weights_path} | device={self.device}"
        )

    def __call__(self, img_bgr: np.ndarray, dets_xywh_idx: np.ndarray) -> List[np.ndarray]:
        if dets_xywh_idx is None or len(dets_xywh_idx) == 0:
            return []

        dets = np.asarray(dets_xywh_idx, dtype=np.float32)
        boxes = dets[:, :4]
        img_h, img_w = img_bgr.shape[:2]

        crops = []
        for cx, cy, w, h in boxes:
            w = w * (1.0 + 2.0 * self.margin)
            h = h * (1.0 + 2.0 * self.margin)
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            x1 = int(max(0, np.floor(x1)))
            y1 = int(max(0, np.floor(y1)))
            x2 = int(min(img_w - 1, np.ceil(x2)))
            y2 = int(min(img_h - 1, np.ceil(y2)))

            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((1, 1, 3), dtype=img_bgr.dtype)
            else:
                crop = img_bgr[y1:y2, x1:x2]
            crops.append(crop)

        embeddings = []
        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start : start + self.batch_size]
            batch = np.stack([self._preprocess(c) for c in batch_crops], axis=0)
            batch_tensor = torch.from_numpy(batch).to(self.device)

            with torch.no_grad():
                feats = self.model(batch_tensor)
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                feats = F.normalize(feats, dim=1)

            if not self._logged_first:
                mean_norm = float(torch.linalg.norm(feats, dim=1).mean().item())
                print(
                    f"[FastReID] First batch | dim={feats.shape[1]} | mean_l2={mean_norm:.3f} | batch={feats.shape[0]}"
                )
                self._logged_first = True

            embeddings.append(feats.cpu().numpy())

        all_feats = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
        return [all_feats[i] for i in range(all_feats.shape[0])]

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = self._resize_with_pad(crop_rgb, self.size_test)
        resized = resized.astype(np.float32)
        resized = (resized - self.pixel_mean) / self.pixel_std
        resized = np.transpose(resized, (2, 0, 1))
        return resized

    @staticmethod
    def _resize_with_pad(img: np.ndarray, size_hw: tuple) -> np.ndarray:
        target_h, target_w = size_hw
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=img.dtype)
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas
