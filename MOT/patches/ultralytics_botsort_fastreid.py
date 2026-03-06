import os
from pathlib import Path

import torch

from MOT.reid.fastreid_encoder import FastReIDEncoder

_PATCHED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _version_ge(current: str, minimum: str) -> bool:
    try:
        from packaging.version import Version  # type: ignore

        return Version(current) >= Version(minimum)
    except Exception:
        def _parts(v: str) -> tuple:
            items = []
            for part in v.split("."):
                digits = "".join(ch for ch in part if ch.isdigit())
                if digits:
                    items.append(int(digits))
            return tuple(items)

        return _parts(current) >= _parts(minimum)


def apply_botsort_fastreid_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import ultralytics
    from ultralytics.trackers import track as track_module
    from ultralytics.trackers.bot_sort import BOTSORT

    if not _version_ge(ultralytics.__version__, "8.3.114"):
        raise RuntimeError(
            f"Ultralytics {ultralytics.__version__} is too old. Require >= 8.3.114 for BoTSORT ReID plumbing."
        )

    repo_root = _repo_root()
    default_config = repo_root / "third_party" / "bot_sort" / "fast_reid" / "configs" / "MOT17" / "sbs_S50.yml"
    default_weights = repo_root / "weights" / "reid" / "mot17_sbs_S50.pth"

    config_path = os.environ.get("MOT_FASTREID_CONFIG", str(default_config))
    weights_path = os.environ.get("MOT_FASTREID_WEIGHTS", str(default_weights))
    device = os.environ.get("MOT_FASTREID_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("MOT_FASTREID_BATCH", "32"))

    if not Path(config_path).expanduser().exists():
        raise FileNotFoundError(f"FastReID config not found: {config_path}")
    if not Path(weights_path).expanduser().exists():
        raise FileNotFoundError(f"FastReID weights not found: {weights_path}")

    class BOTSORTFastReID(BOTSORT):
        def __init__(self, args, frame_rate=30):
            super().__init__(args, frame_rate)
            self.args.with_reid = True
            self.with_reid = True
            self.encoder = FastReIDEncoder(
                config_path=config_path,
                weights_path=weights_path,
                device=device,
                batch_size=batch_size,
            )

        def update(self, results, img=None, feats=None):
            return super().update(results, img, feats=None)

    track_module.TRACKER_MAP["botsort"] = BOTSORTFastReID
    _PATCHED = True
    print("[FastReID] Patched Ultralytics TRACKER_MAP['botsort'] with FastReID encoder")
