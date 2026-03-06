# Multi-Object Tracking (MOT) with BoTSORT

This module provides a multi-object tracking application using the BoTSORT algorithm and RT-DETR detector.

## Features
- **BoTSORT Tracking**: Robust multi-object tracking using Kalman Filters and Camera Motion Compensation.
- **ROI Selection**: Manually select objects to track by drawing boxes or clicking on them.
- **Interactive Interface**: Toggle between live tracking and paused selection mode.

## Usage

Run the application:
```bash
python3 MOT/app.py
```

Optional: track specific classes (comma-separated class IDs):
```bash
python3 MOT/app.py --classes 0,2,3
```

### FastReID Integration (BoT-SORT)
FastReID is injected at runtime via a patch that replaces Ultralytics' BoTSORT encoder. This keeps `model.track()` and GMC intact.

Environment overrides:
- `MOT_FASTREID_CONFIG` (default: `third_party/bot_sort/fast_reid/configs/MOT17/sbs_S50.yml`)
- `MOT_FASTREID_WEIGHTS` (default: `weights/reid/mot17_sbs_S50.pth`)
- `MOT_FASTREID_DEVICE` (default: `cuda` if available, else `cpu`)
- `MOT_FASTREID_BATCH` (default: `32`)

Initialize the BoT-SORT submodule if needed:
```bash
git submodule update --init --recursive
```

Note: MOT17-SBS-S50 is a person-focused ReID model; expect best results on pedestrian data.

### Controls
- **P**: Pause/Resume the video.
- **Mouse Click**: Toggle activation of a track (click on a bounding box).
- **Mouse Drag (Paused)**: Draw a Region of Interest (ROI) to activate the nearest track.
- **R**: Reset/Clear all active tracks.
- **Q**: Quit the application.

## Implementation Details
- **Detector**: RT-DETR (Real-Time DEtection TRansformer) via `ultralytics`.
- **Tracker**: BoTSORT (Bolstered Tracking and SORT) integrated with the detector.
- **Filtering**: The application maintains a set of "activated" track IDs. In live mode, only these IDs are displayed prominently.
