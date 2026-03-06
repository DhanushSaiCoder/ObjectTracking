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
