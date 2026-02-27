import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
)

from rtdetr_detector import RTDETRDetector
from ostrack_tracker import OSTrackTracker, BBox as TrkBBox
from utils_selection import pick_detection_by_click


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class ClickableLabel(QLabel):
    clicked = Signal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)


class Worker(QObject):
    frame_ready = Signal(QImage)
    status_ready = Signal(str)
    fps_ready = Signal(float)

    def __init__(self):
        super().__init__()

        self.repo_root = Path(__file__).resolve().parents[1]

        self.detector = RTDETRDetector(
            model_path="rtdetr-l.pt",
            device="cuda:0",
            imgsz=640,
            conf=0.25,
            iou=0.7,
            classes=(0,),
            max_det=50,
            half=True,
            verbose=False,
        )

        self.tracker = OSTrackTracker(
            repo_root=str(self.repo_root),
            tracker_name="ostrack",
            param_name="vitb_384_mae_ce_32x4_ep300",
            dataset_name="demo",
            verbose=False,
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.mode = "SELECT"

        self.last_frame: Optional[np.ndarray] = None
        self.last_detections = []

        self.width = 640
        self.height = 480

    @Slot()
    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            self.status_ready.emit("ERROR: Could not open camera")
            return

        self.running = True
        self.status_ready.emit("Mode: SELECT | Click a person bbox to start tracking")

        prev_t = time.perf_counter()
        fps_smooth = 0.0

        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.status_ready.emit("ERROR: Camera read failed")
                break

            self.last_frame = frame.copy()
            draw = frame.copy()

            now_t = time.perf_counter()
            dt = max(now_t - prev_t, 1e-6)
            prev_t = now_t
            fps = 1.0 / dt
            fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)

            if self.mode == "SELECT":
                detections = self.detector.detect(frame)
                self.last_detections = detections

                for det in detections:
                    b = det.bbox
                    x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        draw,
                        f"{det.class_name} {det.score:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    draw,
                    "CLICK a bbox to start OSTrack",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            elif self.mode == "TRACK":
                result = self.tracker.update(frame)
                if result.ok and result.bbox is not None:
                    b = result.bbox
                    x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        draw,
                        "TRACKING",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    self.tracker.reset()
                    self.mode = "SELECT"
                    self.status_ready.emit("Mode: SELECT | Target LOST. Click again.")

            cv2.putText(
                draw,
                f"FPS: {fps_smooth:.1f}",
                (10, self.height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            self.frame_ready.emit(bgr_to_qimage(draw))
            self.fps_ready.emit(float(fps_smooth))

        if self.cap is not None:
            self.cap.release()
        self.running = False

    @Slot()
    def stop(self):
        self.running = False

    @Slot()
    def reset(self):
        self.tracker.reset()
        self.mode = "SELECT"
        self.status_ready.emit("Mode: SELECT | Click a person bbox to start tracking")

    @Slot(int, int)
    def on_click(self, x: int, y: int):
        if self.mode != "SELECT":
            return

        if self.last_frame is None or not self.last_detections:
            self.status_ready.emit("No detections available yet")
            return

        selected = pick_detection_by_click(self.last_detections, x, y)
        if selected is None:
            self.status_ready.emit("Click inside a person bbox")
            return

        b = selected.bbox
        ok = self.tracker.initialize(
            self.last_frame,
            TrkBBox(b.x1, b.y1, b.x2, b.y2)
        )

        if ok:
            self.mode = "TRACK"
            self.status_ready.emit("Mode: TRACK | OSTrack initialized")
        else:
            self.status_ready.emit("OSTrack init failed")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTDETR + OSTrack Demo")

        self.video = ClickableLabel()
        self.video.setFixedSize(640, 480)
        self.video.setStyleSheet("background-color: black;")

        self.status = QLabel("Starting...")
        self.fps = QLabel("FPS: --")
        self.reset_btn = QPushButton("Reset")
        self.quit_btn = QPushButton("Quit")

        right = QVBoxLayout()
        right.addWidget(self.status)
        right.addWidget(self.fps)
        right.addStretch()
        right.addWidget(self.reset_btn)
        right.addWidget(self.quit_btn)

        layout = QHBoxLayout(self)
        layout.addWidget(self.video)
        layout.addLayout(right)

        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_ready.connect(self.status.setText)
        self.worker.fps_ready.connect(lambda f: self.fps.setText(f"FPS: {f:.1f}"))

        self.video.clicked.connect(self.worker.on_click)
        self.reset_btn.clicked.connect(self.worker.reset)
        self.quit_btn.clicked.connect(self.close)

        self.thread.start()

    @Slot(QImage)
    def update_frame(self, image: QImage):
        self.video.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())