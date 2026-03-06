import cv2
import time
from pathlib import Path
from typing import List, Set, Tuple, Optional
from ultralytics import RTDETR
import numpy as np

# Styling constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (80, 255, 160), (200, 80, 255)
]
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
    
    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: 'BBox'):
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0: return 0.0
        
        union = self.area + other.area - inter
        return inter / union

class MOTApp:
    def __init__(self, model_path: str, source: str):
        self.model = RTDETR(model_path)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open source {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.paused = False
        self.activated_ids: Set[int] = set()
        self.current_tracks = [] # List of (id, bbox, cls, conf)
        self.last_frame = None
        
        self.is_drawing = False
        self.drag_start = None
        self.drag_current = None
        self.pending_rois = []

        self.window_name = "BoTSORT Multi-Object Tracker"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if not self.paused:
            # In live mode, allow single click to toggle ID if shown
            if event == cv2.EVENT_LBUTTONDOWN:
                self.match_and_activate(x, y)
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
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            bx1, bx2 = sorted([x1, x2])
            by1, by2 = sorted([y1, y2])
            if (bx2 - bx1) > 5 and (by2 - by1) > 5:
                roi = BBox(bx1, by1, bx2, by2)
                self.match_and_activate_roi(roi)
            self.drag_start = None
            self.drag_current = None

    def match_and_activate(self, x, y):
        best_id = -1
        min_area = float('inf')
        for tid, bbox, cls, conf in self.current_tracks:
            if bbox.contains(x, y):
                if bbox.area < min_area:
                    min_area = bbox.area
                    best_id = tid
        
        if best_id != -1:
            if best_id in self.activated_ids:
                self.activated_ids.remove(best_id)
                print(f"Deactivated ID {best_id}")
            else:
                self.activated_ids.add(best_id)
                print(f"Activated ID {best_id}")

    def match_and_activate_roi(self, roi: BBox):
        best_id = -1
        max_iou = 0.1
        for tid, bbox, cls, conf in self.current_tracks:
            iou = roi.iou(bbox)
            if iou > max_iou:
                max_iou = iou
                best_id = tid
        
        if best_id != -1:
            self.activated_ids.add(best_id)
            print(f"Activated ID {best_id} via ROI (IoU={max_iou:.2f})")
        else:
            print("No matching track found for ROI")

    def run(self):
        print("Controls:")
        print("  P: pause/resume")
        print("  Click: toggle track activation")
        print("  Mouse drag (paused): draw ROI to activate track")
        print("  R: clear all activations")
        print("  Q: quit")

        prev_time = time.perf_counter()
        
        while True:
            if not self.paused:
                ok, frame = self.cap.read()
                if not ok: break
                self.last_frame = frame.copy()
                
                # Run BoTSORT tracking with custom configuration
                results = self.model.track(frame, persist=True, tracker="./custom_botsort.yaml", verbose=False, classes=[4])
                
                self.current_tracks = []
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    ids = boxes.id.cpu().numpy().astype(int)
                    coords = boxes.xyxy.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    
                    for i in range(len(ids)):
                        self.current_tracks.append((
                            ids[i],
                            BBox(coords[i][0], coords[i][1], coords[i][2], coords[i][3]),
                            clss[i],
                            confs[i]
                        ))
            else:
                frame = self.last_frame.copy()

            # Draw
            now = time.perf_counter()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            
            # Display status
            mode_str = "PAUSED" if self.paused else "LIVE"
            cv2.putText(frame, f"{mode_str} | Active IDs: {list(self.activated_ids)} | FPS: {fps:.1f}", (10, 30), FONT, 0.7, WHITE, 2)

            for tid, bbox, cls, conf in self.current_tracks:
                is_active = tid in self.activated_ids
                
                # In paused mode, show all tracks for selection
                # In live mode, only show active tracks (unless none active, then show all?)
                # Let's say: show all if paused, show only active if live.
                if self.paused or is_active:
                    color = COLORS[tid % len(COLORS)] if is_active else (128, 128, 128)
                    thickness = 3 if is_active else 1
                    
                    cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, thickness)
                    label = f"ID {tid}" + (f" {conf:.2f}" if is_active else "")
                    cv2.putText(frame, label, (int(bbox.x1), int(bbox.y1) - 10), FONT, 0.6, color, 2)

            if self.is_drawing and self.drag_start and self.drag_current:
                cv2.rectangle(frame, self.drag_start, self.drag_current, YELLOW, 2)

            cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('r'):
                self.activated_ids.clear()
                print("Cleared all activations")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use a default model and video if available
    repo_root = Path(__file__).resolve().parents[1]
    model_path = str(repo_root / "OSTrack_demo/OSTrack/rtdetr-l.pt")
    video_path = str(repo_root / "OSTrack_demo/OSTrack/demo_app/assets/air_show.mp4")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
    if not Path(video_path).exists():
        print(f"ERROR: Video not found at {video_path}")
    
    # Ensure display is available for cv2.imshow
    import os
    if "DISPLAY" not in os.environ:
        print("WARNING: No DISPLAY found. Application will not be able to show windows.")
        # In a real scenario, we might want to save output to a file instead
    
    app = MOTApp(model_path, video_path)
    app.run()
