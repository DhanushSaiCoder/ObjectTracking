import cv2
import os
from pathlib import Path

def frames_to_video(
    frames_folder,
    output_path,
    fps=30,
    codec='mp4v'
):
    frames_folder = Path(frames_folder)

    # Get sorted image list
    images = sorted([
        f for f in frames_folder.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if not images:
        raise ValueError("No images found in folder")

    # Read first image to get size
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    for img_path in images:
        frame = cv2.imread(str(img_path))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at: {output_path}")


# Usage
frames_to_video(
    frames_folder="./truck",
    output_path="../assets/truck.mp4",
    fps=10
)