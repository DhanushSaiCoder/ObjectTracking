from __future__ import annotations


def pick_detection_by_click(detections, x: float, y: float):
    """
    detections: list of objects with .bbox and .score
    bbox must have x1, y1, x2, y2, area
    """

    hits = []
    for det in detections:
        b = det.bbox
        if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
            hits.append(det)

    if not hits:
        return None

    # Prefer smallest area box first (better when boxes overlap),
    # then highest confidence.
    hits.sort(key=lambda d: (d.bbox.area, -float(getattr(d, "score", 0.0))))
    return hits[0]