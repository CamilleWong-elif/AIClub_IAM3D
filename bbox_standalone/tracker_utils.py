"""
Tracking utilities for multi-object tracking (MOT).
Uses IoU-based cost matrix and linear sum assignment for track–detection association.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Union, Any


def _box_coords(box: Any) -> Tuple[float, float, float, float]:
    """Extract (x1, y1, x2, y2) from a box (tuple, dict, or object with x1, y1, x2, y2)."""
    if hasattr(box, "x1") and hasattr(box, "x2"):
        return (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
    if isinstance(box, dict) and "x1" in box and "x2" in box:
        return (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    raise TypeError("box must have x1,y1,x2,y2 attributes or be a 4+ element sequence or dict")


def bbox_iou(box1: Union[Tuple[float, ...], Any], box2: Union[Tuple[float, ...], Any]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Boxes can be (x1, y1, x2, y2) tuples or any object with .x1, .y1, .x2, .y2.
    Returns a value in [0, 1]; 0 means no overlap, 1 means identical boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU score in [0, 1].
    """
    x1_1, y1_1, x2_1, y2_1 = _box_coords(box1)
    x1_2, y1_2, x2_2, y2_2 = _box_coords(box2)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area_1 + area_2 - inter_area

    return inter_area / union if union > 0 else 0.0


def cost_matrix_iou(
    tracks: List[Any],
    detections: List[Any],
) -> np.ndarray:
    """
    Build a cost matrix between existing tracks and new detections.

    Cost[i, j] = 1 - IoU(tracks[i], detections[j]). Minimizing total cost
    corresponds to maximizing total IoU (best overlap). Empty dimensions
    are handled (zero rows or columns).

    Args:
        tracks: List of track boxes (each with x1, y1, x2, y2 or (x1,y1,x2,y2)).
        detections: List of detection boxes in the same format.

    Returns:
        2D numpy array of shape (len(tracks), len(detections)) with values in [0, 1].
    """
    n_t = len(tracks)
    n_d = len(detections)
    if n_t == 0 or n_d == 0:
        return np.zeros((n_t, n_d), dtype=np.float64)

    cost = np.ones((n_t, n_d), dtype=np.float64)
    for i, tr in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = bbox_iou(tr, det)
            cost[i, j] = 1.0 - iou
    return cost


def track_association(
    tracks: List[Any],
    detections: List[Any],
    iou_th: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate tracks to detections by minimizing 1-IoU cost (maximizing IoU).
    Only pairs with IoU >= iou_th are considered valid matches.

    Args:
        tracks: List of current track boxes.
        detections: List of detection boxes for the current frame.
        iou_th: Minimum IoU to accept a match. Pairs below this are treated as unmatched.

    Returns:
        matches: List of (track_index, detection_index) for assigned pairs.
        unmatched_track_indices: Indices of tracks with no detection.
        unmatched_detection_indices: Indices of detections that started a new track.
    """
    n_t = len(tracks)
    n_d = len(detections)

    if n_t == 0:
        return [], [], list(range(n_d))
    if n_d == 0:
        return [], list(range(n_t)), []

    cost = cost_matrix_iou(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_track_set = set()
    matched_det_set = set()
    matches = []

    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, c]
        if iou >= iou_th:
            matches.append((r, c))
            matched_track_set.add(r)
            matched_det_set.add(c)

    unmatched_tracks = sorted(set(range(n_t)) - matched_track_set)
    unmatched_dets = sorted(set(range(n_d)) - matched_det_set)
    return matches, unmatched_tracks, unmatched_dets
