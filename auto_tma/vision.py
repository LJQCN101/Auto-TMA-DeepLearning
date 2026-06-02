from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

LineSegment = tuple[int, int, int, int]


def detect_candidate_lines(
    image: np.ndarray,
    *,
    bilateral_diameter: int = 1,
    bilateral_sigma_color: float = 80.0,
    bilateral_sigma_space: float = 11.0,
    canny_threshold1: float = 300.0,
    canny_threshold2: float = 350.0,
    hough_rho: float = 1.0,
    hough_theta: float = math.pi / 180.0,
    hough_threshold: int = 100,
    min_line_length: float = 100.0,
    max_line_gap: float = 50.0,
) -> list[LineSegment]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    filtered = cv2.bilateralFilter(gray, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    detected = cv2.HoughLinesP(
        edges,
        rho=hough_rho,
        theta=hough_theta,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if detected is None:
        return []

    del filtered
    return [tuple(int(value) for value in line[0]) for line in detected]


def reduce_lines(lines: Sequence[LineSegment]) -> list[LineSegment]:
    if not lines:
        return []

    clusters = _cluster_lines(lines)
    reduced: list[LineSegment] = []
    for cluster in clusters:
        points = np.asarray(
            [[line[0], line[1]] for line in cluster] + [[line[2], line[3]] for line in cluster],
            dtype=np.float32,
        )
        reduced.append(_fit_line_segment(points))
    return reduced


def _cluster_lines(lines: Sequence[LineSegment]) -> list[list[LineSegment]]:
    parent = list(range(len(lines)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            if equivalent_lines(lines[left], lines[right]):
                union(left, right)

    grouped: dict[int, list[LineSegment]] = {}
    for index, line in enumerate(lines):
        grouped.setdefault(find(index), []).append(line)
    return list(grouped.values())


def equivalent_lines(
    left: LineSegment,
    right: LineSegment,
    *,
    bearing_tolerance_deg: float = 1.5,
    rho_tolerance_px: float = 5.0,
) -> bool:
    left_bearing = _line_bearing_deg(left)
    right_bearing = _line_bearing_deg(right)
    if abs(left_bearing - right_bearing) > bearing_tolerance_deg:
        return False

    left_rho = _line_rho(left)
    right_rho = _line_rho(right)
    if not math.isfinite(left_rho) or not math.isfinite(right_rho):
        return False
    return abs(left_rho - right_rho) <= rho_tolerance_px


def _line_bearing_deg(line: LineSegment) -> float:
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(x1 - x2, y1 - y2))


def _line_rho(line: LineSegment) -> float:
    x1, y1, x2, y2 = line
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0.0:
        return math.nan
    return ((x1 * y2) - (x2 * y1)) / length


def _fit_line_segment(points: np.ndarray) -> LineSegment:
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.asarray([vx, vy], dtype=float)
    origin = np.asarray([x0, y0], dtype=float)
    projections = (points - origin) @ direction
    min_projection = float(np.min(projections))
    max_projection = float(np.max(projections))
    start = origin + (direction * min_projection)
    end = origin + (direction * max_projection)
    return (
        int(round(start[0])),
        int(round(start[1])),
        int(round(end[0])),
        int(round(end[1])),
    )
