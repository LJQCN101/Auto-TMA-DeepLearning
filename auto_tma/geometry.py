from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .models import BearingMeasurement

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def normalize_course_deg(course_deg: float) -> float:
    return course_deg % 360.0


def signed_angle_delta_deg(left_deg: float, right_deg: float) -> float:
    return (left_deg - right_deg + 180.0) % 360.0 - 180.0


def angle_delta_deg(left_deg: float, right_deg: float) -> float:
    return abs(signed_angle_delta_deg(left_deg, right_deg))


def velocity_components(speed: float, course_deg: float) -> tuple[float, float]:
    course_rad = course_deg * DEG_TO_RAD
    return speed * math.sin(course_rad), speed * math.cos(course_rad)


def speed_course_from_velocity(velocity_x: float, velocity_y: float) -> tuple[float, float]:
    speed = math.hypot(velocity_x, velocity_y)
    course_deg = normalize_course_deg(math.degrees(math.atan2(velocity_x, velocity_y)))
    return speed, course_deg


def range_between_points(origin_x: float, origin_y: float, target_x: float, target_y: float) -> float:
    return math.hypot(target_x - origin_x, target_y - origin_y)


def bearing_between_points(origin_x: float, origin_y: float, target_x: float, target_y: float) -> float:
    return normalize_course_deg(math.degrees(math.atan2(target_x - origin_x, target_y - origin_y)))


def bearing_from_state(state: Sequence[float], ownship_x: float, ownship_y: float) -> float:
    return bearing_between_points(ownship_x, ownship_y, float(state[0]), float(state[1]))


def bearing_measurement_jacobian(target_x: float, target_y: float, ownship_x: float, ownship_y: float) -> np.ndarray:
    delta_x = target_x - ownship_x
    delta_y = target_y - ownship_y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 1e-12:
        raise ValueError("bearing jacobian is undefined when ownship and target positions coincide")
    scale = RAD_TO_DEG / denominator
    return np.asarray([[scale * delta_y, -scale * delta_x, 0.0, 0.0]], dtype=float)


def target_start_from_initial_range(
    reference: BearingMeasurement,
    initial_range: float,
) -> tuple[float, float]:
    bearing_rad = reference.bearing_deg * DEG_TO_RAD
    return (
        reference.ownship_x + initial_range * math.sin(bearing_rad),
        reference.ownship_y + initial_range * math.cos(bearing_rad),
    )


def target_position_at_time(
    reference: BearingMeasurement,
    initial_range: float,
    speed: float,
    course_deg: float,
    time_seconds: float,
) -> tuple[float, float]:
    start_x, start_y = target_start_from_initial_range(reference, initial_range)
    velocity_x, velocity_y = velocity_components(speed, course_deg)
    elapsed = time_seconds - reference.time_seconds
    return start_x + velocity_x * elapsed, start_y + velocity_y * elapsed


def prepare_measurements(measurements: Iterable[BearingMeasurement]) -> tuple[BearingMeasurement, ...]:
    ordered = tuple(sorted(measurements, key=lambda item: item.time_seconds))
    if len(ordered) < 2:
        raise ValueError("at least two bearing measurements are required")
    distinct_times = {item.time_seconds for item in ordered}
    if len(distinct_times) < 2:
        raise ValueError("at least two distinct timestamps are required")
    return ordered


def squared_line_error(
    measurements: Sequence[BearingMeasurement],
    initial_range: float,
    speed: float,
    course_deg: float,
) -> float:
    ordered = prepare_measurements(measurements)
    reference = ordered[0]
    start_x, start_y = target_start_from_initial_range(reference, initial_range)
    velocity_x, velocity_y = velocity_components(speed, course_deg)

    sample = ordered[1:]
    elapsed = np.asarray([item.time_seconds - reference.time_seconds for item in sample], dtype=float)
    ownship_x = np.asarray([item.ownship_x for item in sample], dtype=float)
    ownship_y = np.asarray([item.ownship_y for item in sample], dtype=float)
    bearings_rad = np.deg2rad(np.asarray([item.bearing_deg for item in sample], dtype=float))

    target_x = start_x + velocity_x * elapsed
    target_y = start_y + velocity_y * elapsed
    errors = (target_y - ownship_y) * np.sin(bearings_rad) - (target_x - ownship_x) * np.cos(bearings_rad)
    return float(np.dot(errors, errors))
