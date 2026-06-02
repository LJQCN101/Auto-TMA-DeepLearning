from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .geometry import normalize_course_deg, prepare_measurements, range_between_points, target_position_at_time
from .models import BearingMeasurement


@dataclass(frozen=True, slots=True)
class SteadyCourseSolution:
    initial_range: float
    speed: float
    course_deg: float
    objective: float


def solve_steady_course(
    measurements: Sequence[BearingMeasurement],
    *,
    max_initial_range: float,
    max_target_speed: float,
) -> SteadyCourseSolution:
    ordered = prepare_measurements(measurements)
    upper_range = max(float(max_initial_range), 1.0)
    upper_speed = max(float(max_target_speed), 0.1)
    lower_range = max(upper_range * 1e-3, 1.0)

    best = _search_grid(
        ordered,
        range_values=np.linspace(lower_range, upper_range, num=33, dtype=np.float64),
        speed_values=np.linspace(0.0, upper_speed, num=25, dtype=np.float64),
        course_values=np.linspace(0.0, 360.0, num=72, endpoint=False, dtype=np.float64),
    )

    range_span = upper_range * 0.35
    speed_span = upper_speed * 0.35
    course_span = 45.0
    for _ in range(4):
        best = _search_grid(
            ordered,
            range_values=_bounded_linspace(best.initial_range, range_span, lower_range, upper_range, count=25),
            speed_values=_bounded_linspace(best.speed, speed_span, 0.0, upper_speed, count=21),
            course_values=np.asarray(
                [normalize_course_deg(best.course_deg + offset) for offset in np.linspace(-course_span, course_span, 49)],
                dtype=np.float64,
            ),
        )
        range_span *= 0.4
        speed_span *= 0.4
        course_span *= 0.4

    return best


def predict_steady_course_track(
    measurements: Sequence[BearingMeasurement],
    *,
    max_initial_range: float,
    max_target_speed: float,
) -> tuple[SteadyCourseSolution, tuple[tuple[float, float], ...], tuple[float, ...]]:
    ordered = prepare_measurements(measurements)
    solution = solve_steady_course(
        ordered,
        max_initial_range=max_initial_range,
        max_target_speed=max_target_speed,
    )
    positions = tuple(_track_positions_from_solution(ordered, solution))
    ranges = tuple(
        float(range_between_points(measurement.ownship_x, measurement.ownship_y, target_x, target_y))
        for measurement, (target_x, target_y) in zip(ordered, positions)
    )
    return solution, positions, ranges


def _track_positions_from_solution(
    measurements: Sequence[BearingMeasurement],
    solution: SteadyCourseSolution,
) -> tuple[tuple[float, float], ...]:
    ordered = prepare_measurements(measurements)
    reference = ordered[0]
    return tuple(
        tuple(
            float(value)
            for value in target_position_at_time(
                reference,
                solution.initial_range,
                solution.speed,
                solution.course_deg,
                measurement.time_seconds,
            )
        )
        for measurement in ordered
    )


def _bounded_linspace(center: float, span: float, lower: float, upper: float, *, count: int) -> np.ndarray:
    start = max(lower, center - span)
    stop = min(upper, center + span)
    if math.isclose(start, stop, rel_tol=0.0, abs_tol=1e-9):
        return np.asarray([start], dtype=np.float64)
    return np.linspace(start, stop, num=count, dtype=np.float64)


def _search_grid(
    measurements: tuple[BearingMeasurement, ...],
    *,
    range_values: np.ndarray,
    speed_values: np.ndarray,
    course_values: np.ndarray,
) -> SteadyCourseSolution:
    reference = measurements[0]
    sample = measurements[1:]
    elapsed = np.asarray([item.time_seconds - reference.time_seconds for item in sample], dtype=np.float64)
    ownship_x = np.asarray([item.ownship_x for item in sample], dtype=np.float64)
    ownship_y = np.asarray([item.ownship_y for item in sample], dtype=np.float64)
    bearing_rad = np.deg2rad(np.asarray([item.bearing_deg for item in sample], dtype=np.float64))
    line_sin = np.sin(bearing_rad)
    line_cos = np.cos(bearing_rad)

    reference_bearing = math.radians(reference.bearing_deg)
    start_x = reference.ownship_x + range_values[:, None, None] * math.sin(reference_bearing)
    start_y = reference.ownship_y + range_values[:, None, None] * math.cos(reference_bearing)

    course_rad = np.deg2rad(course_values)[None, None, :]
    speed_grid = speed_values[None, :, None]
    velocity_x = speed_grid * np.sin(course_rad)
    velocity_y = speed_grid * np.cos(course_rad)

    target_x = start_x[..., None] + velocity_x[..., None] * elapsed
    target_y = start_y[..., None] + velocity_y[..., None] * elapsed
    line_error = (target_y - ownship_y) * line_sin - (target_x - ownship_x) * line_cos
    objective = np.sum(line_error * line_error, axis=-1)

    best_index = np.unravel_index(int(np.argmin(objective)), objective.shape)
    return SteadyCourseSolution(
        initial_range=float(range_values[best_index[0]]),
        speed=float(speed_values[best_index[1]]),
        course_deg=float(normalize_course_deg(course_values[best_index[2]])),
        objective=float(objective[best_index]),
    )


__all__ = [
    "SteadyCourseSolution",
    "predict_steady_course_track",
    "solve_steady_course",
]