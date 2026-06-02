from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .geometry import bearing_between_points, velocity_components
from .models import BearingMeasurement


def generate_variable_velocity_track(
    times: Sequence[float],
    start_x: float,
    start_y: float,
    speeds: Sequence[float],
    courses_deg: Sequence[float],
) -> np.ndarray:
    ordered_times = np.asarray(times, dtype=float)
    segment_speeds = np.asarray(speeds, dtype=float)
    segment_courses_deg = np.asarray(courses_deg, dtype=float)

    if ordered_times.ndim != 1 or ordered_times.size == 0:
        raise ValueError("times must be a one-dimensional non-empty sequence")
    if segment_speeds.shape != ordered_times.shape:
        raise ValueError("speeds must have the same shape as times")
    if segment_courses_deg.shape != ordered_times.shape:
        raise ValueError("courses_deg must have the same shape as times")

    track = np.zeros((ordered_times.size, 2), dtype=float)
    track[0] = (start_x, start_y)
    for index in range(1, ordered_times.size):
        delta_t = float(ordered_times[index] - ordered_times[index - 1])
        if delta_t <= 0.0:
            raise ValueError("times must be strictly increasing")
        velocity_x, velocity_y = velocity_components(
            float(segment_speeds[index - 1]),
            float(segment_courses_deg[index - 1]),
        )
        track[index] = track[index - 1] + delta_t * np.asarray([velocity_x, velocity_y], dtype=float)
    return track


def generate_constant_velocity_track(
    times: Sequence[float],
    start_x: float,
    start_y: float,
    speed: float,
    course_deg: float,
) -> np.ndarray:
    ordered_times = np.asarray(times, dtype=float)
    return generate_variable_velocity_track(
        times=ordered_times,
        start_x=start_x,
        start_y=start_y,
        speeds=np.full(ordered_times.shape, float(speed), dtype=float),
        courses_deg=np.full(ordered_times.shape, float(course_deg), dtype=float),
    )


def simulate_bearing_measurements_from_track(
    times: Sequence[float],
    ownship_positions: Iterable[Sequence[float]],
    target_track: np.ndarray,
    bearing_noise_std_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[BearingMeasurement]:
    ordered_times = np.asarray(times, dtype=float)
    ownship = np.asarray(list(ownship_positions), dtype=float)
    track = np.asarray(target_track, dtype=float)

    if ownship.shape != (ordered_times.size, 2):
        raise ValueError("ownship_positions must have shape (len(times), 2)")
    if track.shape != (ordered_times.size, 2):
        raise ValueError("target_track must have shape (len(times), 2)")

    noise_source = rng or np.random.default_rng()
    measurements: list[BearingMeasurement] = []
    for index, time_seconds in enumerate(ordered_times):
        ownship_x, ownship_y = ownship[index]
        target_x, target_y = track[index]
        bearing_deg = bearing_between_points(ownship_x, ownship_y, target_x, target_y)
        if bearing_noise_std_deg > 0.0:
            bearing_deg += float(noise_source.normal(0.0, bearing_noise_std_deg))
        measurements.append(
            BearingMeasurement(
                time_seconds=float(time_seconds),
                ownship_x=float(ownship_x),
                ownship_y=float(ownship_y),
                bearing_deg=float(bearing_deg % 360.0),
            )
        )
    return measurements


def simulate_bearing_measurements(
    times: Sequence[float],
    ownship_positions: Iterable[Sequence[float]],
    target_start_x: float,
    target_start_y: float,
    target_speed: float,
    target_course_deg: float,
    bearing_noise_std_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[BearingMeasurement]:
    ordered_times = np.asarray(times, dtype=float)
    target_track = generate_constant_velocity_track(
        times=ordered_times,
        start_x=target_start_x,
        start_y=target_start_y,
        speed=target_speed,
        course_deg=target_course_deg,
    )
    return simulate_bearing_measurements_from_track(
        times=ordered_times,
        ownship_positions=ownship_positions,
        target_track=target_track,
        bearing_noise_std_deg=bearing_noise_std_deg,
        rng=rng,
    )
