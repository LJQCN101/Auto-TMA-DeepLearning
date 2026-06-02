from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BearingMeasurement:
    time_seconds: float
    ownship_x: float
    ownship_y: float
    bearing_deg: float
