from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .deep_learning import (
    build_range_inference_dataset,
    decode_range_regression_dataset_sample,
    load_range_regression_dataset,
    load_range_regressor_checkpoint,
    predict_range_and_velocity,
)
from .geometry import DEG_TO_RAD, bearing_between_points, prepare_measurements, speed_course_from_velocity
from .models import BearingMeasurement
from .steady_course import SteadyCourseSolution, predict_steady_course_track
from .vision import detect_candidate_lines, reduce_lines

LineSegment = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    measurements: tuple[BearingMeasurement, ...]
    source_description: str


@dataclass(frozen=True, slots=True)
class ImageLineAnnotation:
    line_index: int
    observation_index: int | None
    time_seconds: float | None
    ownship_endpoint: str


@dataclass(frozen=True, slots=True)
class PredictionSnapshot:
    mode: str
    display_label: str
    all_measurements: tuple[BearingMeasurement, ...]
    active_measurements: tuple[BearingMeasurement, ...]
    predicted_positions: tuple[tuple[float, float], ...]
    predicted_ranges: tuple[float, ...]
    predicted_velocities: tuple[tuple[float, float], ...] | None
    checkpoint_path: Path
    ground_truth_positions: tuple[tuple[float, float], ...] | None = None
    ground_truth_ranges: tuple[float, ...] | None = None
    ground_truth_velocities: tuple[tuple[float, float], ...] | None = None
    ground_truth_label: str | None = None
    steady_course_positions: tuple[tuple[float, float], ...] | None = None
    steady_course_ranges: tuple[float, ...] | None = None
    steady_course_solution: SteadyCourseSolution | None = None
    steady_course_label: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict target positions from measurements, cached validation datasets, "
            "interactive observations, or detected image bearing lines"
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional trained checkpoint to load. Defaults to the best available local checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--measurements-path",
        type=Path,
        help="Measurement JSON file containing bearing observations for neural inference",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Collect measurements incrementally through the console and update the prediction after each observation",
    )
    input_group.add_argument(
        "--validation-dataset-path",
        type=Path,
        help="Cached validation dataset artifact (.npz) to reconstruct measurements and visualize against ground truth",
    )
    input_group.add_argument(
        "--image-path",
        type=Path,
        help="Image containing bearing lines to detect and convert into measurements",
    )
    parser.add_argument(
        "--dataset-sample-index",
        type=int,
        default=0,
        help="Sample index within --validation-dataset-path. Negative values index from the end.",
    )
    parser.add_argument(
        "--line-annotations-path",
        type=Path,
        default=None,
        help="Optional JSON file describing which detected image lines to keep and which endpoint is the ownship point",
    )
    parser.add_argument(
        "--time-step-seconds",
        type=float,
        default=30.0,
        help="Default time step used for image annotations or interactive observation indices",
    )
    parser.add_argument(
        "--units-per-pixel",
        type=float,
        default=1.0,
        help="Scale factor for image-mode coordinates. Use this to calibrate pixel distances into real units.",
    )
    parser.add_argument(
        "--visualization-path",
        type=Path,
        default=None,
        help="Optional image path for the rendered prediction plot",
    )
    parser.add_argument(
        "--save-measurements-path",
        type=Path,
        default=None,
        help="Optional JSON path to save the normalized measurements extracted from interactive or image input",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path or _default_checkpoint_path()
    if checkpoint_path is None or not checkpoint_path.exists():
        parser.error("no checkpoint was provided and no default trained checkpoint was found")

    resolved_device = _resolve_device_name(args.device)
    loaded = load_range_regressor_checkpoint(checkpoint_path, device=resolved_device)

    show_plots = not args.no_show
    if args.measurements_path is not None:
        bundle = _load_measurement_bundle(args.measurements_path)
        snapshot = _predict_snapshot(bundle, loaded)
        if snapshot is None:
            parser.error(
                f"at least {loaded.scenario_config.sequence_length} measurements are required for prediction"
            )
        _print_snapshot_summary(snapshot)
        _render_snapshot(
            snapshot,
            visualization_path=args.visualization_path,
            show=show_plots,
            annotated_image=None,
        )
        return

    if args.validation_dataset_path is not None:
        try:
            bundle, truth_ranges, truth_velocities, resolved_index, sample_count = _load_dataset_sample_bundle(
                args.validation_dataset_path,
                args.dataset_sample_index,
                fallback_scenario_config=loaded.scenario_config,
            )
        except (IndexError, ValueError) as error:
            parser.error(str(error))
        snapshot = _predict_snapshot(
            bundle,
            loaded,
            ground_truth_ranges=truth_ranges,
            ground_truth_velocities=truth_velocities,
            ground_truth_label=f"Validation truth (sample {resolved_index}/{sample_count - 1})",
        )
        if snapshot is None:
            parser.error(
                f"selected dataset sample does not contain {loaded.scenario_config.sequence_length} measurements"
            )
        _print_snapshot_summary(snapshot)
        _render_snapshot(
            snapshot,
            visualization_path=args.visualization_path,
            show=show_plots,
            annotated_image=None,
        )
        return

    if args.interactive:
        bundle, snapshot = _run_interactive_console(loaded, show_plots=show_plots)
        if args.save_measurements_path is not None:
            _save_measurement_bundle(bundle, args.save_measurements_path)
        if snapshot is not None and args.visualization_path is not None:
            _render_snapshot(snapshot, visualization_path=args.visualization_path, show=False, annotated_image=None)
        return

    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"failed to read image: {args.image_path}")
    reduced_lines = reduce_lines(detect_candidate_lines(image))
    if not reduced_lines:
        parser.error("no candidate bearing lines were detected in the image")

    annotated_image = _annotate_image(image, reduced_lines)
    line_annotations, annotation_overrides = _resolve_image_annotations(
        reduced_lines,
        args.line_annotations_path,
        show_image=show_plots,
        annotated_image=annotated_image,
    )
    time_step_seconds = float(annotation_overrides.get("time_step_seconds", args.time_step_seconds))
    units_per_pixel = float(annotation_overrides.get("units_per_pixel", args.units_per_pixel))
    bundle = ObservationBundle(
        measurements=_lines_to_measurements(
            reduced_lines,
            line_annotations,
            image_shape=image.shape,
            time_step_seconds=time_step_seconds,
            units_per_pixel=units_per_pixel,
        ),
        source_description=str(args.image_path),
    )
    if args.save_measurements_path is not None:
        _save_measurement_bundle(bundle, args.save_measurements_path)

    snapshot = _predict_snapshot(bundle, loaded)
    if snapshot is None:
        parser.error(
            f"at least {loaded.scenario_config.sequence_length} image-derived measurements are required for prediction"
        )
    _print_snapshot_summary(snapshot)
    _render_snapshot(
        snapshot,
        visualization_path=args.visualization_path,
        show=show_plots,
        annotated_image=annotated_image,
    )


def _resolve_device_name(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _default_checkpoint_path() -> Path | None:
    candidates = (
        Path("outputs/baseline_regression_large_2m.pt"),
        Path("outputs/baseline_regression_large_2m_epoch_012.pt"),
        Path("outputs/kronos_regression_large_2m.pt"),
        Path("outputs/kronos_regression_large_2m_epoch_012.pt"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_measurement_bundle(input_path: Path) -> ObservationBundle:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    measurements = tuple(
        BearingMeasurement(
            time_seconds=float(item["time_seconds"]),
            ownship_x=float(item["ownship_x"]),
            ownship_y=float(item["ownship_y"]),
            bearing_deg=float(item["bearing_deg"]),
        )
        for item in payload["measurements"]
    )
    return ObservationBundle(
        measurements=measurements,
        source_description=str(input_path),
    )


def _save_measurement_bundle(bundle: ObservationBundle, output_path: Path) -> None:
    payload: dict[str, object] = {
        "measurements": [
            {
                "time_seconds": measurement.time_seconds,
                "ownship_x": measurement.ownship_x,
                "ownship_y": measurement.ownship_y,
                "bearing_deg": measurement.bearing_deg,
            }
            for measurement in bundle.measurements
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_dataset_sample_bundle(
    input_path: Path,
    sample_index: int,
    *,
    fallback_scenario_config=None,
) -> tuple[ObservationBundle, tuple[float, ...], tuple[tuple[float, float], ...], int, int]:
    artifact = load_range_regression_dataset(input_path)
    scenario_config = artifact.scenario_config or fallback_scenario_config
    if scenario_config is None:
        raise ValueError("dataset artifact does not include scenario_config and no fallback scenario_config was provided")

    resolved_index = sample_index
    if resolved_index < 0:
        resolved_index += artifact.dataset.sample_count
    if resolved_index < 0 or resolved_index >= artifact.dataset.sample_count:
        raise IndexError(
            f"sample index {sample_index} is out of range for dataset of size {artifact.dataset.sample_count}"
        )

    sample = decode_range_regression_dataset_sample(
        artifact.dataset,
        resolved_index,
        scenario_config=scenario_config,
    )
    return (
        ObservationBundle(
            measurements=sample.measurements,
            source_description=f"{input_path} [sample {resolved_index}]",
        ),
        sample.truth_ranges,
        sample.truth_velocities,
        resolved_index,
        artifact.dataset.sample_count,
    )


def _run_interactive_console(
    loaded,
    *,
    show_plots: bool,
) -> tuple[ObservationBundle, PredictionSnapshot | None]:
    print("Enter observations as: time_seconds ownship_x ownship_y bearing_deg")
    print(
        f"The predictor emits a neural estimate once {loaded.scenario_config.sequence_length} observations are available."
    )
    print("Enter 'done' or a blank line to finish.")

    bundle = ObservationBundle(
        measurements=(),
        source_description="interactive_console",
    )
    figure_state = _create_figure(include_image=False) if show_plots else None
    snapshot: PredictionSnapshot | None = None

    while True:
        raw = input("observation> ").strip()
        if raw == "" or raw.lower() in {"done", "quit", "exit"}:
            break
        parts = raw.replace(",", " ").split()
        if len(parts) != 4:
            print("Expected four values: time_seconds ownship_x ownship_y bearing_deg")
            continue
        try:
            observation = BearingMeasurement(
                time_seconds=float(parts[0]),
                ownship_x=float(parts[1]),
                ownship_y=float(parts[2]),
                bearing_deg=float(parts[3]),
            )
        except ValueError:
            print("Could not parse the observation; please enter numeric values only")
            continue

        bundle = ObservationBundle(
            measurements=tuple(sorted(bundle.measurements + (observation,), key=lambda item: item.time_seconds)),
            source_description=bundle.source_description,
        )
        if len(bundle.measurements) < loaded.scenario_config.sequence_length:
            remaining = loaded.scenario_config.sequence_length - len(bundle.measurements)
            print(f"Need {remaining} more observations before prediction is available")
            continue

        snapshot = _predict_snapshot(bundle, loaded)
        _print_snapshot_summary(snapshot)
        if figure_state is not None:
            _draw_snapshot(*figure_state, snapshot=snapshot, annotated_image=None)

    if show_plots and figure_state is not None:
        _finalize_figure(figure_state[0])
    return bundle, snapshot


def _resolve_image_annotations(
    lines: Sequence[LineSegment],
    annotations_path: Path | None,
    *,
    show_image: bool,
    annotated_image: np.ndarray,
) -> tuple[tuple[ImageLineAnnotation, ...], dict[str, float]]:
    overrides: dict[str, float] = {}
    if annotations_path is not None:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "time_step_seconds" in payload:
            overrides["time_step_seconds"] = float(payload["time_step_seconds"])
        if "units_per_pixel" in payload:
            overrides["units_per_pixel"] = float(payload["units_per_pixel"])
        annotations = tuple(_annotation_from_payload(item) for item in payload.get("lines", ()))
        return annotations, overrides

    if show_image:
        figure_state = _create_figure(include_image=True)
        _draw_snapshot(*figure_state, snapshot=None, annotated_image=annotated_image)

    print("Detected reduced lines are indexed on the image. Choose which lines correspond to observations.")
    print("For each selected line, enter the observation index and whether the ownship point is the start or end endpoint.")
    annotations: list[ImageLineAnnotation] = []
    for index, line in enumerate(lines):
        print(f"line {index}: start=({line[0]}, {line[1]}), end=({line[2]}, {line[3]})")
        raw_index = input("  observation index (negative to skip)> ").strip()
        try:
            observation_index = int(raw_index)
        except ValueError:
            observation_index = -1
        if observation_index < 0:
            continue
        endpoint = input("  ownship endpoint [start/end, default=end]> ").strip().lower()
        annotations.append(
            ImageLineAnnotation(
                line_index=index,
                observation_index=observation_index,
                time_seconds=None,
                ownship_endpoint=_normalize_endpoint(endpoint),
            )
        )

    return tuple(annotations), overrides


def _annotation_from_payload(payload: dict[str, object]) -> ImageLineAnnotation:
    if not isinstance(payload, dict):
        raise ValueError("line annotation entries must be objects")
    return ImageLineAnnotation(
        line_index=int(payload["line_index"]),
        observation_index=None if payload.get("observation_index") is None else int(payload["observation_index"]),
        time_seconds=None if payload.get("time_seconds") is None else float(payload["time_seconds"]),
        ownship_endpoint=_normalize_endpoint(str(payload.get("ownship_endpoint", "end"))),
    )


def _normalize_endpoint(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"start", "a", "s", "0", "first"}:
        return "start"
    return "end"


def _lines_to_measurements(
    lines: Sequence[LineSegment],
    annotations: Sequence[ImageLineAnnotation],
    *,
    image_shape: Sequence[int],
    time_step_seconds: float,
    units_per_pixel: float,
) -> tuple[BearingMeasurement, ...]:
    image_height = int(image_shape[0])
    measurements: list[BearingMeasurement] = []
    for annotation in annotations:
        x1, y1, x2, y2 = lines[annotation.line_index]
        if annotation.ownship_endpoint == "start":
            ownship_img = (x1, y1)
            target_img = (x2, y2)
        else:
            ownship_img = (x2, y2)
            target_img = (x1, y1)

        ownship_x = float(ownship_img[0] * units_per_pixel)
        ownship_y = float((image_height - ownship_img[1]) * units_per_pixel)
        target_x = float(target_img[0] * units_per_pixel)
        target_y = float((image_height - target_img[1]) * units_per_pixel)
        measurement_time = (
            float(annotation.time_seconds)
            if annotation.time_seconds is not None
            else float(annotation.observation_index * time_step_seconds)
        )
        measurements.append(
            BearingMeasurement(
                time_seconds=measurement_time,
                ownship_x=ownship_x,
                ownship_y=ownship_y,
                bearing_deg=float(bearing_between_points(ownship_x, ownship_y, target_x, target_y)),
            )
        )
    return tuple(sorted(measurements, key=lambda item: item.time_seconds))


def _predict_snapshot(
    bundle: ObservationBundle,
    loaded,
    *,
    ground_truth_ranges: Sequence[float] | None = None,
    ground_truth_velocities: Sequence[tuple[float, float]] | None = None,
    ground_truth_label: str | None = None,
) -> PredictionSnapshot | None:
    if len(bundle.measurements) < loaded.scenario_config.sequence_length:
        return None

    ordered_measurements = prepare_measurements(bundle.measurements)
    active_measurements = ordered_measurements[-loaded.scenario_config.sequence_length :]
    steady_course_solution, steady_course_positions, steady_course_ranges = predict_steady_course_track(
        active_measurements,
        max_initial_range=loaded.scenario_config.max_initial_range,
        max_target_speed=loaded.scenario_config.max_target_speed,
    )
    dataset = build_range_inference_dataset(active_measurements, scenario_config=loaded.scenario_config)
    range_predictions, velocity_predictions = predict_range_and_velocity(
        loaded.model,
        dataset,
        device=loaded.device,
        batch_size=1,
    )
    predicted_ranges = tuple(float(value) for value in range_predictions[0])
    predicted_velocities = tuple((float(x), float(y)) for x, y in velocity_predictions[0])
    model_label = "baseline transformer" if loaded.model_config.architecture == "baseline" else "kronos regressor"
    truth_positions, active_truth_ranges, active_truth_velocities, resolved_truth_label = _resolve_ground_truth_overlay(
        active_measurements,
        ordered_measurements=ordered_measurements,
        ground_truth_ranges=ground_truth_ranges,
        ground_truth_velocities=ground_truth_velocities,
        ground_truth_label=ground_truth_label,
    )
    return PredictionSnapshot(
        mode="neural",
        display_label=f"{model_label} (latest {loaded.scenario_config.sequence_length}/{len(ordered_measurements)} obs)",
        all_measurements=ordered_measurements,
        active_measurements=active_measurements,
        predicted_positions=_positions_from_ranges(active_measurements, predicted_ranges),
        predicted_ranges=predicted_ranges,
        predicted_velocities=predicted_velocities,
        checkpoint_path=loaded.checkpoint_path,
        ground_truth_positions=truth_positions,
        ground_truth_ranges=active_truth_ranges,
        ground_truth_velocities=active_truth_velocities,
        ground_truth_label=resolved_truth_label,
        steady_course_positions=steady_course_positions,
        steady_course_ranges=steady_course_ranges,
        steady_course_solution=steady_course_solution,
        steady_course_label="Steady-course baseline (auto_tma_python_simple)",
    )


def _resolve_ground_truth_overlay(
    active_measurements: Sequence[BearingMeasurement],
    *,
    ordered_measurements: Sequence[BearingMeasurement],
    ground_truth_ranges: Sequence[float] | None,
    ground_truth_velocities: Sequence[tuple[float, float]] | None,
    ground_truth_label: str | None,
) -> tuple[
    tuple[tuple[float, float], ...] | None,
    tuple[float, ...] | None,
    tuple[tuple[float, float], ...] | None,
    str | None,
]:
    if ground_truth_ranges is None:
        return None, None, None, None
    if len(ground_truth_ranges) != len(ordered_measurements):
        raise ValueError(
            f"ground truth range length {len(ground_truth_ranges)} does not match measurement count {len(ordered_measurements)}"
        )

    active_length = len(active_measurements)
    active_truth_ranges = tuple(float(value) for value in ground_truth_ranges[-active_length:])
    active_truth_velocities = None
    if ground_truth_velocities is not None:
        if len(ground_truth_velocities) != len(ordered_measurements):
            raise ValueError(
                "ground truth velocity length "
                f"{len(ground_truth_velocities)} does not match measurement count {len(ordered_measurements)}"
            )
        active_truth_velocities = tuple(
            (float(velocity_x), float(velocity_y))
            for velocity_x, velocity_y in ground_truth_velocities[-active_length:]
        )
    return (
        _positions_from_ranges(active_measurements, active_truth_ranges),
        active_truth_ranges,
        active_truth_velocities,
        ground_truth_label or "Ground truth",
    )


def _positions_from_ranges(
    measurements: Sequence[BearingMeasurement],
    predicted_ranges: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    positions: list[tuple[float, float]] = []
    for measurement, predicted_range in zip(measurements, predicted_ranges):
        bearing_rad = measurement.bearing_deg * DEG_TO_RAD
        positions.append(
            (
                float(measurement.ownship_x + predicted_range * math.sin(bearing_rad)),
                float(measurement.ownship_y + predicted_range * math.cos(bearing_rad)),
            )
        )
    return tuple(positions)


def _print_snapshot_summary(snapshot: PredictionSnapshot) -> None:
    final_position = snapshot.predicted_positions[-1]
    final_range = snapshot.predicted_ranges[-1]
    summary = (
        f"{snapshot.display_label}: final_target=({final_position[0]:.3f}, {final_position[1]:.3f}), "
        f"final_range={final_range:.3f}, source_checkpoint={snapshot.checkpoint_path}"
    )
    if snapshot.predicted_velocities is not None:
        speed, course_deg = speed_course_from_velocity(*snapshot.predicted_velocities[-1])
        summary += f", final_speed={speed:.3f}, final_course={course_deg:.3f}"
    if snapshot.ground_truth_positions is not None and snapshot.ground_truth_ranges is not None:
        truth_position = snapshot.ground_truth_positions[-1]
        truth_range = snapshot.ground_truth_ranges[-1]
        position_error = math.dist(final_position, truth_position)
        summary += (
            f", truth_target=({truth_position[0]:.3f}, {truth_position[1]:.3f}), truth_range={truth_range:.3f}, "
            f"range_error={final_range - truth_range:.3f}, position_error={position_error:.3f}"
        )
    if (
        snapshot.steady_course_positions is not None
        and snapshot.steady_course_ranges is not None
        and snapshot.steady_course_solution is not None
    ):
        steady_position = snapshot.steady_course_positions[-1]
        steady_range = snapshot.steady_course_ranges[-1]
        summary += (
            f", steady_target=({steady_position[0]:.3f}, {steady_position[1]:.3f}), "
            f"steady_range={steady_range:.3f}, steady_speed={snapshot.steady_course_solution.speed:.3f}, "
            f"steady_course={snapshot.steady_course_solution.course_deg:.3f}"
        )
        if snapshot.ground_truth_positions is not None and snapshot.ground_truth_ranges is not None:
            truth_position = snapshot.ground_truth_positions[-1]
            truth_range = snapshot.ground_truth_ranges[-1]
            summary += (
                f", steady_range_error={steady_range - truth_range:.3f}, "
                f"steady_position_error={math.dist(steady_position, truth_position):.3f}"
            )
    print(summary)


def _create_figure(include_image: bool):
    import matplotlib.pyplot as plt

    plt.ion()
    if include_image:
        figure, (image_axis, geometry_axis) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        figure, geometry_axis = plt.subplots(1, 1, figsize=(8, 7))
        image_axis = None
    return figure, image_axis, geometry_axis


def _draw_snapshot(figure, image_axis, geometry_axis, *, snapshot: PredictionSnapshot | None, annotated_image: np.ndarray | None) -> None:
    import matplotlib.pyplot as plt

    if image_axis is not None:
        image_axis.clear()
        if annotated_image is not None:
            image_axis.imshow(annotated_image)
            image_axis.set_title("Detected bearing lines")
        image_axis.axis("off")

    geometry_axis.clear()
    if snapshot is not None:
        active_times = {measurement.time_seconds for measurement in snapshot.active_measurements}
        ownship_x = [measurement.ownship_x for measurement in snapshot.all_measurements]
        ownship_y = [measurement.ownship_y for measurement in snapshot.all_measurements]
        geometry_axis.plot(ownship_x, ownship_y, color="black", linewidth=1.5, marker="o", label="Ownship path")

        max_range = max(snapshot.predicted_ranges)
        if snapshot.ground_truth_ranges is not None:
            max_range = max(max_range, max(snapshot.ground_truth_ranges))
        if snapshot.steady_course_ranges is not None:
            max_range = max(max_range, max(snapshot.steady_course_ranges))
        ray_length = max(max_range * 1.15, 50.0)
        for measurement in snapshot.all_measurements:
            bearing_rad = measurement.bearing_deg * DEG_TO_RAD
            ray_end_x = measurement.ownship_x + ray_length * math.sin(bearing_rad)
            ray_end_y = measurement.ownship_y + ray_length * math.cos(bearing_rad)
            is_active = measurement.time_seconds in active_times
            geometry_axis.plot(
                [measurement.ownship_x, ray_end_x],
                [measurement.ownship_y, ray_end_y],
                linestyle="--",
                linewidth=1.0 if is_active else 0.8,
                color="#4f86c6" if is_active else "#b0b7c3",
                alpha=0.75 if is_active else 0.35,
            )

        if snapshot.ground_truth_positions is not None:
            truth_x = [position[0] for position in snapshot.ground_truth_positions]
            truth_y = [position[1] for position in snapshot.ground_truth_positions]
            geometry_axis.plot(
                truth_x,
                truth_y,
                color="#1f7a4d",
                linewidth=2.0,
                linestyle="-.",
                marker="s",
                label=snapshot.ground_truth_label or "Ground truth",
            )
            geometry_axis.scatter(
                [truth_x[-1]],
                [truth_y[-1]],
                color="#2ecc71",
                edgecolors="black",
                marker="D",
                s=90,
                label="Ground truth final target",
            )

        if snapshot.steady_course_positions is not None:
            steady_x = [position[0] for position in snapshot.steady_course_positions]
            steady_y = [position[1] for position in snapshot.steady_course_positions]
            geometry_axis.plot(
                steady_x,
                steady_y,
                color="#8f6a3f",
                linewidth=1.8,
                linestyle=":",
                marker="^",
                label=snapshot.steady_course_label or "Steady-course baseline",
            )
            geometry_axis.scatter(
                [steady_x[-1]],
                [steady_y[-1]],
                color="#d4a15e",
                edgecolors="black",
                marker="^",
                s=110,
                label="Steady-course final target",
            )

        target_x = [position[0] for position in snapshot.predicted_positions]
        target_y = [position[1] for position in snapshot.predicted_positions]
        geometry_axis.plot(
            target_x,
            target_y,
            color="#c0392b",
            linewidth=2.0,
            marker="o",
            label="Predicted target track",
        )
        geometry_axis.scatter(
            [target_x[-1]],
            [target_y[-1]],
            color="#f1c40f",
            edgecolors="black",
            marker="*",
            s=180,
            label="Predicted final target",
        )
        geometry_axis.set_title(snapshot.display_label)
        geometry_axis.legend(loc="best")

    geometry_axis.set_xlabel("x")
    geometry_axis.set_ylabel("y")
    geometry_axis.grid(True, alpha=0.3)
    geometry_axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    plt.pause(0.001)


def _render_snapshot(
    snapshot: PredictionSnapshot,
    *,
    visualization_path: Path | None,
    show: bool,
    annotated_image: np.ndarray | None,
) -> None:
    figure_state = _create_figure(include_image=annotated_image is not None)
    _draw_snapshot(*figure_state, snapshot=snapshot, annotated_image=annotated_image)
    figure = figure_state[0]
    if visualization_path is not None:
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(visualization_path, dpi=200, bbox_inches="tight")
    if show:
        _finalize_figure(figure)
    else:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _finalize_figure(figure) -> None:
    import matplotlib.pyplot as plt

    plt.ioff()
    figure.canvas.draw_idle()
    plt.show()


def _annotate_image(image: np.ndarray, lines: Sequence[LineSegment]) -> np.ndarray:
    annotated = image.copy()
    for index, (x1, y1, x2, y2) in enumerate(lines):
        cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(annotated, (x1, y1), 4, (0, 255, 0), -1)
        cv2.circle(annotated, (x2, y2), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{index}",
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(annotated, "A", (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, "B", (x2 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


__all__ = [
    "ImageLineAnnotation",
    "ObservationBundle",
    "PredictionSnapshot",
    "main",
    "_load_dataset_sample_bundle",
    "_lines_to_measurements",
    "_positions_from_ranges",
]


if __name__ == "__main__":
    main()
