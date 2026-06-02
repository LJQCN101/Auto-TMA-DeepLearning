from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from auto_tma.deep_learning import RangeRegressionDataset, RangeScenarioConfig, RangeTransformerConfig, save_range_regression_dataset
from auto_tma.geometry import bearing_between_points, signed_angle_delta_deg, target_position_at_time
from auto_tma.models import BearingMeasurement
from auto_tma.predict import (
    ImageLineAnnotation,
    ObservationBundle,
    _load_dataset_sample_bundle,
    _lines_to_measurements,
    _positions_from_ranges,
    _predict_snapshot,
)
from auto_tma.steady_course import SteadyCourseSolution, predict_steady_course_track


class PredictionHelperTests(unittest.TestCase):
    def test_positions_from_ranges_follow_measurement_bearings(self) -> None:
        measurements = (
            BearingMeasurement(time_seconds=0.0, ownship_x=0.0, ownship_y=0.0, bearing_deg=90.0),
            BearingMeasurement(time_seconds=30.0, ownship_x=10.0, ownship_y=0.0, bearing_deg=0.0),
        )

        positions = _positions_from_ranges(measurements, (100.0, 50.0))

        self.assertAlmostEqual(positions[0][0], 100.0)
        self.assertAlmostEqual(positions[0][1], 0.0)
        self.assertAlmostEqual(positions[1][0], 10.0)
        self.assertAlmostEqual(positions[1][1], 50.0)

    def test_lines_to_measurements_converts_and_sorts_annotations(self) -> None:
        lines = (
            (10, 90, 60, 40),
            (20, 80, 20, 30),
        )
        annotations = (
            ImageLineAnnotation(line_index=1, observation_index=2, time_seconds=None, ownship_endpoint="start"),
            ImageLineAnnotation(line_index=0, observation_index=0, time_seconds=None, ownship_endpoint="end"),
        )

        measurements = _lines_to_measurements(
            lines,
            annotations,
            image_shape=(100, 200, 3),
            time_step_seconds=15.0,
            units_per_pixel=2.0,
        )

        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0].time_seconds, 0.0)
        self.assertEqual(measurements[1].time_seconds, 30.0)
        self.assertAlmostEqual(measurements[0].ownship_x, 120.0)
        self.assertAlmostEqual(measurements[0].ownship_y, 120.0)
        self.assertAlmostEqual(measurements[1].ownship_x, 40.0)
        self.assertAlmostEqual(measurements[1].ownship_y, 40.0)
        self.assertAlmostEqual(
            measurements[0].bearing_deg,
            bearing_between_points(120.0, 120.0, 20.0, 20.0),
        )
        self.assertAlmostEqual(
            measurements[1].bearing_deg,
            bearing_between_points(40.0, 40.0, 40.0, 140.0),
        )

    def test_load_dataset_sample_bundle_reconstructs_measurements_and_truth(self) -> None:
        scenario_config = RangeScenarioConfig(sequence_length=2, time_step_seconds=20.0, max_initial_range=1000.0, max_target_speed=5.0)
        dataset = RangeRegressionDataset(
            features=np.asarray(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.2, -0.1, 1.0, 0.0, 0.25],
                    ]
                ],
                dtype=np.float32,
            ),
            targets=np.asarray([[0.9, 1.1]], dtype=np.float32),
            target_scale=1000.0,
            velocity_targets=np.asarray([[[0.1, -0.2], [0.3, 0.4]]], dtype=np.float32),
            velocity_scale=5.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "validation_dataset.npz"
            save_range_regression_dataset(dataset_path, dataset=dataset, scenario_config=scenario_config)

            bundle, truth_ranges, truth_velocities, resolved_index, sample_count = _load_dataset_sample_bundle(
                dataset_path,
                0,
            )

        self.assertEqual(resolved_index, 0)
        self.assertEqual(sample_count, 1)
        self.assertEqual(bundle.source_description, f"{dataset_path} [sample 0]")
        self.assertEqual(len(bundle.measurements), 2)
        self.assertAlmostEqual(bundle.measurements[0].time_seconds, 0.0)
        self.assertAlmostEqual(bundle.measurements[1].time_seconds, 20.0)
        self.assertAlmostEqual(bundle.measurements[1].ownship_x, 200.0)
        self.assertAlmostEqual(bundle.measurements[1].ownship_y, -100.0)
        self.assertAlmostEqual(bundle.measurements[0].bearing_deg, 0.0)
        self.assertAlmostEqual(bundle.measurements[1].bearing_deg, 90.0)
        self.assertEqual(truth_ranges, (900.0, 1100.0))
        self.assertEqual(truth_velocities, ((0.5, -1.0), (1.5, 2.0)))

    def test_predict_snapshot_attaches_ground_truth_overlay(self) -> None:
        measurements = (
            BearingMeasurement(time_seconds=0.0, ownship_x=0.0, ownship_y=0.0, bearing_deg=0.0),
            BearingMeasurement(time_seconds=20.0, ownship_x=20.0, ownship_y=0.0, bearing_deg=90.0),
        )
        bundle = ObservationBundle(
            measurements=measurements,
            source_description="test",
        )

        fake_loaded = type(
            "Loaded",
            (),
            {
                "model": object(),
                "device": "cpu",
                "checkpoint_path": Path("fake.pt"),
                "scenario_config": RangeScenarioConfig(sequence_length=2, max_initial_range=1000.0, max_target_speed=5.0),
                "model_config": RangeTransformerConfig(architecture="baseline"),
            },
        )()

        with patch("auto_tma.predict.predict_range_and_velocity") as mocked_predict:
            mocked_predict.return_value = (
                np.asarray([[100.0, 120.0]], dtype=np.float32),
                np.asarray([[[1.0, 0.0], [0.5, 0.5]]], dtype=np.float32),
            )
            snapshot = _predict_snapshot(
                bundle,
                fake_loaded,
                ground_truth_ranges=(95.0, 125.0),
                ground_truth_velocities=((0.9, 0.1), (0.4, 0.6)),
                ground_truth_label="Validation truth",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.ground_truth_ranges, (95.0, 125.0))
        self.assertEqual(snapshot.ground_truth_velocities, ((0.9, 0.1), (0.4, 0.6)))
        self.assertEqual(snapshot.ground_truth_label, "Validation truth")
        self.assertEqual(snapshot.ground_truth_positions, _positions_from_ranges(measurements, (95.0, 125.0)))

    def test_predict_steady_course_track_recovers_constant_motion(self) -> None:
        reference = BearingMeasurement(time_seconds=0.0, ownship_x=0.0, ownship_y=0.0, bearing_deg=35.0)
        truth_initial_range = 800.0
        truth_speed = 4.5
        truth_course = 60.0
        ownship_path = (
            (30.0, 40.0, 0.0),
            (60.0, 90.0, 5.0),
            (90.0, 150.0, 15.0),
            (120.0, 220.0, 20.0),
        )
        measurements = [reference]
        truth_positions = [target_position_at_time(reference, truth_initial_range, truth_speed, truth_course, 0.0)]
        for time_seconds, ownship_x, ownship_y in ownship_path:
            target_position = target_position_at_time(
                reference,
                truth_initial_range,
                truth_speed,
                truth_course,
                time_seconds,
            )
            truth_positions.append(target_position)
            measurements.append(
                BearingMeasurement(
                    time_seconds=time_seconds,
                    ownship_x=ownship_x,
                    ownship_y=ownship_y,
                    bearing_deg=bearing_between_points(ownship_x, ownship_y, target_position[0], target_position[1]),
                )
            )

        solution, positions, ranges = predict_steady_course_track(
            measurements,
            max_initial_range=2000.0,
            max_target_speed=8.0,
        )

        self.assertAlmostEqual(solution.initial_range, truth_initial_range, delta=40.0)
        self.assertAlmostEqual(solution.speed, truth_speed, delta=0.4)
        self.assertLess(abs(signed_angle_delta_deg(solution.course_deg, truth_course)), 4.0)
        self.assertAlmostEqual(positions[-1][0], truth_positions[-1][0], delta=35.0)
        self.assertAlmostEqual(positions[-1][1], truth_positions[-1][1], delta=35.0)
        self.assertEqual(len(ranges), len(measurements))

    def test_predict_snapshot_attaches_steady_course_overlay(self) -> None:
        measurements = (
            BearingMeasurement(time_seconds=0.0, ownship_x=0.0, ownship_y=0.0, bearing_deg=0.0),
            BearingMeasurement(time_seconds=20.0, ownship_x=20.0, ownship_y=0.0, bearing_deg=90.0),
        )
        bundle = ObservationBundle(
            measurements=measurements,
            source_description="test",
        )

        fake_loaded = type(
            "Loaded",
            (),
            {
                "model": object(),
                "device": "cpu",
                "checkpoint_path": Path("fake.pt"),
                "scenario_config": RangeScenarioConfig(sequence_length=2, max_initial_range=1000.0, max_target_speed=5.0),
                "model_config": RangeTransformerConfig(architecture="baseline"),
            },
        )()
        steady_solution = SteadyCourseSolution(initial_range=110.0, speed=2.5, course_deg=45.0, objective=0.0)

        with patch("auto_tma.predict.predict_range_and_velocity") as mocked_predict, patch(
            "auto_tma.predict.predict_steady_course_track"
        ) as mocked_steady:
            mocked_predict.return_value = (
                np.asarray([[100.0, 120.0]], dtype=np.float32),
                np.asarray([[[1.0, 0.0], [0.5, 0.5]]], dtype=np.float32),
            )
            mocked_steady.return_value = (
                steady_solution,
                ((0.0, 110.0), (98.0, 78.0)),
                (110.0, 110.72488428533129),
            )
            snapshot = _predict_snapshot(bundle, fake_loaded)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.steady_course_solution, steady_solution)
        self.assertEqual(snapshot.steady_course_positions, ((0.0, 110.0), (98.0, 78.0)))
        self.assertEqual(snapshot.steady_course_ranges, (110.0, 110.72488428533129))
        self.assertEqual(snapshot.steady_course_label, "Steady-course baseline (auto_tma_python_simple)")

    def test_predict_snapshot_requires_full_window(self) -> None:
        measurements = (
            BearingMeasurement(time_seconds=0.0, ownship_x=0.0, ownship_y=0.0, bearing_deg=0.0),
            BearingMeasurement(time_seconds=10.0, ownship_x=20.0, ownship_y=0.0, bearing_deg=10.0),
        )
        bundle = ObservationBundle(
            measurements=measurements,
            source_description="test",
        )

        fake_loaded = type(
            "Loaded",
            (),
            {
                "model": object(),
                "device": "cpu",
                "checkpoint_path": Path("fake.pt"),
                "scenario_config": RangeScenarioConfig(sequence_length=3, max_initial_range=1000.0, max_target_speed=5.0),
                "model_config": RangeTransformerConfig(architecture="baseline"),
            },
        )()

        self.assertIsNone(_predict_snapshot(bundle, fake_loaded))


if __name__ == "__main__":
    unittest.main()
