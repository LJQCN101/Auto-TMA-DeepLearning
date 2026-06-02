from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from auto_tma.deep_learning import (
    _build_cli_parser,
    DEFAULT_CLI_CHECKPOINT_PATH,
    DEFAULT_CLI_TRAIN_DATASET_PATH,
    DEFAULT_CLI_VALIDATION_DATASET_PATH,
    build_range_inference_dataset,
    RangeScenarioConfig,
    RangeTrainingConfig,
    RangeTransformerConfig,
    generate_range_regression_dataset,
    generate_range_scenario_samples,
    load_range_regression_dataset,
    load_range_regressor_checkpoint,
    predict_range_and_velocity,
    predict_range_series,
    predict_initial_ranges,
    predict_velocity_series,
    run_range_regression_experiment,
    save_range_regression_dataset,
)


class DeepLearningExperimentTests(unittest.TestCase):
    def test_cli_defaults_match_large_baseline_recipe(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args([])

        self.assertEqual(args.train_samples, 2_000_000)
        self.assertEqual(args.validation_samples, 65_536)
        self.assertEqual(args.architecture, "baseline")
        self.assertEqual(args.d_model, 512)
        self.assertEqual(args.num_heads, 8)
        self.assertEqual(args.num_layers, 8)
        self.assertEqual(args.ff_dim, 1024)
        self.assertEqual(args.batch_size, 4096)
        self.assertAlmostEqual(args.learning_rate, 3e-5)
        self.assertAlmostEqual(args.weight_decay, 0.01)
        self.assertTrue(args.continuous_ownship_maneuvering)
        self.assertAlmostEqual(args.max_target_turn_deg_per_step, 7.0)
        self.assertAlmostEqual(args.max_ownship_turn_deg_per_step, 6.0)
        self.assertAlmostEqual(args.ownship_speed_std, 0.15)
        self.assertEqual(args.checkpoint_path, DEFAULT_CLI_CHECKPOINT_PATH)
        self.assertEqual(args.train_dataset_path, DEFAULT_CLI_TRAIN_DATASET_PATH)
        self.assertEqual(args.validation_dataset_path, DEFAULT_CLI_VALIDATION_DATASET_PATH)

    def test_dataset_generation_has_expected_shape(self) -> None:
        scenario_config = RangeScenarioConfig(
            sequence_length=9,
            time_step_seconds=20.0,
            min_initial_range=700.0,
            max_initial_range=1500.0,
            bearing_noise_std_deg=0.05,
        )

        dataset = generate_range_regression_dataset(
            6,
            scenario_config=scenario_config,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(dataset.features.shape, (6, 9, 6))
        self.assertEqual(dataset.targets.shape, (6, 9))
        self.assertEqual(dataset.velocity_targets.shape, (6, 9, 2))
        self.assertTrue(np.all(dataset.targets > 0.0))
        self.assertTrue(np.all(dataset.targets[:, 0] <= 1.0))
        self.assertTrue(np.all(np.abs(dataset.velocity_targets) <= 1.0 + 1e-6))
        self.assertEqual(dataset.target_scale, scenario_config.max_initial_range)
        self.assertEqual(dataset.velocity_scale, scenario_config.max_target_speed)

    def test_saved_datasets_can_be_reused_for_training(self) -> None:
        scenario_config = RangeScenarioConfig(
            sequence_length=9,
            time_step_seconds=20.0,
            min_initial_range=700.0,
            max_initial_range=1200.0,
            min_target_speed=1.8,
            max_target_speed=3.0,
            max_target_turn_deg_per_step=2.5,
            bearing_noise_std_deg=0.0,
        )
        train_dataset = generate_range_regression_dataset(
            24,
            scenario_config=scenario_config,
            rng=np.random.default_rng(13),
        )
        validation_dataset = generate_range_regression_dataset(
            8,
            scenario_config=scenario_config,
            rng=np.random.default_rng(17),
        )
        model_config = RangeTransformerConfig(
            d_model=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.0,
        )
        training_config = RangeTrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=2e-3,
            weight_decay=0.0,
            seed=19,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train_dataset.npz"
            validation_path = Path(temp_dir) / "validation_dataset.npz"
            save_range_regression_dataset(train_path, dataset=train_dataset, scenario_config=scenario_config)
            save_range_regression_dataset(validation_path, dataset=validation_dataset, scenario_config=scenario_config)

            loaded_train = load_range_regression_dataset(train_path)
            loaded_validation = load_range_regression_dataset(validation_path)
            self.assertTrue(np.array_equal(loaded_train.dataset.features, train_dataset.features))
            self.assertTrue(np.array_equal(loaded_validation.dataset.targets, validation_dataset.targets))

            _, result = run_range_regression_experiment(
                train_samples=24,
                validation_samples=8,
                scenario_config=scenario_config,
                model_config=model_config,
                training_config=training_config,
                train_dataset_path=train_path,
                validation_dataset_path=validation_path,
            )

        self.assertEqual(result.train_samples, 24)
        self.assertEqual(result.validation_samples, 8)
        self.assertTrue(np.isfinite(result.validation_mae))

    def test_range_regression_experiment_learns_small_synthetic_problem(self) -> None:
        scenario_config = RangeScenarioConfig(
            sequence_length=9,
            time_step_seconds=20.0,
            min_initial_range=700.0,
            max_initial_range=1200.0,
            min_target_speed=1.8,
            max_target_speed=3.0,
            min_ownship_speed=2.0,
            max_ownship_speed=4.0,
            max_target_turn_deg_per_step=2.5,
            bearing_noise_std_deg=0.0,
        )
        model_config = RangeTransformerConfig(
            d_model=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.0,
        )
        training_config = RangeTrainingConfig(
            epochs=6,
            batch_size=8,
            learning_rate=2e-3,
            weight_decay=0.0,
            seed=11,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "range_regressor.pt"

            model, result = run_range_regression_experiment(
                train_samples=48,
                validation_samples=16,
                scenario_config=scenario_config,
                model_config=model_config,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
            )

            self.assertTrue(checkpoint_path.exists())
            for epoch in range(1, training_config.epochs + 1):
                epoch_checkpoint = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}_epoch_{epoch:03d}{checkpoint_path.suffix}"
                )
                self.assertTrue(epoch_checkpoint.exists())

            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint_payload["model_config"]["architecture"], "baseline")

            loaded_checkpoint = load_range_regressor_checkpoint(checkpoint_path, device="cpu")
            self.assertEqual(loaded_checkpoint.model_config.architecture, "baseline")
            self.assertEqual(loaded_checkpoint.scenario_config.sequence_length, 9)

            inference_dataset = build_range_inference_dataset(
                generate_range_scenario_samples(
                    1,
                    scenario_config=scenario_config,
                    rng=np.random.default_rng(123),
                )[0].measurements,
                scenario_config=scenario_config,
            )
            loaded_range_predictions, loaded_velocity_predictions = predict_range_and_velocity(
                loaded_checkpoint.model,
                inference_dataset,
                device="cpu",
                batch_size=1,
            )
            self.assertEqual(loaded_range_predictions.shape, (1, 9))
            self.assertEqual(loaded_velocity_predictions.shape, (1, 9, 2))

        self.assertEqual(result.scenario_config.time_step_seconds, 20.0)
        self.assertEqual(result.scenario_config.sequence_length, 9)
        self.assertLess(result.train_loss_history[-1], result.train_loss_history[0])
        self.assertTrue(np.isfinite(result.validation_mae))
        self.assertTrue(np.isfinite(result.validation_rmse))
        self.assertTrue(np.isfinite(result.validation_velocity_mae))
        self.assertTrue(np.isfinite(result.validation_velocity_rmse))

        prediction_dataset = generate_range_regression_dataset(
            4,
            scenario_config=scenario_config,
            rng=np.random.default_rng(91),
        )
        range_predictions, velocity_predictions = predict_range_and_velocity(model, prediction_dataset, device="cpu")
        predictions = predict_range_series(model, prediction_dataset, device="cpu")
        initial_predictions = predict_initial_ranges(model, prediction_dataset, device="cpu")
        predicted_velocity_series = predict_velocity_series(model, prediction_dataset, device="cpu")
        batched_range_predictions, batched_velocity_predictions = predict_range_and_velocity(
            model,
            prediction_dataset,
            device="cpu",
            batch_size=2,
        )

        self.assertEqual(predictions.shape, (4, 9))
        self.assertEqual(initial_predictions.shape, (4,))
        self.assertEqual(range_predictions.shape, (4, 9))
        self.assertEqual(velocity_predictions.shape, (4, 9, 2))
        self.assertEqual(predicted_velocity_series.shape, (4, 9, 2))
        self.assertEqual(batched_range_predictions.shape, (4, 9))
        self.assertEqual(batched_velocity_predictions.shape, (4, 9, 2))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(np.isfinite(initial_predictions)))
        self.assertTrue(np.all(np.isfinite(range_predictions)))
        self.assertTrue(np.all(np.isfinite(velocity_predictions)))
        self.assertTrue(np.all(np.isfinite(predicted_velocity_series)))
        self.assertTrue(np.allclose(range_predictions, batched_range_predictions))
        self.assertTrue(np.allclose(velocity_predictions, batched_velocity_predictions))

    def test_scenario_generator_mixes_constant_and_maneuvering_targets(self) -> None:
        scenario_config = RangeScenarioConfig(
            sequence_length=9,
            time_step_seconds=20.0,
            min_initial_range=700.0,
            max_initial_range=1200.0,
            min_target_speed=1.8,
            max_target_speed=3.0,
            max_target_turn_deg_per_step=2.5,
            constant_target_fraction=0.5,
            bearing_noise_std_deg=0.0,
        )

        scenarios = generate_range_scenario_samples(
            32,
            scenario_config=scenario_config,
            rng=np.random.default_rng(7),
        )

        constant_count = 0
        for scenario in scenarios:
            velocity = np.asarray(scenario.target_velocity, dtype=float)
            if np.allclose(velocity, velocity[0], atol=1e-9):
                constant_count += 1

        self.assertGreater(constant_count, 0)
        self.assertLess(constant_count, len(scenarios))

    def test_kronos_architecture_runs_regression_pipeline(self) -> None:
        scenario_config = RangeScenarioConfig(
            sequence_length=9,
            time_step_seconds=20.0,
            min_initial_range=700.0,
            max_initial_range=1200.0,
            min_target_speed=1.8,
            max_target_speed=3.0,
            max_target_turn_deg_per_step=2.5,
            bearing_noise_std_deg=0.0,
        )
        model_config = RangeTransformerConfig(
            d_model=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.0,
            architecture="kronos",
        )
        training_config = RangeTrainingConfig(
            epochs=3,
            batch_size=8,
            learning_rate=2e-3,
            weight_decay=0.0,
            seed=5,
            device="cpu",
        )

        model, result = run_range_regression_experiment(
            train_samples=24,
            validation_samples=8,
            scenario_config=scenario_config,
            model_config=model_config,
            training_config=training_config,
        )
        prediction_dataset = generate_range_regression_dataset(
            2,
            scenario_config=scenario_config,
            rng=np.random.default_rng(33),
        )
        range_predictions, velocity_predictions = predict_range_and_velocity(model, prediction_dataset, device="cpu")

        self.assertEqual(result.model_config.architecture, "kronos")
        self.assertEqual(range_predictions.shape, (2, 9))
        self.assertEqual(velocity_predictions.shape, (2, 9, 2))
        self.assertTrue(np.all(np.isfinite(range_predictions)))
        self.assertTrue(np.all(np.isfinite(velocity_predictions)))


if __name__ == "__main__":
    unittest.main()