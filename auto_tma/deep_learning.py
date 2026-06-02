from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .geometry import (
    DEG_TO_RAD,
    normalize_course_deg,
    prepare_measurements,
    range_between_points,
    signed_angle_delta_deg,
    velocity_components,
)
from .models import BearingMeasurement
from .simulation import (
    generate_constant_velocity_track,
    generate_variable_velocity_track,
    simulate_bearing_measurements_from_track,
)


@dataclass(frozen=True, slots=True)
class RangeScenarioConfig:
    sequence_length: int = 17
    time_step_seconds: float = 30.0
    min_initial_range: float = 800.0
    max_initial_range: float = 2600.0
    min_target_speed: float = 1.5
    max_target_speed: float = 4.0
    target_speed_std: float = 0.18
    min_ownship_speed: float = 2.0
    max_ownship_speed: float = 5.0
    ownship_speed_std: float = 0.12
    max_target_turn_deg_per_step: float = 5.0
    max_ownship_turn_deg_per_step: float = 4.0
    continuous_ownship_maneuvering: bool = False
    constant_target_fraction: float = 0.25
    bearing_noise_std_deg: float = 0.2


@dataclass(frozen=True, slots=True)
class RangeScenarioSample:
    measurements: tuple[BearingMeasurement, ...]
    initial_range: float
    ownship_positions: tuple[tuple[float, float], ...]
    target_track: tuple[tuple[float, float], ...]
    range_to_ownship: tuple[float, ...]
    target_velocity: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class RangeRegressionDataset:
    features: np.ndarray
    targets: np.ndarray
    target_scale: float
    velocity_targets: np.ndarray
    velocity_scale: float

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def sequence_length(self) -> int:
        return int(self.features.shape[1])

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class RangeRegressionDatasetArtifact:
    dataset: RangeRegressionDataset
    scenario_config: RangeScenarioConfig | None


@dataclass(frozen=True, slots=True)
class RangeRegressionDatasetSample:
    measurements: tuple[BearingMeasurement, ...]
    truth_ranges: tuple[float, ...]
    truth_velocities: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class RangeTransformerConfig:
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 128
    dropout: float = 0.1
    architecture: str = "baseline"


@dataclass(frozen=True, slots=True)
class RangeTrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    velocity_loss_weight: float = 0.5
    seed: int = 17
    device: str = "auto"


DEFAULT_CLI_TRAIN_SAMPLES = 2_000_000
DEFAULT_CLI_VALIDATION_SAMPLES = 65_536
DEFAULT_CLI_SCENARIO_CONFIG = RangeScenarioConfig(
    sequence_length=17,
    time_step_seconds=30.0,
    ownship_speed_std=0.15,
    max_target_turn_deg_per_step=7.0,
    max_ownship_turn_deg_per_step=6.0,
    continuous_ownship_maneuvering=True,
    constant_target_fraction=0.25,
    bearing_noise_std_deg=0.2,
)
DEFAULT_CLI_MODEL_CONFIG = RangeTransformerConfig(
    d_model=512,
    num_heads=8,
    num_layers=8,
    ff_dim=1024,
    dropout=0.1,
    architecture="baseline",
)
DEFAULT_CLI_TRAINING_CONFIG = RangeTrainingConfig(
    epochs=12,
    batch_size=4096,
    learning_rate=3e-5,
    weight_decay=0.01,
    velocity_loss_weight=0.5,
    seed=17,
    device="auto",
)
DEFAULT_CLI_CHECKPOINT_PATH = Path("outputs/baseline_regression_large_2m.pt")
DEFAULT_CLI_TRAIN_DATASET_PATH = Path("outputs/datasets/train_dataset.npz")
DEFAULT_CLI_VALIDATION_DATASET_PATH = Path("outputs/datasets/validation_dataset.npz")


@dataclass(frozen=True, slots=True)
class RangeExperimentResult:
    scenario_config: RangeScenarioConfig
    model_config: RangeTransformerConfig
    training_config: RangeTrainingConfig
    train_samples: int
    validation_samples: int
    train_loss_history: tuple[float, ...]
    validation_loss_history: tuple[float, ...]
    validation_mae: float
    validation_rmse: float
    validation_velocity_mae: float
    validation_velocity_rmse: float
    device: str


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values * torch.rsqrt(torch.mean(values * values, dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.value_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.out_proj = nn.Linear(ff_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.out_proj(F.silu(self.gate_proj(values)) * self.value_proj(values)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self._cached_sequence_length: int | None = None
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin_cache(query, query.shape[-2])
        return (
            (query * cos) + (self._rotate_half(query) * sin),
            (key * cos) + (self._rotate_half(key) * sin),
        )

    def _cos_sin_cache(self, values: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_length != self._cached_sequence_length:
            self._cached_sequence_length = sequence_length
            positions = torch.arange(sequence_length, device=values.device, dtype=self.inverse_frequency.dtype)
            frequencies = torch.einsum("i,j->ij", positions, self.inverse_frequency)
            embedding = torch.cat((frequencies, frequencies), dim=-1).to(values.device)
            self._cached_cos = embedding.cos()[None, None, :, :]
            self._cached_sin = embedding.sin()[None, None, :, :]
        return self._cached_cos, self._cached_sin

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first_half, second_half = values.chunk(2, dim=-1)
        return torch.cat((-second_half, first_half), dim=-1)


class KronosSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape

        query = self.query_proj(values).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(values).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(values).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key = self.rotary(query, key)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attention_scores = attention_scores.masked_fill(mask, torch.finfo(attention_scores.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.residual_dropout(self.output_proj(attended))


class KronosTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = KronosSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = GatedFeedForward(d_model, ff_dim, dropout)

    def forward(self, values: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_input = self.attention_norm(values)
        values = values + self.attention(attention_input, key_padding_mask=key_padding_mask)
        feed_forward_input = self.ffn_norm(values)
        values = values + self.feed_forward(feed_forward_input)
        return values


class RangeTransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, config: RangeTransformerConfig) -> None:
        super().__init__()
        self.architecture = config.architecture
        self.input_proj = nn.Linear(input_dim, config.d_model)
        if self.architecture == "baseline":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length + 1, config.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.kronos_blocks: nn.ModuleList | None = None
            self.output_norm: nn.Module = nn.Identity()
        else:
            self.cls_token = None
            self.positional_embedding = None
            self.encoder = None
            self.kronos_blocks = nn.ModuleList(
                KronosTransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            )
            self.output_norm = RMSNorm(config.d_model)
        self.range_head = nn.Sequential(
            _make_head_norm(config.d_model, architecture=self.architecture),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.velocity_head = nn.Sequential(
            _make_head_norm(config.d_model, architecture=self.architecture),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 2),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(features)
        if self.architecture == "baseline":
            cls_token = self.cls_token.expand(features.shape[0], -1, -1)
            hidden = torch.cat([cls_token, hidden], dim=1)
            hidden = hidden + self.positional_embedding[:, : hidden.shape[1], :]
            encoded = self.encoder(hidden)
            token_states = encoded[:, 1:, :]
        else:
            for block in self.kronos_blocks:
                hidden = block(hidden)
            token_states = self.output_norm(hidden)
        return self.range_head(token_states).squeeze(-1), self.velocity_head(token_states)


@dataclass(frozen=True, slots=True)
class LoadedRangeRegressor:
    checkpoint_path: Path
    model: RangeTransformerRegressor
    scenario_config: RangeScenarioConfig
    model_config: RangeTransformerConfig
    training_config: RangeTrainingConfig
    metrics: dict[str, object]
    device: str


def generate_range_regression_dataset(
    sample_count: int,
    *,
    scenario_config: RangeScenarioConfig | None = None,
    rng: np.random.Generator | None = None,
) -> RangeRegressionDataset:
    active_config = scenario_config or RangeScenarioConfig()
    scenarios = generate_range_scenario_samples(
        sample_count,
        scenario_config=active_config,
        rng=rng,
    )
    return build_range_regression_dataset(
        scenarios,
        target_scale=active_config.max_initial_range,
        velocity_scale=active_config.max_target_speed,
    )


def generate_range_scenario_samples(
    sample_count: int,
    *,
    scenario_config: RangeScenarioConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[RangeScenarioSample, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    active_config = scenario_config or RangeScenarioConfig()
    _validate_scenario_config(active_config)
    random_source = rng or np.random.default_rng()

    return tuple(_generate_synthetic_scenario(active_config, random_source) for _ in range(sample_count))


def build_range_regression_dataset(
    scenarios: tuple[RangeScenarioSample, ...] | list[RangeScenarioSample],
    *,
    target_scale: float,
    velocity_scale: float,
) -> RangeRegressionDataset:
    if not scenarios:
        raise ValueError("at least one scenario is required")

    scenario_list = list(scenarios)
    sequence_length = len(scenario_list[0].measurements)
    features = np.zeros((len(scenario_list), sequence_length, 6), dtype=np.float32)
    targets = np.zeros((len(scenario_list), sequence_length), dtype=np.float32)
    velocity_targets = np.zeros((len(scenario_list), sequence_length, 2), dtype=np.float32)

    for index, scenario in enumerate(scenario_list):
        if len(scenario.measurements) != sequence_length:
            raise ValueError("all scenarios must have the same sequence length")
        features[index] = _measurements_to_features(list(scenario.measurements), position_scale=target_scale)
        targets[index] = np.asarray(scenario.range_to_ownship, dtype=np.float32) / float(target_scale)
        velocity_targets[index] = np.asarray(scenario.target_velocity, dtype=np.float32) / float(velocity_scale)

    return RangeRegressionDataset(
        features=features,
        targets=targets,
        target_scale=target_scale,
        velocity_targets=velocity_targets,
        velocity_scale=velocity_scale,
    )


def build_range_inference_dataset(
    measurements: list[BearingMeasurement] | tuple[BearingMeasurement, ...],
    *,
    scenario_config: RangeScenarioConfig,
) -> RangeRegressionDataset:
    ordered_measurements = prepare_measurements(measurements)
    if len(ordered_measurements) != scenario_config.sequence_length:
        raise ValueError(
            f"expected exactly {scenario_config.sequence_length} measurements, got {len(ordered_measurements)}"
        )

    features = np.zeros((1, scenario_config.sequence_length, 6), dtype=np.float32)
    features[0] = _measurements_to_features(
        list(ordered_measurements),
        position_scale=scenario_config.max_initial_range,
    )
    return RangeRegressionDataset(
        features=features,
        targets=np.zeros((1, scenario_config.sequence_length), dtype=np.float32),
        target_scale=scenario_config.max_initial_range,
        velocity_targets=np.zeros((1, scenario_config.sequence_length, 2), dtype=np.float32),
        velocity_scale=scenario_config.max_target_speed,
    )


def save_range_regression_dataset(
    output_path: Path,
    *,
    dataset: RangeRegressionDataset,
    scenario_config: RangeScenarioConfig | None = None,
) -> None:
    payload = {
        "features": dataset.features,
        "targets": dataset.targets,
        "target_scale": np.asarray(dataset.target_scale, dtype=np.float32),
        "velocity_targets": dataset.velocity_targets,
        "velocity_scale": np.asarray(dataset.velocity_scale, dtype=np.float32),
        "scenario_config_json": np.asarray(
            json.dumps(None if scenario_config is None else asdict(scenario_config)),
            dtype=np.str_,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez(handle, **payload)


def load_range_regression_dataset(input_path: Path) -> RangeRegressionDatasetArtifact:
    with np.load(input_path, allow_pickle=False) as payload:
        dataset = RangeRegressionDataset(
            features=np.array(payload["features"], dtype=np.float32, copy=True),
            targets=np.array(payload["targets"], dtype=np.float32, copy=True),
            target_scale=float(np.asarray(payload["target_scale"]).item()),
            velocity_targets=np.array(payload["velocity_targets"], dtype=np.float32, copy=True),
            velocity_scale=float(np.asarray(payload["velocity_scale"]).item()),
        )
        scenario_payload = json.loads(str(np.asarray(payload["scenario_config_json"]).item()))

    scenario_config = None if scenario_payload is None else RangeScenarioConfig(**scenario_payload)
    return RangeRegressionDatasetArtifact(dataset=dataset, scenario_config=scenario_config)


def decode_range_regression_dataset_sample(
    dataset: RangeRegressionDataset,
    sample_index: int,
    *,
    scenario_config: RangeScenarioConfig,
) -> RangeRegressionDatasetSample:
    resolved_index = sample_index
    if resolved_index < 0:
        resolved_index += dataset.sample_count
    if resolved_index < 0 or resolved_index >= dataset.sample_count:
        raise IndexError(f"sample index {sample_index} is out of range for dataset of size {dataset.sample_count}")

    measurements = _measurements_from_feature_sequence(
        dataset.features[resolved_index],
        position_scale=dataset.target_scale,
        scenario_config=scenario_config,
    )
    truth_ranges = tuple(float(value) for value in dataset.targets[resolved_index] * dataset.target_scale)
    truth_velocity_array = dataset.velocity_targets[resolved_index] * dataset.velocity_scale
    return RangeRegressionDatasetSample(
        measurements=measurements,
        truth_ranges=truth_ranges,
        truth_velocities=tuple((float(x), float(y)) for x, y in truth_velocity_array),
    )


def load_range_regressor_checkpoint(
    input_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedRangeRegressor:
    resolved_device = torch.device(device)
    payload = torch.load(input_path, map_location=resolved_device)
    model_config = RangeTransformerConfig(**payload["model_config"])
    training_config = RangeTrainingConfig(**payload["training_config"])
    scenario_config = RangeScenarioConfig(**payload["scenario_config"])
    state_dict = payload["model_state_dict"]
    input_dim = int(state_dict["input_proj.weight"].shape[1])
    model = RangeTransformerRegressor(
        input_dim=input_dim,
        sequence_length=scenario_config.sequence_length,
        config=model_config,
    )
    model.load_state_dict(state_dict)
    model = model.to(resolved_device)
    return LoadedRangeRegressor(
        checkpoint_path=input_path,
        model=model,
        scenario_config=scenario_config,
        model_config=model_config,
        training_config=training_config,
        metrics=dict(payload.get("metrics", {})),
        device=str(resolved_device),
    )


def train_range_regressor(
    train_dataset: RangeRegressionDataset,
    validation_dataset: RangeRegressionDataset,
    *,
    scenario_config: RangeScenarioConfig | None = None,
    model_config: RangeTransformerConfig | None = None,
    training_config: RangeTrainingConfig | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[RangeTransformerRegressor, RangeExperimentResult]:
    if train_dataset.feature_dim != validation_dataset.feature_dim:
        raise ValueError("train and validation datasets must have the same feature dimension")
    if train_dataset.sequence_length != validation_dataset.sequence_length:
        raise ValueError("train and validation datasets must have the same sequence length")
    if not math.isclose(train_dataset.target_scale, validation_dataset.target_scale):
        raise ValueError("train and validation datasets must use the same target scale")
    if not math.isclose(train_dataset.velocity_scale, validation_dataset.velocity_scale):
        raise ValueError("train and validation datasets must use the same velocity scale")

    active_scenario_config = scenario_config or RangeScenarioConfig(
        sequence_length=train_dataset.sequence_length,
        max_initial_range=train_dataset.target_scale,
    )
    active_model_config = model_config or RangeTransformerConfig()
    active_training_config = training_config or RangeTrainingConfig()
    _validate_model_config(active_model_config)
    _validate_training_config(active_training_config)

    torch.manual_seed(active_training_config.seed)
    device = _resolve_device(active_training_config.device)

    model = RangeTransformerRegressor(
        input_dim=train_dataset.feature_dim,
        sequence_length=train_dataset.sequence_length,
        config=active_model_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_training_config.learning_rate,
        weight_decay=active_training_config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_loader = _dataset_to_loader(train_dataset, active_training_config.batch_size, shuffle=True)
    validation_loader = _dataset_to_loader(validation_dataset, active_training_config.batch_size, shuffle=False)

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []

    for epoch_index in range(active_training_config.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets, batch_velocity_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_velocity_targets = batch_velocity_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            range_predictions, velocity_predictions = model(batch_features)
            range_loss = loss_fn(range_predictions, batch_targets)
            velocity_loss = loss_fn(velocity_predictions, batch_velocity_targets)
            loss = range_loss + active_training_config.velocity_loss_weight * velocity_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss_history.append(float(np.mean(batch_losses)))
        validation_loss_history.append(
            _mean_loss(
                model,
                validation_loader,
                loss_fn,
                device,
                active_training_config.velocity_loss_weight,
            )
        )

        if checkpoint_path is not None:
            save_range_regressor_training_checkpoint(
                _epoch_checkpoint_path(checkpoint_path, epoch_index + 1),
                model=model,
                scenario_config=active_scenario_config,
                model_config=active_model_config,
                training_config=active_training_config,
                train_samples=int(train_dataset.features.shape[0]),
                validation_samples=int(validation_dataset.features.shape[0]),
                train_loss_history=tuple(train_loss_history),
                validation_loss_history=tuple(validation_loss_history),
                epoch=epoch_index + 1,
            )

    validation_predictions = predict_range_series(model, validation_dataset, device=device)
    validation_truth = validation_dataset.targets * validation_dataset.target_scale
    validation_errors = validation_predictions - validation_truth
    validation_velocity_predictions = predict_velocity_series(model, validation_dataset, device=device)
    validation_velocity_truth = validation_dataset.velocity_targets * validation_dataset.velocity_scale
    validation_velocity_errors = validation_velocity_predictions - validation_velocity_truth

    result = RangeExperimentResult(
        scenario_config=active_scenario_config,
        model_config=active_model_config,
        training_config=active_training_config,
        train_samples=int(train_dataset.features.shape[0]),
        validation_samples=int(validation_dataset.features.shape[0]),
        train_loss_history=tuple(train_loss_history),
        validation_loss_history=tuple(validation_loss_history),
        validation_mae=float(np.mean(np.abs(validation_errors))),
        validation_rmse=float(np.sqrt(np.mean(np.square(validation_errors)))),
        validation_velocity_mae=float(np.mean(np.abs(validation_velocity_errors))),
        validation_velocity_rmse=float(np.sqrt(np.mean(np.square(validation_velocity_errors)))),
        device=str(device),
    )
    return model, result


def predict_initial_ranges(
    model: RangeTransformerRegressor,
    dataset: RangeRegressionDataset,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    return predict_range_series(model, dataset, device=device)[:, 0]


def predict_velocity_series(
    model: RangeTransformerRegressor,
    dataset: RangeRegressionDataset,
    *,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
) -> np.ndarray:
    _, velocity_predictions = predict_range_and_velocity(model, dataset, device=device, batch_size=batch_size)
    return velocity_predictions


def predict_range_and_velocity(
    model: RangeTransformerRegressor,
    dataset: RangeRegressionDataset,
    *,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()
    effective_batch_size = _resolve_prediction_batch_size(
        batch_size,
        sample_count=dataset.sample_count,
        device=resolved_device,
    )
    range_batches: list[np.ndarray] = []
    velocity_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start_index in range(0, dataset.sample_count, effective_batch_size):
            stop_index = min(start_index + effective_batch_size, dataset.sample_count)
            features = torch.from_numpy(dataset.features[start_index:stop_index]).to(resolved_device)
            range_predictions, velocity_predictions = model(features)
            range_batches.append(range_predictions.detach().cpu().numpy())
            velocity_batches.append(velocity_predictions.detach().cpu().numpy())
    return (
        np.concatenate(range_batches, axis=0) * dataset.target_scale,
        np.concatenate(velocity_batches, axis=0) * dataset.velocity_scale,
    )


def predict_range_series(
    model: RangeTransformerRegressor,
    dataset: RangeRegressionDataset,
    *,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
) -> np.ndarray:
    range_predictions, _ = predict_range_and_velocity(model, dataset, device=device, batch_size=batch_size)
    return range_predictions


def run_range_regression_experiment(
    *,
    train_samples: int = 512,
    validation_samples: int = 128,
    scenario_config: RangeScenarioConfig | None = None,
    model_config: RangeTransformerConfig | None = None,
    training_config: RangeTrainingConfig | None = None,
    checkpoint_path: Path | None = None,
    train_dataset_path: Path | None = None,
    validation_dataset_path: Path | None = None,
) -> tuple[RangeTransformerRegressor, RangeExperimentResult]:
    active_scenario_config = scenario_config or RangeScenarioConfig()
    active_training_config = training_config or RangeTrainingConfig()
    train_rng = np.random.default_rng(active_training_config.seed)
    validation_rng = np.random.default_rng(active_training_config.seed + 1)

    train_dataset = _load_or_generate_range_dataset(
        train_samples,
        scenario_config=active_scenario_config,
        rng=train_rng,
        dataset_path=train_dataset_path,
        dataset_label="train",
    )
    validation_dataset = _load_or_generate_range_dataset(
        validation_samples,
        scenario_config=active_scenario_config,
        rng=validation_rng,
        dataset_path=validation_dataset_path,
        dataset_label="validation",
    )

    model, result = train_range_regressor(
        train_dataset,
        validation_dataset,
        scenario_config=active_scenario_config,
        model_config=model_config,
        training_config=active_training_config,
        checkpoint_path=checkpoint_path,
    )
    if checkpoint_path is not None:
        save_range_regressor_checkpoint(checkpoint_path, model=model, result=result)
    return model, result


def save_range_regressor_checkpoint(
    output_path: Path,
    *,
    model: RangeTransformerRegressor,
    result: RangeExperimentResult,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "scenario_config": asdict(result.scenario_config),
        "metrics": {
            "train_samples": result.train_samples,
            "validation_samples": result.validation_samples,
            "train_loss_history": list(result.train_loss_history),
            "validation_loss_history": list(result.validation_loss_history),
            "validation_mae": result.validation_mae,
            "validation_rmse": result.validation_rmse,
            "validation_velocity_mae": result.validation_velocity_mae,
            "validation_velocity_rmse": result.validation_velocity_rmse,
            "device": result.device,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def save_range_regressor_training_checkpoint(
    output_path: Path,
    *,
    model: RangeTransformerRegressor,
    scenario_config: RangeScenarioConfig,
    model_config: RangeTransformerConfig,
    training_config: RangeTrainingConfig,
    train_samples: int,
    validation_samples: int,
    train_loss_history: tuple[float, ...],
    validation_loss_history: tuple[float, ...],
    epoch: int,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "scenario_config": asdict(scenario_config),
        "metrics": {
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "train_loss_history": list(train_loss_history),
            "validation_loss_history": list(validation_loss_history),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a transformer-based range-over-time regression experiment")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_CLI_TRAIN_SAMPLES,
        help="Number of synthetic training scenarios",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=DEFAULT_CLI_VALIDATION_SAMPLES,
        help="Number of synthetic validation scenarios",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CLI_TRAINING_CONFIG.epochs, help="Training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CLI_TRAINING_CONFIG.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CLI_TRAINING_CONFIG.learning_rate,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_CLI_TRAINING_CONFIG.weight_decay,
        help="Optimizer weight decay",
    )
    parser.add_argument(
        "--velocity-loss-weight",
        type=float,
        default=DEFAULT_CLI_TRAINING_CONFIG.velocity_loss_weight,
        help="Weight of the auxiliary target-velocity loss relative to the range loss",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_CLI_SCENARIO_CONFIG.sequence_length,
        help="Measurements per synthetic scenario",
    )
    parser.add_argument(
        "--time-step-seconds",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.time_step_seconds,
        help="Seconds between measurements",
    )
    parser.add_argument(
        "--bearing-noise-std-deg",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.bearing_noise_std_deg,
        help="Synthetic bearing noise",
    )
    parser.add_argument(
        "--ownship-speed-std",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.ownship_speed_std,
        help="Ownship speed random-walk standard deviation for synthetic scenarios",
    )
    parser.add_argument(
        "--max-target-turn-deg-per-step",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.max_target_turn_deg_per_step,
        help="Maximum target course change per step in synthetic scenarios",
    )
    parser.add_argument(
        "--max-ownship-turn-deg-per-step",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.max_ownship_turn_deg_per_step,
        help="Maximum ownship course change per step when continuous ownship maneuvering is enabled",
    )
    parser.add_argument(
        "--continuous-ownship-maneuvering",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CLI_SCENARIO_CONFIG.continuous_ownship_maneuvering,
        help="Use continuously maneuvering ownship tracks instead of the simpler three-leg path",
    )
    parser.add_argument(
        "--constant-target-fraction",
        type=float,
        default=DEFAULT_CLI_SCENARIO_CONFIG.constant_target_fraction,
        help="Fraction of synthetic scenarios that keep the target on constant course and speed to balance easy and hard cases",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_CLI_MODEL_CONFIG.d_model,
        help="Transformer embedding dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=DEFAULT_CLI_MODEL_CONFIG.num_heads,
        help="Transformer attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_CLI_MODEL_CONFIG.num_layers,
        help="Transformer encoder layers",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=DEFAULT_CLI_MODEL_CONFIG.ff_dim,
        help="Transformer feed-forward width",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_CLI_MODEL_CONFIG.dropout,
        help="Transformer dropout",
    )
    parser.add_argument(
        "--architecture",
        choices=("baseline", "kronos"),
        default=DEFAULT_CLI_MODEL_CONFIG.architecture,
        help="Model family: the original baseline encoder or a larger Kronos-style regressor without the tokenizer",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CLI_TRAINING_CONFIG.seed, help="Random seed")
    parser.add_argument(
        "--device",
        default=DEFAULT_CLI_TRAINING_CONFIG.device,
        help="Training device: auto, cpu, or cuda",
    )
    parser.add_argument("--output-format", choices=("text", "json"), default="text", help="Summary format")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CLI_CHECKPOINT_PATH,
        help="Path to save the trained checkpoint",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        default=DEFAULT_CLI_TRAIN_DATASET_PATH,
        help="Path to save or reuse the generated training dataset artifact",
    )
    parser.add_argument(
        "--validation-dataset-path",
        type=Path,
        default=DEFAULT_CLI_VALIDATION_DATASET_PATH,
        help="Path to save or reuse the generated validation dataset artifact",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    scenario_config = RangeScenarioConfig(
        sequence_length=args.sequence_length,
        time_step_seconds=args.time_step_seconds,
        ownship_speed_std=args.ownship_speed_std,
        max_target_turn_deg_per_step=args.max_target_turn_deg_per_step,
        max_ownship_turn_deg_per_step=args.max_ownship_turn_deg_per_step,
        continuous_ownship_maneuvering=args.continuous_ownship_maneuvering,
        constant_target_fraction=args.constant_target_fraction,
        bearing_noise_std_deg=args.bearing_noise_std_deg,
    )
    model_config = RangeTransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        architecture=args.architecture,
    )
    training_config = RangeTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        velocity_loss_weight=args.velocity_loss_weight,
        seed=args.seed,
        device=args.device,
    )

    model, result = run_range_regression_experiment(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        scenario_config=scenario_config,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=args.checkpoint_path,
        train_dataset_path=args.train_dataset_path,
        validation_dataset_path=args.validation_dataset_path,
    )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "mode": "deep_learning_range_regression",
                    "scenario_config": asdict(result.scenario_config),
                    "model_config": asdict(result.model_config),
                    "training_config": asdict(result.training_config),
                    "train_samples": result.train_samples,
                    "validation_samples": result.validation_samples,
                    "train_loss_history": list(result.train_loss_history),
                    "validation_loss_history": list(result.validation_loss_history),
                    "validation_mae": result.validation_mae,
                    "validation_rmse": result.validation_rmse,
                    "validation_velocity_mae": result.validation_velocity_mae,
                    "validation_velocity_rmse": result.validation_velocity_rmse,
                    "device": result.device,
                    "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
                    "train_dataset_path": None if args.train_dataset_path is None else str(args.train_dataset_path),
                    "validation_dataset_path": None
                    if args.validation_dataset_path is None
                    else str(args.validation_dataset_path),
                    "epoch_checkpoint_pattern": None
                    if args.checkpoint_path is None
                    else str(_epoch_checkpoint_pattern(args.checkpoint_path)),
                },
                indent=2,
            )
        )
        return

    print("mode: deep learning range trajectory regression")
    print(f"train_samples={result.train_samples}, validation_samples={result.validation_samples}, device={result.device}")
    print(
        f"sequence_length={result.scenario_config.sequence_length}, "
        f"time_step_seconds={result.scenario_config.time_step_seconds:.3f}, "
        f"bearing_noise_std_deg={result.scenario_config.bearing_noise_std_deg:.3f}, "
        f"constant_target_fraction={result.scenario_config.constant_target_fraction:.3f}"
    )
    print(
        f"model: architecture={result.model_config.architecture}, d_model={result.model_config.d_model}, "
        f"heads={result.model_config.num_heads}, layers={result.model_config.num_layers}, "
        f"ff_dim={result.model_config.ff_dim}, dropout={result.model_config.dropout:.3f}"
    )
    print(
        f"training: epochs={result.training_config.epochs}, batch_size={result.training_config.batch_size}, "
        f"learning_rate={result.training_config.learning_rate:.6f}, weight_decay={result.training_config.weight_decay:.6f}, "
        f"velocity_loss_weight={result.training_config.velocity_loss_weight:.3f}"
    )
    print(
        f"validation trajectory: mae={result.validation_mae:.3f}, rmse={result.validation_rmse:.3f}, "
        f"velocity_mae={result.validation_velocity_mae:.3f}, velocity_rmse={result.validation_velocity_rmse:.3f}, "
        f"train_loss={result.train_loss_history[-1]:.6f}, val_loss={result.validation_loss_history[-1]:.6f}"
    )
    if args.checkpoint_path is not None:
        print(f"checkpoint: {args.checkpoint_path}")
        print(f"epoch checkpoints: {_epoch_checkpoint_pattern(args.checkpoint_path)}")
    if args.train_dataset_path is not None:
        print(f"train dataset: {args.train_dataset_path}")
    if args.validation_dataset_path is not None:
        print(f"validation dataset: {args.validation_dataset_path}")


def _validate_scenario_config(config: RangeScenarioConfig) -> None:
    if config.sequence_length < 4:
        raise ValueError("sequence_length must be at least 4")
    if config.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if config.min_initial_range <= 0.0 or config.max_initial_range <= config.min_initial_range:
        raise ValueError("initial range bounds must be positive and increasing")
    if config.min_target_speed <= 0.0 or config.max_target_speed <= config.min_target_speed:
        raise ValueError("target speed bounds must be positive and increasing")
    if config.min_ownship_speed <= 0.0 or config.max_ownship_speed <= config.min_ownship_speed:
        raise ValueError("ownship speed bounds must be positive and increasing")
    if config.ownship_speed_std < 0.0:
        raise ValueError("ownship_speed_std cannot be negative")
    if config.target_speed_std < 0.0:
        raise ValueError("target_speed_std cannot be negative")
    if config.max_target_turn_deg_per_step <= 0.0:
        raise ValueError("max_target_turn_deg_per_step must be positive")
    if config.max_ownship_turn_deg_per_step <= 0.0:
        raise ValueError("max_ownship_turn_deg_per_step must be positive")
    if not 0.0 <= config.constant_target_fraction <= 1.0:
        raise ValueError("constant_target_fraction must be in [0.0, 1.0]")
    if config.bearing_noise_std_deg < 0.0:
        raise ValueError("bearing_noise_std_deg cannot be negative")


def _validate_training_config(config: RangeTrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if config.velocity_loss_weight < 0.0:
        raise ValueError("velocity_loss_weight cannot be negative")


def _validate_model_config(config: RangeTransformerConfig) -> None:
    if config.architecture not in {"baseline", "kronos"}:
        raise ValueError("architecture must be one of: baseline, kronos")
    if config.d_model <= 0:
        raise ValueError("d_model must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.ff_dim <= 0:
        raise ValueError("ff_dim must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if config.architecture == "kronos" and ((config.d_model // config.num_heads) % 2 != 0):
        raise ValueError("kronos architecture requires an even attention head dimension for rotary embeddings")


def _generate_synthetic_scenario(
    config: RangeScenarioConfig,
    rng: np.random.Generator,
) -> RangeScenarioSample:
    times = np.arange(config.sequence_length, dtype=float) * config.time_step_seconds
    ownship_positions = _generate_ownship_positions(times, config, rng)

    initial_range = float(rng.uniform(config.min_initial_range, config.max_initial_range))
    initial_bearing_deg = float(rng.uniform(0.0, 360.0))
    target_start_x = float(ownship_positions[0, 0] + initial_range * math.sin(initial_bearing_deg * DEG_TO_RAD))
    target_start_y = float(ownship_positions[0, 1] + initial_range * math.cos(initial_bearing_deg * DEG_TO_RAD))

    base_target_speed = float(rng.uniform(config.min_target_speed, config.max_target_speed))
    base_target_course = float(rng.uniform(0.0, 360.0))
    if rng.random() < config.constant_target_fraction:
        target_track = generate_constant_velocity_track(
            times=times,
            start_x=target_start_x,
            start_y=target_start_y,
            speed=base_target_speed,
            course_deg=base_target_course,
        )
    else:
        speed_walk = np.cumsum(rng.normal(0.0, config.target_speed_std, size=config.sequence_length))
        speeds = np.clip(
            base_target_speed + speed_walk,
            config.min_target_speed,
            config.max_target_speed,
        )

        turn_steps = rng.normal(
            0.0,
            config.max_target_turn_deg_per_step / 2.5,
            size=config.sequence_length - 1,
        )
        turn_steps = np.clip(
            turn_steps,
            -config.max_target_turn_deg_per_step,
            config.max_target_turn_deg_per_step,
        )
        if config.sequence_length > 6:
            pivot = int(rng.integers(2, config.sequence_length - 3))
            dog_leg = float(rng.uniform(-config.max_target_turn_deg_per_step, config.max_target_turn_deg_per_step))
            span = min(3, config.sequence_length - 1 - pivot)
            turn_steps[pivot : pivot + span] += dog_leg / span
            turn_steps = np.clip(
                turn_steps,
                -config.max_target_turn_deg_per_step,
                config.max_target_turn_deg_per_step,
            )

        courses_deg = np.zeros(config.sequence_length, dtype=float)
        courses_deg[0] = base_target_course
        for index in range(1, config.sequence_length):
            courses_deg[index] = normalize_course_deg(courses_deg[index - 1] + turn_steps[index - 1])

        target_track = generate_variable_velocity_track(
            times=times,
            start_x=target_start_x,
            start_y=target_start_y,
            speeds=speeds,
            courses_deg=courses_deg,
        )
    measurements = simulate_bearing_measurements_from_track(
        times=times,
        ownship_positions=ownship_positions,
        target_track=target_track,
        bearing_noise_std_deg=config.bearing_noise_std_deg,
        rng=rng,
    )
    range_to_ownship = tuple(
        float(range_between_points(ownship_x, ownship_y, target_x, target_y))
        for (ownship_x, ownship_y), (target_x, target_y) in zip(ownship_positions, target_track)
    )
    target_velocity = _track_to_velocity_series(target_track, times)
    return RangeScenarioSample(
        measurements=tuple(measurements),
        initial_range=initial_range,
        ownship_positions=tuple((float(x), float(y)) for x, y in ownship_positions),
        target_track=tuple((float(x), float(y)) for x, y in target_track),
        range_to_ownship=range_to_ownship,
        target_velocity=target_velocity,
    )


def _generate_ownship_positions(
    times: np.ndarray,
    config: RangeScenarioConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.continuous_ownship_maneuvering:
        ownship_speed = float(rng.uniform(config.min_ownship_speed, config.max_ownship_speed))
        speed_walk = np.cumsum(rng.normal(0.0, config.ownship_speed_std, size=times.size))
        speeds = np.clip(
            ownship_speed + speed_walk,
            config.min_ownship_speed,
            config.max_ownship_speed,
        )
        base_course = float(rng.uniform(30.0, 150.0))
        turn_steps = rng.normal(
            0.0,
            config.max_ownship_turn_deg_per_step / 2.0,
            size=times.size - 1,
        )
        turn_steps = np.clip(
            turn_steps,
            -config.max_ownship_turn_deg_per_step,
            config.max_ownship_turn_deg_per_step,
        )
        courses = np.zeros(times.size, dtype=float)
        courses[0] = base_course
        for index in range(1, times.size):
            courses[index] = normalize_course_deg(courses[index - 1] + turn_steps[index - 1])
        return generate_variable_velocity_track(
            times=times,
            start_x=0.0,
            start_y=0.0,
            speeds=speeds,
            courses_deg=courses,
        )

    ownship_speed = float(rng.uniform(config.min_ownship_speed, config.max_ownship_speed))
    first_course = float(rng.uniform(50.0, 130.0))
    second_course = normalize_course_deg(first_course + rng.choice((-1.0, 1.0)) * float(rng.uniform(50.0, 100.0)))
    third_course = normalize_course_deg(second_course + rng.choice((-1.0, 1.0)) * float(rng.uniform(25.0, 70.0)))
    leg_courses = np.asarray([first_course, second_course, third_course], dtype=float)

    positions = np.zeros((times.size, 2), dtype=float)
    for index in range(1, times.size):
        delta_t = float(times[index] - times[index - 1])
        leg_index = min(((index - 1) * 3) // max(times.size - 1, 1), 2)
        velocity_x, velocity_y = velocity_components(ownship_speed, leg_courses[leg_index])
        positions[index] = positions[index - 1] + delta_t * np.asarray([velocity_x, velocity_y], dtype=float)
    return positions


def _measurements_to_features(
    measurements: list[BearingMeasurement],
    *,
    position_scale: float,
) -> np.ndarray:
    total_duration = max(measurements[-1].time_seconds - measurements[0].time_seconds, 1.0)
    reference_x = measurements[0].ownship_x
    reference_y = measurements[0].ownship_y
    features = np.zeros((len(measurements), 6), dtype=np.float32)

    previous_bearing_deg = measurements[0].bearing_deg
    for index, measurement in enumerate(measurements):
        bearing_rad = measurement.bearing_deg * DEG_TO_RAD
        features[index, 0] = float((measurement.time_seconds - measurements[0].time_seconds) / total_duration)
        features[index, 1] = float((measurement.ownship_x - reference_x) / position_scale)
        features[index, 2] = float((measurement.ownship_y - reference_y) / position_scale)
        features[index, 3] = float(math.sin(bearing_rad))
        features[index, 4] = float(math.cos(bearing_rad))
        features[index, 5] = float(signed_angle_delta_deg(measurement.bearing_deg, previous_bearing_deg) / 180.0)
        previous_bearing_deg = measurement.bearing_deg
    return features


def _dataset_to_loader(dataset: RangeRegressionDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.features).float(),
        torch.from_numpy(dataset.targets).float(),
        torch.from_numpy(dataset.velocity_targets).float(),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def _mean_loss(
    model: RangeTransformerRegressor,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    velocity_loss_weight: float,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_features, batch_targets, batch_velocity_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_velocity_targets = batch_velocity_targets.to(device)
            range_predictions, velocity_predictions = model(batch_features)
            range_loss = loss_fn(range_predictions, batch_targets)
            velocity_loss = loss_fn(velocity_predictions, batch_velocity_targets)
            losses.append(float((range_loss + velocity_loss_weight * velocity_loss).item()))
    return float(np.mean(losses))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _make_head_norm(dim: int, *, architecture: str) -> nn.Module:
    if architecture == "kronos":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


def _resolve_prediction_batch_size(
    batch_size: int | None,
    *,
    sample_count: int,
    device: torch.device,
) -> int:
    if sample_count <= 0:
        raise ValueError("dataset must contain at least one sample")
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return min(batch_size, sample_count)
    if device.type == "cuda":
        return min(1024, sample_count)
    return sample_count


def _load_or_generate_range_dataset(
    sample_count: int,
    *,
    scenario_config: RangeScenarioConfig,
    rng: np.random.Generator,
    dataset_path: Path | None,
    dataset_label: str,
) -> RangeRegressionDataset:
    if dataset_path is not None and dataset_path.exists():
        artifact = load_range_regression_dataset(dataset_path)
        _validate_loaded_dataset_artifact(
            artifact,
            expected_sample_count=sample_count,
            expected_scenario_config=scenario_config,
            dataset_label=dataset_label,
            dataset_path=dataset_path,
        )
        return artifact.dataset

    dataset = generate_range_regression_dataset(
        sample_count,
        scenario_config=scenario_config,
        rng=rng,
    )
    if dataset_path is not None:
        save_range_regression_dataset(dataset_path, dataset=dataset, scenario_config=scenario_config)
    return dataset


def _validate_loaded_dataset_artifact(
    artifact: RangeRegressionDatasetArtifact,
    *,
    expected_sample_count: int,
    expected_scenario_config: RangeScenarioConfig,
    dataset_label: str,
    dataset_path: Path,
) -> None:
    if artifact.dataset.sample_count != expected_sample_count:
        raise ValueError(
            f"{dataset_label} dataset at {dataset_path} contains {artifact.dataset.sample_count} samples, "
            f"expected {expected_sample_count}"
        )
    if artifact.scenario_config is not None and artifact.scenario_config != expected_scenario_config:
        raise ValueError(
            f"{dataset_label} dataset at {dataset_path} was generated with a different scenario configuration"
        )


def _measurements_from_feature_sequence(
    feature_sequence: np.ndarray,
    *,
    position_scale: float,
    scenario_config: RangeScenarioConfig,
) -> tuple[BearingMeasurement, ...]:
    total_duration = scenario_config.time_step_seconds * max(feature_sequence.shape[0] - 1, 1)
    measurements: list[BearingMeasurement] = []
    for feature_row in feature_sequence:
        bearing_deg = math.degrees(math.atan2(float(feature_row[3]), float(feature_row[4]))) % 360.0
        measurements.append(
            BearingMeasurement(
                time_seconds=float(feature_row[0] * total_duration),
                ownship_x=float(feature_row[1] * position_scale),
                ownship_y=float(feature_row[2] * position_scale),
                bearing_deg=float(bearing_deg),
            )
        )
    return tuple(measurements)


def _epoch_checkpoint_path(output_path: Path, epoch: int) -> Path:
    return output_path.with_name(f"{output_path.stem}_epoch_{epoch:03d}{output_path.suffix}")


def _epoch_checkpoint_pattern(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_epoch_{{epoch:03d}}{output_path.suffix}")


def _track_to_velocity_series(
    track: np.ndarray,
    times: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    if track.shape[0] < 2:
        return tuple((0.0, 0.0) for _ in range(track.shape[0]))

    deltas = np.diff(times)
    step_velocity = np.diff(track, axis=0) / deltas[:, None]
    velocity = np.zeros_like(track, dtype=float)
    velocity[:-1] = step_velocity
    velocity[-1] = step_velocity[-1]
    return tuple((float(x), float(y)) for x, y in velocity)


if __name__ == "__main__":
    main()