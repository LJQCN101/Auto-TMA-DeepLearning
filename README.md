# Auto-TMA-DeepLearning

Deep-learning bearing-only target motion analysis with synthetic training, checkpointed transformer regressors, and visualization for numeric or image-derived observations.

## Scope
Implemented:
1. Synthetic bearing-only scenario generation with mixed easy and hard target behavior.
2. Range-over-time transformer regression with an auxiliary velocity head.
3. Two model families: a compact baseline encoder and a larger tokenizer-free Kronos-style regressor.
4. Dataset caching to `.npz`, final checkpoints, and per-epoch checkpoints.
5. Prediction and visualization from measurement JSON, cached validation samples, interactive console input, or image-detected bearing lines.
6. OpenCV line detection and reduction for image-based bearing extraction.

Open items:
1. Real-world labeled training data and checkpoint curation.
2. Better image-mode calibration and annotation ergonomics.
3. Broader deployment packaging beyond the current workstation setup.

## Setup
Use the requested environment:

Install pytorch: https://pytorch.org/get-started/locally/
Install the package into the same environment:

```bash
pip install -e .
```

## Training
The default CLI invocation now matches the large baseline run used for the shared baseline-versus-Kronos comparison: 2,000,000 training samples, 65,536 validation samples, a 512/8/8/1024 baseline encoder, batch size 4096, learning rate 3e-5, continuous ownship maneuvering, cached datasets under `outputs/datasets/`, and checkpoint output at `outputs/baseline_regression_large_2m.pt`.

Train the transformer regressor:

```bash
python -m auto_tma.deep_learning   --architecture baseline   --d-model 512   --num-heads 8   --num-layers 8   --ff-dim 1024   --dropout 0.1   --train-samples 2000000   --validation-samples 65536   --epochs 12   --batch-size 4096   --learning-rate 3e-5   --weight-decay 0.01   --velocity-loss-weight 0.5   --constant-target-fraction 0.25   --device cuda   --sequence-length 17   --time-step-seconds 30   --bearing-noise-std-deg 0.2   --continuous-ownship-maneuvering   --max-target-turn-deg-per-step 30   --max-ownship-turn-deg-per-step 30   --ownship-speed-std 0.15   --train-dataset-path outputs/datasets/train_dataset.npz   --validation-dataset-path outputs/datasets/validation_dataset.npz   --checkpoint-path outputs/baseline_regression_large_2m.pt
```

When `--checkpoint-path` is set, the trainer also writes per-epoch checkpoints next to the final checkpoint as `*_epoch_###.pt`.

## Prediction
Predict from a measurement JSON file:

```bash
python -m auto_tma.predict --measurements-path measurements.json --visualization-path outputs/prediction_from_file.png
```

The neural predictor requires at least the model window length of observations, which is 17 by default. If more observations are provided, it uses the latest fixed-length window.
The rendered geometry view also overlays a steady-course/steady-speed baseline derived from the original `auto_tma_python_simple` assumption, so each prediction can be compared against a simple constant-velocity fit.

Predict from a cached validation sample and overlay ground truth:

```bash
python -m auto_tma.predict --validation-dataset-path outputs/datasets/validation_dataset.npz --dataset-sample-index 0 --visualization-path outputs/prediction_from_validation.png
```

Negative values for `--dataset-sample-index` count backward from the end of the cached dataset.
Validation plots include all three tracks on the same geometry axes when available: the neural prediction, the steady-course baseline, and the ground-truth target trajectory.

Collect observations incrementally through the console:

```bash
python -m auto_tma.predict --interactive
```

The console path buffers observations until the full neural window is available, then updates the rendered prediction after each new observation.

Predict from image-derived bearing lines:

```bash
python -m auto_tma.predict --image-path XXX.jpg --visualization-path outputs/prediction_from_image.png
```

For image mode, use `--units-per-pixel` or a line-annotation JSON file when you can calibrate the image into real units.

## Model Notes
The deep-learning model consumes continuous TMA features:
`(normalized time, ownship x, ownship y, sin bearing, cos bearing, bearing delta)`.

The baseline encoder is the default deployment choice in this repo because it slightly outperformed the larger Kronos variant on the shared cached validation set while keeping the simpler architecture. The Kronos path remains available for larger-capacity experiments.

The network predicts the full range-to-ownship sequence over the active window and also predicts target velocity components at each step through an auxiliary head. Training minimizes range loss plus a configurable velocity loss.
