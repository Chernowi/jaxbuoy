# Buoy Agent Training and Visualization Guide

This document explains how to train the buoy-search agent and visualize a trained run.

## 1) Prerequisites

From the repository root:

```bash
pip install -r requirements.txt
```

## 2) Configuration Files

Training is driven by a single YAML file that includes:
- run settings (`run`)
- algorithm settings (`algorithm`)
- environment settings (`environment`)
- Weights & Biases settings (`wandb`)

Available examples:
- `configs/buoy_rudder_ppo.yaml`
- `configs/buoy_thruster_ppo.yaml`

Environment observation options (under `environment`):
- `include_episode_time_obs`: adds normalized episode progress scalar (`step_count / max_steps`).
- `include_center_distance_obs`: adds normalized distance-to-center scalar.
- `include_current_obs`: adds normalized current vector components `[current_vx, current_vy]`.
- `include_wind_obs`: adds normalized wind vector components `[wind_vx, wind_vy]`.

Current and wind observations are normalized by `max_current_speed_mps` and
`max_wind_speed_mps` respectively (or kept at `0` when those maxima are `0`).

## 3) Train an Agent

Run from repository root:

```bash
python purejaxrl/train_buoy.py --config configs/buoy_rudder_ppo.yaml
```

or

```bash
python purejaxrl/train_buoy.py --config configs/buoy_thruster_ppo.yaml
```

### Run naming

By default, run name is the YAML filename stem:
- `configs/buoy_rudder_ppo.yaml` -> run name `buoy_rudder_ppo`
- `configs/buoy_thruster_ppo.yaml` -> run name `buoy_thruster_ppo`

You can override:

```bash
python purejaxrl/train_buoy.py --config configs/buoy_rudder_ppo.yaml --run-name my_custom_run
```

### Training outputs

Outputs are saved under `runs/<run_name>/`:
- `checkpoint.msgpack` (trained policy parameters)
- `config.yaml` (copied run configuration)
- `summary.json` (final metrics snapshot)

## 4) Weights & Biases Monitoring

Set in YAML under `wandb.mode`:
- `online`: logs to W&B
- `disabled`: no W&B logging

Example:

```yaml
wandb:
  mode: online
  project: jaxbuoy
  entity: null
```

## 5) Visualize a Trained Run

After training, replay one episode:

```bash
python purejaxrl/visualize_buoy.py --run buoy_rudder_ppo --speed 20
```

### Visualization options

- `--run`: run folder name inside `runs/`
- `--output-dir`: root output folder (default: `runs`)
- `--speed`: playback multiplier (higher = faster)
- `--seed`: episode seed for replay

Example with custom output directory and faster playback:

```bash
python purejaxrl/visualize_buoy.py --run my_custom_run --output-dir runs --speed 40 --seed 3
```

## 6) Typical Workflow

1. Pick a YAML config in `configs/`.
2. Train with `train_buoy.py`.
3. Verify `runs/<run_name>/checkpoint.msgpack` exists.
4. Replay with `visualize_buoy.py --run <run_name>`.

## 7) Deterministic spiral baseline and coverage-based episode time

You can evaluate a deterministic Archimedean-spiral policy to estimate how long it takes to visually cover the training area at agent speed.

```bash
python purejaxrl/evaluate_buoy.py \
  --run buoy_thruster_ppo \
  --policy-mode spiral \
  --episodes 100
```

The JSON output includes:
- `coverage_estimate.steps_to_target`
- `coverage_estimate.time_to_target_s`
- `coverage_estimate.suggested_max_steps`

Use these to set `environment.max_steps` in config (optionally with a safety factor using `--max-steps-margin`).

To compare trained vs deterministic spiral over the same seeds:

```bash
python purejaxrl/evaluate_buoy.py --run buoy_thruster_ppo --policy-mode deterministic --episodes 100
python purejaxrl/evaluate_buoy.py --run buoy_thruster_ppo --policy-mode spiral --episodes 100
```

To visualize the deterministic spiral policy:

```bash
python purejaxrl/visualize_buoy.py --run buoy_thruster_ppo --policy-mode spiral --speed 20
```

To tune spiral parameters interactively:

```bash
python purejaxrl/tune_spiral_policy.py --config configs/buoy_thruster_ppo_rnn_curiosity.yaml
```

Interactive tuner controls:
- `a [m]`, `b [m/rad]` (Archimedean spiral parameters)
- `lookahead`, `heading_gain`, `radial_gain`
- `thruster_forward` (used only in thruster mode)

The tuner shows trajectory, visited map, coverage %, and suggested `max_steps`.

## 8) Troubleshooting

- **Run not found in visualization**: confirm `runs/<run_name>/` exists.
- **No W&B logs**: check `wandb.mode` is `online`.
- **Slow training**: reduce `TOTAL_TIMESTEPS`, `NUM_ENVS`, or `NUM_STEPS` in config for quick tests.

## 9) Metric interpretation (important)

- **Collector metrics** (`train/*` in W&B and `collector_metrics` in `summary.json`) are computed from episodes that happened to finish during the latest training update window.
- **Evaluation metrics** (`episodic_return`, `success_rate`, `out_of_bounds_rate`, etc. at the top level of `summary.json`) are computed by fresh post-training rollouts from reset seeds.
- Collector metrics can look much better than evaluation because they are a narrower on-policy slice and can be biased by which episodes finish in that window.

Use evaluation metrics as the primary signal for comparing checkpoints or grid-search trials.

### Observation normalization consistency

- Training now respects `algorithm.params.NORMALIZE_OBS`.
- If `NORMALIZE_OBS=true`, normalization stats are saved to `obs_norm_stats.npz` and used by evaluation/visualization.
- For older checkpoints, evaluation/visualization automatically use `obs_norm_stats.npz` when available, even if config says `NORMALIZE_OBS=false`.
