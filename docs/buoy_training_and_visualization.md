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

## 7) Troubleshooting

- **Run not found in visualization**: confirm `runs/<run_name>/` exists.
- **No W&B logs**: check `wandb.mode` is `online`.
- **Slow training**: reduce `TOTAL_TIMESTEPS`, `NUM_ENVS`, or `NUM_STEPS` in config for quick tests.
