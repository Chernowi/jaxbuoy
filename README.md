# PureJaxRL (End-to-End RL Training in Pure Jax)

[<img src="https://img.shields.io/badge/license-Apache2.0-blue.svg">](https://github.com/luchris429/purejaxrl/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

PureJaxRL is a high-performance, end-to-end Jax Reinforcement Learning (RL) implementation. When running many agents in parallel on GPUs, our implementation is over 1000x faster than standard PyTorch RL implementations. Unlike other Jax RL implementations, we implement the *entire training pipeline in JAX*, including the environment. This allows us to get significant speedups through JIT compilation and by avoiding CPU-GPU data transfer. It also results in easier debugging because the system is fully synchronous. More importantly, this code allows you to use jax to `jit`, `vmap`, `pmap`, and `scan` entire RL training pipelines. With this, we can:

- 🏃 Efficiently run tons of seeds in parallel on one GPU
- 💻 Perform rapid hyperparameter tuning
- 🦎 Discover new RL algorithms with meta-evolution

For more details, visit the accompanying blog post: https://chrislu.page/blog/meta-disco/

This notebook walks through the basic usage: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

## CHECK OUT [RESOURCES.MD](https://github.com/luchris429/purejaxrl/blob/main/RESOURCES.md) to see github repos that are part of the Jax RL Ecosystem!

## Performance

Without vectorization, our implementation runs 10x faster than [CleanRL's PyTorch baselines](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), as shown in the single-thread performance plot.

Cartpole                   |  Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_seconds.png)  |  ![](docs/minatar_plot_seconds.png)


With vectorized training, we can train 2048 PPO agents in half the time it takes to train a single PyTorch PPO agent on a single GPU. The vectorized agent training allows for simultaneous training across multiple seeds, rapid hyperparameter tuning, and even evolutionary Meta-RL. 

Vectorised Cartpole        |  Vectorised Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_parallel.png)  |  ![](docs/minatar_plot_parallel.png)


## Code Philosophy

PureJaxRL is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), providing high-quality single-file implementations with research-friendly features. Like CleanRL, this is not a modular library and is not meant to be imported. The repository focuses on simplicity and clarity in its implementations, making it an excellent resource for researchers and practitioners.

## Installation

Install dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Example Usage

[`examples/walkthrough.ipynb`](https://github.com/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb) walks through the basic usage. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

[`examples/brax_minatar.ipynb`](https://github.com/luchris429/purejaxrl/blob/main/examples/brax_minatar.ipynb) walks through using PureJaxRL for Brax and MinAtar. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/brax_minatar.ipynb)

## Buoy Search Training (JaxBuoy)

This repository includes a single-agent buoy search framework with:

- Circular training area (100 m default radius)
- Two action modes: `simplified_rudder` and `thruster`
- Optional visited-grid exploration reward and directional exploration observations
- Optional environmental force observations via `include_current_obs` and `include_wind_obs`
- YAML-based run configuration (algorithm + environment + W&B in one file)
- W&B monitoring during training
- Lightweight episode visualization tool

### Train

```bash
python purejaxrl/train_buoy.py --config configs/buoy_thruster_ppo_rnn_curiosity_wind.yaml
```

Run name defaults to the YAML filename stem (for example, `buoy_rudder_ppo`).
Outputs are saved to `runs/<run_name>/`:

- `checkpoint.msgpack`
- `config.yaml`
- `summary.json`

### Grid Search (Buoy)

Run a hyperparameter grid search using the stable curiosity config as the base. Each trial is forced to a 15M-step training budget by default:

```bash
python purejaxrl/grid_search_buoy.py \
    --base-config configs/buoy_thruster_ppo.yaml
```

For a quick smoke test, limit the number of trials and avoid launching training:

```bash
python purejaxrl/grid_search_buoy.py \
    --base-config configs/buoy_rudder_ppo_rnn_curiosity_stable.yaml \
    --max-runs 2 \
    --dry-run
```

### Visualize

```bash
python purejaxrl/visualize_buoy.py --run buoy_thruster_ppo
```

Saved visualizations use one frame per simulated RL step at a fixed playback rate of 4 steps/second.

To save a rendered animation instead of opening a window:

```bash
python purejaxrl/visualize_buoy.py --run runs/fast_wind_100M --save outputs/episode.mp4
```

### Deterministic spiral baseline (coverage-time based)

Evaluate a deterministic Archimedean-spiral policy (same environment config as a trained run):

```bash
python purejaxrl/evaluate_buoy.py \
    --run buoy_thruster_ppo_rnn_curiosity_wind \
    --policy-mode spiral \
    --episodes 100
```

The output includes:
- baseline episode metrics (`episodic_return`, `success_rate`, etc.)
- `coverage_estimate.steps_to_target` and `coverage_estimate.time_to_target_s`
- `coverage_estimate.suggested_max_steps` (using `--max-steps-margin`)

Compare directly with the trained policy on the same seeds:

```bash
python purejaxrl/evaluate_buoy.py --run buoy_thruster_ppo_rnn_curiosity_wind --policy-mode deterministic --episodes 100
python purejaxrl/evaluate_buoy.py --run buoy_thruster_ppo --policy-mode spiral --episodes 100
```

Visualize a spiral episode:

```bash
python purejaxrl/visualize_buoy.py --run fast_wind_100M --policy-mode stochastic
```

Tune spiral parameters interactively with live trajectory and coverage feedback:

```bash
python purejaxrl/tune_spiral_policy.py --config configs/buoy_thruster_ppo_rnn_curiosity_wind.yaml
```

You can also load environment settings from an existing run:

```bash
python purejaxrl/tune_spiral_policy.py --run buoy_thruster_ppo
```

## Related Work

Check out the list of [RESOURCES](https://github.com/luchris429/purejaxrl/blob/main/RESOURCES.md) to see libraries that are closely related to PureJaxRL!

The following repositories and projects were pre-cursors to `purejaxrl`:

- [Model-Free Opponent Shaping](https://arxiv.org/abs/2205.01447) (ICML 2022) (https://github.com/luchris429/Model-Free-Opponent-Shaping)

- [Discovered Policy Optimisation](https://arxiv.org/abs/2210.05639) (NeurIPS 2022) (https://github.com/luchris429/discovered-policy-optimisation)

- [Adversarial Cheap Talk](https://arxiv.org/abs/2211.11030) (ICML 2023) (https://github.com/luchris429/adversarial-cheap-talk)

## Citation

If you use PureJaxRL in your work, please cite the following paper:

```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```
