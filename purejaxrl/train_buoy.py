import argparse
import dataclasses
import json
from pathlib import Path

import flax.serialization
import jax
import numpy as np
import wandb
import yaml

from buoy_env import BuoySearchEnv
from buoy_ppo import make_train as make_train_ppo
from buoy_ppo_rnn import make_train as make_train_ppo_rnn
from buoy_ppo_rnn_curiosity import make_train as make_train_ppo_rnn_curiosity


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_ppo_cfg(overrides):
    cfg = {
        "LR": 3e-4,
        "NUM_ENVS": 256,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "HIDDEN_SIZES": [256, 256],
        "ANNEAL_LR": True,
        "NORMALIZE_OBS": False,
        "NORMALIZE_REWARD": False,
        "DEBUG": False,
    }
    cfg.update(overrides)
    return cfg


def _default_ppo_rnn_cfg(overrides):
    cfg = {
        "LR": 3e-4,
        "NUM_ENVS": 256,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "HIDDEN_SIZES": [128],
        "RNN_HIDDEN_SIZE": 128,
        "ANNEAL_LR": True,
        "NORMALIZE_OBS": False,
        "NORMALIZE_REWARD": False,
        "DEBUG": False,
    }
    cfg.update(overrides)
    return cfg


def _default_ppo_rnn_curiosity_cfg(overrides):
    cfg = {
        "LR": 3e-4,
        "CURIOSITY_LR": 3e-4,
        "CURIOSITY_COEF": 0.01,
        "CURIOSITY_OUTPUT_DIM": 64,
        "CURIOSITY_HIDDEN_SIZES": [256, 256],
        "CURIOSITY_ACTIVATION": "relu",
        "NUM_ENVS": 256,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "HIDDEN_SIZES": [128],
        "RNN_HIDDEN_SIZE": 128,
        "ANNEAL_LR": True,
        "NORMALIZE_OBS": False,
        "NORMALIZE_REWARD": False,
        "DEBUG": False,
    }
    cfg.update(overrides)
    return cfg


def _iter_children(node):
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, (tuple, list)):
        return list(node)
    if dataclasses.is_dataclass(node):
        return [getattr(node, field.name) for field in dataclasses.fields(node)]
    return []


def _extract_obs_norm_stats(node, obs_dim):
    stack = [node]
    seen = set()

    while stack:
        current = stack.pop()
        try:
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
        except TypeError:
            pass

        mean = getattr(current, "mean", None)
        var = getattr(current, "var", None)
        count = getattr(current, "count", None)
        if mean is not None and var is not None and count is not None:
            mean_np = np.asarray(mean)
            var_np = np.asarray(var)
            if mean_np.shape == (obs_dim,) and var_np.shape == (obs_dim,):
                return {
                    "mean": mean_np.astype(np.float32),
                    "var": var_np.astype(np.float32),
                    "count": np.asarray(count).astype(np.float64),
                }

        stack.extend(_iter_children(current))

    return None


def main():
    parser = argparse.ArgumentParser(description="Train buoy-search RL agent")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name override (defaults to config filename stem)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    full_cfg = _load_config(config_path)

    run_name = args.run_name or config_path.stem
    run_cfg = full_cfg.get("run", {})
    env_cfg = full_cfg.get("environment", {})
    algo_cfg = full_cfg.get("algorithm", {})
    algo_name = algo_cfg.get("name", "ppo_continuous")

    if algo_name == "ppo_continuous":
        ppo_cfg = _default_ppo_cfg(algo_cfg.get("params", {}))
        train_builder = make_train_ppo
    elif algo_name == "ppo_continuous_rnn":
        ppo_cfg = _default_ppo_rnn_cfg(algo_cfg.get("params", {}))
        train_builder = make_train_ppo_rnn
    elif algo_name == "ppo_continuous_rnn_curiosity":
        ppo_cfg = _default_ppo_rnn_curiosity_cfg(algo_cfg.get("params", {}))
        train_builder = make_train_ppo_rnn_curiosity
    else:
        raise ValueError(
            f"Unsupported algorithm '{algo_name}'. Supported: ['ppo_continuous', 'ppo_continuous_rnn', 'ppo_continuous_rnn_curiosity']"
        )

    seed = int(run_cfg.get("seed", 0))

    output_root = Path(run_cfg.get("output_dir", "runs")).resolve()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_cfg = full_cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "disabled")
    wandb_enabled = wandb_mode != "disabled"
    if wandb_enabled:
        wandb.init(
            project=wandb_cfg.get("project", "jaxbuoy"),
            entity=wandb_cfg.get("entity", None),
            mode=wandb_mode,
            name=run_name,
            tags=wandb_cfg.get("tags", ["jax", "ppo", "buoy"]),
            config=full_cfg,
        )

    rng = jax.random.PRNGKey(seed)
    train_fn = jax.jit(train_builder(ppo_cfg, env_cfg, wandb_enabled=wandb_enabled))
    out = train_fn(rng)
    out = jax.tree_util.tree_map(jax.device_get, out)

    train_state = out["runner_state"][0]
    checkpoint_path = run_dir / "checkpoint.msgpack"
    checkpoint_path.write_bytes(flax.serialization.to_bytes(train_state.params))

    metrics = out["metrics"]
    env_state = out["runner_state"][1]
    obs_dim = int(BuoySearchEnv(env_cfg).observation_space(None).shape[0])
    obs_norm_stats = _extract_obs_norm_stats(env_state, obs_dim)

    if obs_norm_stats is not None:
        np.savez(run_dir / "obs_norm_stats.npz", **obs_norm_stats)

    final_metrics = {
        "global_step": int(np.asarray(metrics["global_step"][-1])),
        "episodic_return": float(np.asarray(metrics["episodic_return"][-1])),
        "episodic_length": float(np.asarray(metrics["episodic_length"][-1])),
        "success_rate": float(np.asarray(metrics["success_rate"][-1])),
        "out_of_bounds_rate": float(np.asarray(metrics["out_of_bounds_rate"][-1])),
        "timeout_rate": float(np.asarray(metrics["timeout_rate"][-1])),
    }

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(full_cfg, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Run '{run_name}' finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {run_dir / 'summary.json'}")

    if wandb_enabled:
        wandb.summary.update(final_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
