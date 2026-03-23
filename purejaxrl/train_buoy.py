import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml

from buoy_env import BuoySearchEnv
from buoy_ppo import make_train as make_train_ppo
from buoy_ppo_rnn import make_train as make_train_ppo_rnn
from buoy_ppo_rnn_curiosity import make_train as make_train_ppo_rnn_curiosity
import evaluate_buoy as eval_buoy


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
        "ANNEAL_EXPLORE_REWARD": False,
        "EXPLORE_REWARD_START_SCALE": 1.0,
        "EXPLORE_REWARD_END_SCALE": 0.0,
        "EXPLORE_REWARD_ANNEAL_FRACTION": 1.0,
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
        "ANNEAL_EXPLORE_REWARD": False,
        "EXPLORE_REWARD_START_SCALE": 1.0,
        "EXPLORE_REWARD_END_SCALE": 0.0,
        "EXPLORE_REWARD_ANNEAL_FRACTION": 1.0,
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
        "ANNEAL_EXPLORE_REWARD": False,
        "EXPLORE_REWARD_START_SCALE": 1.0,
        "EXPLORE_REWARD_END_SCALE": 0.0,
        "EXPLORE_REWARD_ANNEAL_FRACTION": 1.0,
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


def _evaluate_checkpoint(run_dir, full_cfg, episodes, seed, policy_mode, device_pref):
    env_cfg = full_cfg.get("environment", {})
    algo_cfg = full_cfg.get("algorithm", {})

    env = BuoySearchEnv(env_cfg)
    network, is_rnn = eval_buoy._build_network(algo_cfg, env)

    obs_dim = int(env.observation_space(None).shape[0])
    normalize_obs_enabled = bool(algo_cfg.get("params", {}).get("NORMALIZE_OBS", False))
    obs_norm_stats = eval_buoy._load_obs_norm_stats(run_dir, obs_dim)
    if normalize_obs_enabled and obs_norm_stats is None:
        print(
            "Warning: NORMALIZE_OBS=true but obs_norm_stats.npz was not found. "
            "Checkpoint evaluation will run without observation normalization."
        )
    if (not normalize_obs_enabled) and obs_norm_stats is not None:
        print(
            "Warning: NORMALIZE_OBS=false but obs_norm_stats.npz exists. "
            "Applying saved observation normalization stats for checkpoint evaluation "
            "for compatibility with runs trained before normalization-flag fix."
        )

    device = eval_buoy._select_device(device_pref)

    init_key = jax.random.PRNGKey(seed)
    init_obs, _ = env.reset(init_key, None)

    if is_rnn:
        rnn_hidden_size = int(algo_cfg.get("params", {}).get("RNN_HIDDEN_SIZE", 128))
        init_hidden = eval_buoy.ScannedRNN.initialize_carry(1, rnn_hidden_size)
        init_obs_batch = jnp.zeros((1, 1, *init_obs.shape), dtype=init_obs.dtype)
        init_done_batch = jnp.zeros((1, 1), dtype=bool)
        init_params = network.init(
            jax.random.PRNGKey(seed + 1),
            init_hidden,
            (init_obs_batch, init_done_batch),
        )
    else:
        rnn_hidden_size = 1
        init_params = network.init(
            jax.random.PRNGKey(seed + 1),
            jnp.zeros_like(init_obs),
        )

    checkpoint_bytes = (run_dir / "checkpoint.msgpack").read_bytes()
    params = flax.serialization.from_bytes(init_params, checkpoint_bytes)
    params = jax.device_put(params, device)

    if obs_norm_stats is not None:
        obs_norm_stats = {
            "mean": jax.device_put(obs_norm_stats["mean"], device),
            "var": jax.device_put(obs_norm_stats["var"], device),
        }

    eval_fn = eval_buoy._build_eval_fn(
        env,
        network,
        params,
        is_rnn,
        obs_norm_stats,
        rnn_hidden_size,
        policy_mode,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), int(episodes))
    with jax.default_device(device):
        result = jax.device_get(eval_fn(keys))

    return {
        "episodic_return": eval_buoy._summarize(result["episodic_return"]),
        "episodic_length": eval_buoy._summarize(result["episodic_length"]),
        "success_rate": float(np.mean(np.asarray(result["success"], dtype=np.float32))),
        "out_of_bounds_rate": float(np.mean(np.asarray(result["out_of_bounds"], dtype=np.float32))),
        "timeout_rate": float(np.mean(np.asarray(result["timed_out"], dtype=np.float32))),
    }


def _render_checkpoint(run_dir, run_name, output_root, full_cfg):
    render_cfg = full_cfg.get("render", {})
    eval_cfg = full_cfg.get("evaluation", {})
    run_cfg = full_cfg.get("run", {})

    seed = int(render_cfg.get("seed", eval_cfg.get("seed", run_cfg.get("seed", 0))))
    policy_mode = str(render_cfg.get("policy_mode", eval_cfg.get("policy_mode", "stochastic")))
    num_renders = int(render_cfg.get("num_renders", 1))
    device = str(render_cfg.get("device", eval_cfg.get("device", "auto")))
    save_name = str(render_cfg.get("save_name", "render.gif"))
    save_path = run_dir / save_name

    visualize_script = Path(__file__).resolve().parent / "visualize_buoy.py"
    cmd = [
        sys.executable,
        str(visualize_script),
        "--run",
        run_name,
        "--output-dir",
        str(output_root),
        "--seed",
        str(seed),
        "--policy-mode",
        policy_mode,
        "--num-renders",
        str(max(1, num_renders)),
        "--device",
        device,
        "--save",
        str(save_path),
    ]

    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"Render command failed with exit code {completed.returncode}: {details}")

    generated_paths = [save_path] if num_renders <= 1 else [
        save_path.with_name(f"{save_path.stem}_{idx + 1:03d}{save_path.suffix}")
        for idx in range(num_renders)
    ]
    missing = [str(path) for path in generated_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Render command completed but expected files were not found: {missing}")

    return {
        "status": "ok",
        "save_path": str(save_path),
        "num_renders": int(num_renders),
        "seed": int(seed),
        "policy_mode": policy_mode,
        "device": device,
    }


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

    collector_metrics = {
        "global_step": int(np.asarray(metrics["global_step"][-1])),
        "episodic_return": float(np.asarray(metrics["episodic_return"][-1])),
        "episodic_length": float(np.asarray(metrics["episodic_length"][-1])),
        "success_rate": float(np.asarray(metrics["success_rate"][-1])),
        "out_of_bounds_rate": float(np.asarray(metrics["out_of_bounds_rate"][-1])),
        "timeout_rate": float(np.asarray(metrics["timeout_rate"][-1])),
    }
    collector_metrics_info = {
        "scope": "last_training_update",
        "population": "episodes_completed_during_that_update_only",
        "source": "on-policy collector during training",
    }

    eval_cfg = full_cfg.get("evaluation", {})
    eval_episodes = int(eval_cfg.get("episodes", 256))
    eval_seed = int(eval_cfg.get("seed", 0))
    eval_policy_mode = eval_cfg.get("policy_mode", "stochastic")
    eval_device = eval_cfg.get("device", "auto")

    try:
        checkpoint_eval = _evaluate_checkpoint(
            run_dir=run_dir,
            full_cfg=full_cfg,
            episodes=eval_episodes,
            seed=eval_seed,
            policy_mode=eval_policy_mode,
            device_pref=eval_device,
        )
        final_metrics = {
            "global_step": collector_metrics["global_step"],
            "episodic_return": float(checkpoint_eval["episodic_return"]["mean"]),
            "episodic_length": float(checkpoint_eval["episodic_length"]["mean"]),
            "success_rate": checkpoint_eval["success_rate"],
            "out_of_bounds_rate": checkpoint_eval["out_of_bounds_rate"],
            "timeout_rate": checkpoint_eval["timeout_rate"],
            "evaluation": {
                "episodes": eval_episodes,
                "seed": eval_seed,
                "policy_mode": eval_policy_mode,
                "device": eval_device,
                "episodic_return": checkpoint_eval["episodic_return"],
                "episodic_length": checkpoint_eval["episodic_length"],
            },
            "collector_metrics": collector_metrics,
            "collector_metrics_info": collector_metrics_info,
        }
    except Exception as exc:
        final_metrics = {
            **collector_metrics,
            "collector_metrics_info": collector_metrics_info,
            "evaluation_error": str(exc),
        }

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(full_cfg, sort_keys=False),
        encoding="utf-8",
    )

    render_cfg = full_cfg.get("render", {})
    render_enabled = bool(render_cfg.get("enabled", True))
    if render_enabled:
        try:
            render_info = _render_checkpoint(
                run_dir=run_dir,
                run_name=run_name,
                output_root=output_root,
                full_cfg=full_cfg,
            )
            final_metrics["render"] = render_info
        except Exception as exc:
            final_metrics["render"] = {
                "status": "failed",
                "error": str(exc),
            }
    else:
        final_metrics["render"] = {
            "status": "disabled",
        }

    (run_dir / "summary.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Run '{run_name}' finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {run_dir / 'summary.json'}")
    if final_metrics.get("render", {}).get("status") == "ok":
        print(f"Render: {final_metrics['render']['save_path']}")
    elif final_metrics.get("render", {}).get("status") == "disabled":
        print("Render: disabled by config")
    else:
        print(f"Warning: render generation failed ({final_metrics.get('render', {}).get('error', 'unknown error')})")
    if "collector_metrics" in final_metrics:
        print("Summary metrics come from post-training checkpoint evaluation; collector metrics are available under 'collector_metrics'.")
    elif "evaluation_error" in final_metrics:
        print(f"Warning: post-training checkpoint evaluation failed, summary contains collector metrics only ({final_metrics['evaluation_error']}).")

    if wandb_enabled:
        wandb.summary.update(final_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
