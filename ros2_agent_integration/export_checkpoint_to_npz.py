#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purejaxrl.buoy_env import BuoySearchEnv
from purejaxrl.buoy_models import ActorCriticContinuousRNN, ScannedRNN


def _build_network(cfg):
    env = BuoySearchEnv(cfg["environment"])
    algo = cfg["algorithm"]
    params = algo.get("params", {})

    if algo.get("name") not in ("ppo_continuous_rnn", "ppo_continuous_rnn_curiosity"):
        raise ValueError(
            f"This exporter currently supports RNN actor-critic only, got: {algo.get('name')}"
        )

    network = ActorCriticContinuousRNN(
        action_dim=int(env.action_space(None).shape[0]),
        hidden_sizes=tuple(params.get("HIDDEN_SIZES", [128])),
        activation=str(params.get("ACTIVATION", "tanh")),
        rnn_hidden_size=int(params.get("RNN_HIDDEN_SIZE", 128)),
    )

    return env, network


def _load_params(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    env, network = _build_network(cfg)

    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset(rng, None)

    rnn_hidden_size = int(cfg["algorithm"]["params"].get("RNN_HIDDEN_SIZE", 128))
    init_hidden = ScannedRNN.initialize_carry(1, rnn_hidden_size)
    init_obs = jnp.zeros((1, 1, *obs.shape), dtype=jnp.float32)
    init_done = jnp.zeros((1, 1), dtype=bool)
    init_params = network.init(jax.random.PRNGKey(1), init_hidden, (init_obs, init_done))

    checkpoint_bytes = (run_dir / "checkpoint.msgpack").read_bytes()
    params = flax.serialization.from_bytes(init_params, checkpoint_bytes)

    return cfg, params["params"], int(obs.shape[0]), int(env.action_space(None).shape[0])


def _to_np(x):
    return np.asarray(x, dtype=np.float32)


def _export_arrays(params_tree):
    return {
        "encoder_w": _to_np(params_tree["Dense_0"]["kernel"]),
        "encoder_b": _to_np(params_tree["Dense_0"]["bias"]),
        "gru_ir_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["ir"]["kernel"]),
        "gru_ir_b": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["ir"]["bias"]),
        "gru_iz_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["iz"]["kernel"]),
        "gru_iz_b": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["iz"]["bias"]),
        "gru_in_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["in"]["kernel"]),
        "gru_in_b": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["in"]["bias"]),
        "gru_hr_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["hr"]["kernel"]),
        "gru_hz_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["hz"]["kernel"]),
        "gru_hn_w": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["hn"]["kernel"]),
        "gru_hn_b": _to_np(params_tree["ScannedRNN_0"]["GRUCell_0"]["hn"]["bias"]),
        "actor_hidden_w": _to_np(params_tree["Dense_1"]["kernel"]),
        "actor_hidden_b": _to_np(params_tree["Dense_1"]["bias"]),
        "actor_out_w": _to_np(params_tree["Dense_2"]["kernel"]),
        "actor_out_b": _to_np(params_tree["Dense_2"]["bias"]),
        "log_std": _to_np(params_tree["log_std"]),
    }


def _build_metadata(run_name, cfg, obs_dim, action_dim):
    algo_params = cfg["algorithm"].get("params", {})
    env_cfg = cfg.get("environment", {})
    return {
        "run_name": run_name,
        "algorithm": cfg["algorithm"].get("name"),
        "activation": algo_params.get("ACTIVATION", "tanh"),
        "hidden_sizes": algo_params.get("HIDDEN_SIZES", [128]),
        "rnn_hidden_size": int(algo_params.get("RNN_HIDDEN_SIZE", 128)),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_mode": env_cfg.get("action_mode", "thruster"),
        "obs_layout": {
            "x_norm": True,
            "y_norm": True,
            "heading_norm_pi": True,
            "episode_time_norm": bool(env_cfg.get("include_episode_time_obs", True)),
            "center_distance_norm": bool(env_cfg.get("include_center_distance_obs", False)),
            "current_vx_vy": bool(env_cfg.get("include_current_obs", False)),
            "wind_vx_vy": bool(env_cfg.get("include_wind_obs", False)),
            "directional_exploration_8": bool(env_cfg.get("use_visited_grid", True)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained RNN buoy checkpoint to a NumPy .npz for minimal runtime inference."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/wind_entropy_curiosity_high"),
        help="Directory containing checkpoint.msgpack and config.yaml",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("ros2_agent_integration/model_assets/wind_entropy_curiosity_high_policy.npz"),
        help="Path for output .npz model",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("ros2_agent_integration/model_assets/wind_entropy_curiosity_high_metadata.json"),
        help="Path for output metadata .json",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg, params_tree, obs_dim, action_dim = _load_params(run_dir)
    arrays = _export_arrays(params_tree)
    meta = _build_metadata(run_dir.name, cfg, obs_dim, action_dim)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_model, **arrays)
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Exported model: {args.output_model}")
    print(f"Exported metadata: {args.output_meta}")


if __name__ == "__main__":
    main()
