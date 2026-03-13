import argparse
import time
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from buoy_env import BuoySearchEnv
from buoy_models import ActorCriticContinuous, ActorCriticContinuousRNN, ScannedRNN


def _load_run_dir(run_name, output_dir):
    run_path = Path(output_dir).resolve() / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    return run_path


def _build_network(algorithm_cfg, env):
    algo_name = algorithm_cfg.get("name", "ppo_continuous")
    params = algorithm_cfg.get("params", {})
    hidden_sizes = tuple(params.get("HIDDEN_SIZES", [256, 256]))
    activation = params.get("ACTIVATION", "tanh")
    action_dim = env.action_space(None).shape[0]
    if algo_name == "ppo_continuous_rnn":
        return (
            ActorCriticContinuousRNN(
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                rnn_hidden_size=int(params.get("RNN_HIDDEN_SIZE", 128)),
            ),
            True,
        )

    return (
        ActorCriticContinuous(
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ),
        False,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize one trained buoy-search episode")
    parser.add_argument("--run", required=True, help="Run name (folder under output_dir)")
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Root output directory used during training",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="Playback speed multiplier (higher is faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Episode seed for visualization",
    )
    args = parser.parse_args()

    run_dir = _load_run_dir(args.run, args.output_dir)
    full_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    env_cfg = full_cfg.get("environment", {})
    algo_cfg = full_cfg.get("algorithm", {})

    env = BuoySearchEnv(env_cfg)
    network, is_rnn = _build_network(algo_cfg, env)

    init_key = jax.random.PRNGKey(args.seed)
    init_obs, _ = env.reset(init_key, None)
    if is_rnn:
        rnn_hidden_size = int(algo_cfg.get("params", {}).get("RNN_HIDDEN_SIZE", 128))
        init_hidden = ScannedRNN.initialize_carry(1, rnn_hidden_size)
        init_obs_batch = jnp.zeros((1, 1, *init_obs.shape), dtype=init_obs.dtype)
        init_done_batch = jnp.zeros((1, 1), dtype=bool)
        init_params = network.init(
            jax.random.PRNGKey(args.seed + 1),
            init_hidden,
            (init_obs_batch, init_done_batch),
        )
    else:
        init_params = network.init(
            jax.random.PRNGKey(args.seed + 1),
            jnp.zeros_like(init_obs),
        )
    checkpoint_bytes = (run_dir / "checkpoint.msgpack").read_bytes()
    params = flax.serialization.from_bytes(init_params, checkpoint_bytes)

    key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(key, None)
    rnn_hidden = None
    last_done = False
    if is_rnn:
        rnn_hidden_size = int(algo_cfg.get("params", {}).get("RNN_HIDDEN_SIZE", 128))
        rnn_hidden = ScannedRNN.initialize_carry(1, rnn_hidden_size)

    xs = [float(state.x)]
    ys = [float(state.y)]
    headings = [float(state.heading)]
    buoy_x = float(state.buoy_x)
    buoy_y = float(state.buoy_y)
    visited_tracker = state.visited
    visited_frames = [np.asarray(visited_tracker)] if env.use_visited else None

    done = False
    found = False
    out_of_bounds = False
    timed_out = False

    while not done:
        if is_rnn:
            ac_in = (obs[jnp.newaxis, jnp.newaxis, :], jnp.array([[last_done]], dtype=bool))
            rnn_hidden, pi, _ = network.apply(params, rnn_hidden, ac_in)
            action = pi.mean().squeeze(axis=(0, 1))
        else:
            pi, _ = network.apply(params, obs)
            action = pi.mean()

        key, step_key = jax.random.split(key)
        obs, state, _, done_arr, info = env.step(step_key, state, action, None)

        xs.append(float(info["x"]))
        ys.append(float(info["y"]))
        headings.append(float(info["heading"]))
        buoy_x = float(info["buoy_x"])
        buoy_y = float(info["buoy_y"])

        if env.use_visited:
            visited_tracker, _ = env._update_visited(
                visited_tracker,
                info["x"],
                info["y"],
                info["heading"],
            )
            visited_frames.append(np.asarray(visited_tracker))

        done = bool(done_arr)
        last_done = done
        found = bool(info["success"])
        out_of_bounds = bool(info["out_of_bounds"])
        timed_out = bool(info["timed_out"])

    radius = float(env_cfg.get("radius_m", 100.0))

    if env.use_visited:
        fig, (ax_path, ax_map) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax_path = plt.subplots(figsize=(6, 6))
        ax_map = None

    ax_path.set_aspect("equal", "box")
    ax_path.set_xlim(-radius - 5, radius + 5)
    ax_path.set_ylim(-radius - 5, radius + 5)
    ax_path.set_title(f"Run: {args.run}")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")

    path_boundary = plt.Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.5)
    ax_path.add_patch(path_boundary)
    ax_path.scatter([buoy_x], [buoy_y], c="orange", s=40, label="Buoy")
    (trail_line,) = ax_path.plot([], [], c="tab:blue", linewidth=1.0, alpha=0.8, label="Boat path")
    boat_point = ax_path.scatter([xs[0]], [ys[0]], c="tab:blue", s=25)
    heading_line, = ax_path.plot([], [], c="tab:blue", linewidth=2.0)
    ax_path.legend(loc="upper right")

    if env.use_visited:
        ax_map.set_aspect("equal", "box")
        ax_map.set_xlim(-radius - 5, radius + 5)
        ax_map.set_ylim(-radius - 5, radius + 5)
        ax_map.set_title("Exploration Map")
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")

        map_boundary = plt.Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.5)
        ax_map.add_patch(map_boundary)
        visited_im = ax_map.imshow(
            visited_frames[0],
            extent=(-radius, radius, -radius, radius),
            origin="lower",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            alpha=0.9,
        )
        map_boat = ax_map.scatter([xs[0]], [ys[0]], c="tab:blue", s=25)
    else:
        visited_im = None
        map_boat = None

    dt = max(0.001, 0.05 / max(args.speed, 1e-3))
    heading_len = 5.0

    plt.ion()
    plt.show(block=False)
    for i in range(len(xs)):
        trail_line.set_data(xs[: i + 1], ys[: i + 1])
        boat_point.set_offsets(np.array([[xs[i], ys[i]]]))
        hx = xs[i] + heading_len * np.cos(headings[i])
        hy = ys[i] + heading_len * np.sin(headings[i])
        heading_line.set_data([xs[i], hx], [ys[i], hy])
        if env.use_visited:
            visited_im.set_data(visited_frames[i])
            map_boat.set_offsets(np.array([[xs[i], ys[i]]]))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(dt)

    outcome = "found buoy" if found else "out of bounds" if out_of_bounds else "timed out" if timed_out else "ended"
    print(f"Episode outcome: {outcome}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
