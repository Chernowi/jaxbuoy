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


def _load_obs_norm_stats(run_dir, obs_dim):
    stats_path = run_dir / "obs_norm_stats.npz"
    if not stats_path.exists():
        return None
    data = np.load(stats_path)
    mean = data.get("mean", None)
    var = data.get("var", None)
    if mean is None or var is None:
        return None
    if mean.shape != (obs_dim,) or var.shape != (obs_dim,):
        return None
    return {
        "mean": jnp.asarray(mean, dtype=jnp.float32),
        "var": jnp.asarray(var, dtype=jnp.float32),
    }


def _normalize_obs(obs, obs_norm_stats):
    if obs_norm_stats is None:
        return obs
    return (obs - obs_norm_stats["mean"]) / jnp.sqrt(obs_norm_stats["var"] + 1e-8)


def _select_device(device_pref):
    if device_pref == "cpu":
        for device in jax.devices():
            if device.platform == "cpu":
                return device
        raise RuntimeError("No CPU device available")

    if device_pref == "gpu":
        for device in jax.devices():
            if device.platform == "gpu":
                return device
        raise RuntimeError("GPU requested but no GPU device is available to JAX")

    for device in jax.devices():
        if device.platform == "gpu":
            return device
    return jax.devices()[0]


def _step_no_reset(env, state, action):
    if env.action_mode == "simplified_rudder":
        steer_cmd = jnp.clip(action[0], 0.0, 1.0)
        rudder_angle = (2.0 * steer_cmd - 1.0) * env._max_rudder_rad
        next_heading = env._angle_wrap(state.heading + rudder_angle * env.rudder_turn_rate)
        speed = jnp.clip(env.constant_speed_mps, 0.0, env.max_speed_mps)
    else:
        clipped = jnp.clip(action, -1.0, 1.0)
        left = clipped[0]
        right = clipped[1]
        speed = jnp.clip(0.5 * (left + right) * env.max_speed_mps, 0.0, env.max_speed_mps)
        yaw = (right - left) * env.thruster_yaw_rate
        next_heading = env._angle_wrap(state.heading + yaw * env.dt)

    next_x = state.x + speed * jnp.cos(next_heading) * env.dt
    next_y = state.y + speed * jnp.sin(next_heading) * env.dt

    out_of_bounds = (next_x * next_x + next_y * next_y) > (env.radius_m * env.radius_m)
    found = env._found_buoy(next_x, next_y, next_heading, state.buoy_x, state.buoy_y)
    next_step_count = state.step_count + 1
    timed_out = next_step_count >= env.max_steps
    done = out_of_bounds | found | timed_out

    if env.use_visited:
        next_visited, explore_reward = env._update_visited(
            state.visited, next_x, next_y, next_heading
        )
    else:
        next_visited = state.visited
        explore_reward = jnp.array(0.0, dtype=jnp.float32)

    reward = (
        env.step_reward
        + explore_reward
        + found.astype(jnp.float32) * env.found_reward
        - out_of_bounds.astype(jnp.float32) * env.out_of_bounds_penalty
        - timed_out.astype(jnp.float32) * env.timeout_penalty
    )

    next_state = state.replace(
        x=next_x.astype(jnp.float32),
        y=next_y.astype(jnp.float32),
        heading=next_heading.astype(jnp.float32),
        step_count=next_step_count,
        visited=next_visited,
    )
    next_obs = env._get_obs(next_state)

    return next_obs, next_state, reward.astype(jnp.float32), done, found, out_of_bounds, timed_out


def _build_rollout_fn(env, network, params, is_rnn, obs_norm_stats, seed, rnn_hidden_size):
    @jax.jit
    def _rollout(seed_value):
        key = jax.random.PRNGKey(seed_value)
        obs, state = env.reset(key, None)

        if is_rnn:
            rnn_hidden = ScannedRNN.initialize_carry(1, rnn_hidden_size)
        else:
            rnn_hidden = jnp.zeros((1, 1), dtype=jnp.float32)

        carry = (
            obs,
            state,
            key,
            jnp.array(False),
            jnp.array(False),
            jnp.array(False),
            jnp.array(False),
            jnp.array(0.0, dtype=jnp.float32),
            rnn_hidden,
        )

        def _scan_step(carry_in, _):
            (
                obs_t,
                state_t,
                key_t,
                done_t,
                found_t,
                oob_t,
                timeout_t,
                cum_reward_t,
                rnn_hidden_t,
            ) = carry_in

            def _active_step(args):
                (
                    obs_in,
                    state_in,
                    key_in,
                    _found_in,
                    _oob_in,
                    _timeout_in,
                    cum_reward_in,
                    rnn_hidden_in,
                ) = args
                policy_obs = _normalize_obs(obs_in, obs_norm_stats)
                if is_rnn:
                    ac_in = (
                        policy_obs[jnp.newaxis, jnp.newaxis, :],
                        jnp.array([[False]], dtype=bool),
                    )
                    rnn_hidden_out, pi, _ = network.apply(params, rnn_hidden_in, ac_in)
                    action = pi.mean().squeeze(axis=(0, 1))
                else:
                    pi, _ = network.apply(params, policy_obs)
                    action = pi.mean()
                    rnn_hidden_out = rnn_hidden_in

                key_out, _ = jax.random.split(key_in)
                next_obs, next_state, reward, next_done, found, oob, timed_out = _step_no_reset(
                    env, state_in, action
                )
                cum_reward_out = cum_reward_in + reward
                return (
                    next_obs,
                    next_state,
                    key_out,
                    next_done,
                    found,
                    oob,
                    timed_out,
                    cum_reward_out,
                    rnn_hidden_out,
                )

            def _inactive_step(args):
                (
                    obs_in,
                    state_in,
                    key_in,
                    found_in,
                    oob_in,
                    timeout_in,
                    cum_reward_in,
                    rnn_hidden_in,
                ) = args
                return (
                    obs_in,
                    state_in,
                    key_in,
                    jnp.array(True),
                    found_in,
                    oob_in,
                    timeout_in,
                    cum_reward_in,
                    rnn_hidden_in,
                )

            carry_out = jax.lax.cond(
                done_t,
                _inactive_step,
                _active_step,
                (
                    obs_t,
                    state_t,
                    key_t,
                    found_t,
                    oob_t,
                    timeout_t,
                    cum_reward_t,
                    rnn_hidden_t,
                ),
            )

            (
                obs_n,
                state_n,
                key_n,
                done_n,
                found_n,
                oob_n,
                timeout_n,
                cum_reward_n,
                rnn_hidden_n,
            ) = carry_out

            frame = (
                state_n.x,
                state_n.y,
                state_n.heading,
                state_n.visited,
                done_n,
                found_n,
                oob_n,
                timeout_n,
                cum_reward_n,
            )
            return (
                obs_n,
                state_n,
                key_n,
                done_n,
                found_n,
                oob_n,
                timeout_n,
                cum_reward_n,
                rnn_hidden_n,
            ), frame

        final_carry, frames = jax.lax.scan(
            _scan_step,
            carry,
            xs=None,
            length=env.max_steps,
        )

        return state, frames, final_carry

    return _rollout(seed)


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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="JAX device used for rollout (auto prefers GPU)",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=4,
        help="Render every Nth frame (higher is faster)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between frames to mimic real-time playback",
    )
    args = parser.parse_args()

    run_dir = _load_run_dir(args.run, args.output_dir)
    full_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    env_cfg = full_cfg.get("environment", {})
    algo_cfg = full_cfg.get("algorithm", {})

    env = BuoySearchEnv(env_cfg)
    network, is_rnn = _build_network(algo_cfg, env)
    obs_dim = int(env.observation_space(None).shape[0])
    normalize_obs_enabled = bool(algo_cfg.get("params", {}).get("NORMALIZE_OBS", False))
    obs_norm_stats = _load_obs_norm_stats(run_dir, obs_dim)
    if normalize_obs_enabled and obs_norm_stats is None:
        print(
            "Warning: this run was trained with NORMALIZE_OBS=true, but obs_norm_stats.npz "
            "was not found. Visualization policy inputs are unnormalized and may not match training behavior."
        )

    device = _select_device(args.device)
    print(f"Using JAX device: {device}")

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
    params = jax.device_put(params, device)
    if obs_norm_stats is not None:
        obs_norm_stats = {
            "mean": jax.device_put(obs_norm_stats["mean"], device),
            "var": jax.device_put(obs_norm_stats["var"], device),
        }

    rnn_hidden_size = int(algo_cfg.get("params", {}).get("RNN_HIDDEN_SIZE", 128))

    rollout_start = time.perf_counter()
    with jax.default_device(device):
        init_state, frames, _ = _build_rollout_fn(
            env,
            network,
            params,
            is_rnn,
            obs_norm_stats,
            args.seed,
            rnn_hidden_size,
        )
        init_state = jax.device_get(init_state)
        frames = jax.device_get(frames)
    rollout_time = time.perf_counter() - rollout_start

    x_hist, y_hist, heading_hist, visited_hist, done_hist, found_hist, oob_hist, timeout_hist, reward_hist = (
        frames
    )
    done_np = np.asarray(done_hist, dtype=bool)
    if np.any(done_np):
        end_idx = int(np.argmax(done_np)) + 1
    else:
        end_idx = int(len(done_np))

    xs = np.concatenate(
        [np.array([float(init_state.x)]), np.asarray(x_hist[:end_idx], dtype=np.float32)]
    )
    ys = np.concatenate(
        [np.array([float(init_state.y)]), np.asarray(y_hist[:end_idx], dtype=np.float32)]
    )
    headings = np.concatenate(
        [np.array([float(init_state.heading)]), np.asarray(heading_hist[:end_idx], dtype=np.float32)]
    )
    cumulative_rewards = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.asarray(reward_hist[:end_idx], dtype=np.float32)]
    )

    buoy_x = float(init_state.buoy_x)
    buoy_y = float(init_state.buoy_y)
    visited_frames = None
    if env.use_visited:
        visited_frames = [np.asarray(init_state.visited)]
        visited_frames.extend(np.asarray(visited_hist[:end_idx]))

    found = bool(np.asarray(found_hist[end_idx - 1])) if end_idx > 0 else False
    out_of_bounds = bool(np.asarray(oob_hist[end_idx - 1])) if end_idx > 0 else False
    timed_out = bool(np.asarray(timeout_hist[end_idx - 1])) if end_idx > 0 else False

    print(f"Rollout generated {len(xs)} frames in {rollout_time:.3f}s")

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
    reward_text = ax_path.text(
        0.02,
        0.98,
        "",
        transform=ax_path.transAxes,
        va="top",
        ha="left",
    )
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

    render_every = max(1, int(args.render_every))
    frame_indices = np.arange(0, len(xs), render_every, dtype=np.int32)
    if frame_indices[-1] != len(xs) - 1:
        frame_indices = np.concatenate([frame_indices, np.array([len(xs) - 1], dtype=np.int32)])

    dt = max(0.001, 0.05 / max(args.speed, 1e-3))
    dt *= render_every
    heading_len = 5.0

    plt.ion()
    plt.show(block=False)
    for i in frame_indices:
        trail_line.set_data(xs[: i + 1], ys[: i + 1])
        boat_point.set_offsets(np.array([[xs[i], ys[i]]]))
        hx = xs[i] + heading_len * np.cos(headings[i])
        hy = ys[i] + heading_len * np.sin(headings[i])
        heading_line.set_data([xs[i], hx], [ys[i], hy])
        reward_text.set_text(
            f"Step: {i:4d}\\nCumulative reward: {cumulative_rewards[i]:.3f}"
        )
        if env.use_visited:
            visited_im.set_data(visited_frames[i])
            map_boat.set_offsets(np.array([[xs[i], ys[i]]]))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if args.realtime:
            time.sleep(dt)

    outcome = "found buoy" if found else "out of bounds" if out_of_bounds else "timed out" if timed_out else "ended"
    print(f"Episode outcome: {outcome}")
    print(f"Cumulative reward: {cumulative_rewards[-1]:.3f}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
