import argparse
import sys
import time
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib

if any(arg == "--save" or arg.startswith("--save=") for arg in sys.argv):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FFMpegWriter, PillowWriter

from buoy_env import BuoySearchEnv
from buoy_models import ActorCriticContinuous, ActorCriticContinuousRNN, ScannedRNN
from spiral_policy import (
    build_spiral_params_from_args,
    rollout_spiral,
    spiral_params_dict,
    step_no_reset,
)


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
    if algo_name in ("ppo_continuous_rnn", "ppo_continuous_rnn_curiosity"):
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


def _build_rollout_fn(
    env,
    network,
    params,
    is_rnn,
    obs_norm_stats,
    seed,
    rnn_hidden_size,
    policy_mode,
):
    stochastic = policy_mode == "stochastic"

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
                    if stochastic:
                        key_out, action_key = jax.random.split(key_in)
                        action = pi.sample(seed=action_key).squeeze(axis=(0, 1))
                    else:
                        key_out = key_in
                        action = pi.mean().squeeze(axis=(0, 1))
                else:
                    pi, _ = network.apply(params, policy_obs)
                    if stochastic:
                        key_out, action_key = jax.random.split(key_in)
                        action = pi.sample(seed=action_key)
                    else:
                        key_out = key_in
                        action = pi.mean()
                    rnn_hidden_out = rnn_hidden_in

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


def _indexed_save_path(base_path, index, total):
    if total <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{index + 1:03d}{base_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Visualize one buoy-search episode (trained policy or deterministic spiral)")
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
        "--policy-mode",
        choices=["deterministic", "stochastic", "spiral"],
        default="stochastic",
        help="Action selection mode during visualization",
    )
    parser.add_argument("--spiral-a-m", type=float, default=0.0, help="Spiral parameter a [m]")
    parser.add_argument(
        "--spiral-b-m-per-rad",
        type=float,
        default=1.5,
        help="Spiral parameter b [m/rad]",
    )
    parser.add_argument(
        "--spiral-lookahead-rad",
        type=float,
        default=0.45,
        help="Lookahead angle for target point on spiral [rad]",
    )
    parser.add_argument(
        "--spiral-heading-gain",
        type=float,
        default=1.2,
        help="Heading error gain for spiral controller",
    )
    parser.add_argument(
        "--spiral-radial-gain",
        type=float,
        default=0.8,
        help="Radial error gain for spiral controller",
    )
    parser.add_argument(
        "--spiral-thruster-forward",
        type=float,
        default=0.9,
        help="Forward thruster command in spiral mode (thruster action mode only)",
    )
    parser.add_argument(
        "--num-renders",
        type=int,
        default=4,
        help="Number of episodes to render (uses consecutive seeds starting at --seed)",
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
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output file path (e.g. episode.mp4 or episode.gif) to save animation instead of opening a window",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS when using --save (default derives from speed and render-every)",
    )
    args = parser.parse_args()
    if args.num_renders < 1:
        raise ValueError("--num-renders must be >= 1")
    if args.num_renders > 1 and not args.save:
        raise ValueError("--num-renders > 1 requires --save to generate multiple output files")

    run_dir = _load_run_dir(args.run, args.output_dir)
    full_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    env_cfg = full_cfg.get("environment", {})
    algo_cfg = full_cfg.get("algorithm", {})

    env = BuoySearchEnv(env_cfg)
    spiral = build_spiral_params_from_args(args)

    if args.policy_mode != "spiral":
        network, is_rnn = _build_network(algo_cfg, env)
        obs_dim = int(env.observation_space(None).shape[0])
        normalize_obs_enabled = bool(algo_cfg.get("params", {}).get("NORMALIZE_OBS", False))
        obs_norm_stats = _load_obs_norm_stats(run_dir, obs_dim)
        if normalize_obs_enabled and obs_norm_stats is None:
            print(
                "Warning: this run was trained with NORMALIZE_OBS=true, but obs_norm_stats.npz "
                "was not found. Visualization policy inputs are unnormalized and may not match training behavior."
            )
        elif (not normalize_obs_enabled) and obs_norm_stats is not None:
            print(
                "Warning: config says NORMALIZE_OBS=false, but obs_norm_stats.npz exists. "
                "Applying saved normalization stats for compatibility with checkpoints trained "
                "with observation normalization enabled."
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
    else:
        network = None
        is_rnn = False
        obs_norm_stats = None
        params = None
        rnn_hidden_size = 1
        device = jax.devices("cpu")[0]
        print(f"Using deterministic spiral baseline with params: {spiral_params_dict(spiral)}")

    render_every = max(1, int(args.render_every))
    dt = max(0.001, 0.05 / max(args.speed, 1e-3))
    dt *= render_every
    default_fps = max(1.0, min(120.0, 1.0 / max(dt, 1e-6)))
    output_fps = float(args.fps) if args.fps is not None else default_fps
    radius = float(env_cfg.get("radius_m", 100.0))

    for render_idx in range(args.num_renders):
        episode_seed = args.seed + render_idx

        rollout_start = time.perf_counter()
        if args.policy_mode == "spiral":
            spiral_rollout = rollout_spiral(env, episode_seed, spiral, max_steps=env.max_steps)
            xs = np.asarray(spiral_rollout["x"], dtype=np.float32)
            ys = np.asarray(spiral_rollout["y"], dtype=np.float32)
            headings = np.asarray(spiral_rollout["heading"], dtype=np.float32)
            cumulative_rewards = np.asarray(spiral_rollout["cumulative_reward"], dtype=np.float32)
            init_state = spiral_rollout["state"]
            buoy_x = float(init_state.buoy_x)
            buoy_y = float(init_state.buoy_y)
            visited_frames = spiral_rollout["visited"] if env.use_visited else None
            found = bool(spiral_rollout["found"])
            out_of_bounds = bool(spiral_rollout["out_of_bounds"])
            timed_out = bool(spiral_rollout["timed_out"])
        else:
            with jax.default_device(device):
                init_state, frames, _ = _build_rollout_fn(
                    env,
                    network,
                    params,
                    is_rnn,
                    obs_norm_stats,
                    episode_seed,
                    rnn_hidden_size,
                    args.policy_mode,
                )
                init_state = jax.device_get(init_state)
                frames = jax.device_get(frames)

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
        rollout_time = time.perf_counter() - rollout_start

        print(
            f"Render {render_idx + 1}/{args.num_renders} (seed={episode_seed}) generated {len(xs)} frames in {rollout_time:.3f}s"
        )

        if env.use_visited:
            fig, (ax_path, ax_map) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            fig, ax_path = plt.subplots(figsize=(6, 6))
            ax_map = None

        ax_path.set_aspect("equal", "box")
        ax_path.set_xlim(-radius - 5, radius + 5)
        ax_path.set_ylim(-radius - 5, radius + 5)
        ax_path.set_title(f"Run: {args.run} | Seed: {episode_seed}")
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

        frame_indices = np.arange(0, len(xs), render_every, dtype=np.int32)
        if frame_indices[-1] != len(xs) - 1:
            frame_indices = np.concatenate([frame_indices, np.array([len(xs) - 1], dtype=np.int32)])

        heading_len = 5.0

        def _draw_frame(i):
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

        if args.save:
            base_save_path = Path(args.save).expanduser().resolve()
            save_path = _indexed_save_path(base_save_path, render_idx, args.num_renders)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ext = save_path.suffix.lower()
            print(
                f"Saving {len(frame_indices)} rendered frames to {save_path} at {output_fps:.1f} FPS"
            )

            if ext == ".gif":
                writer = PillowWriter(fps=output_fps)
            else:
                try:
                    writer = FFMpegWriter(fps=output_fps, codec="libx264", bitrate=2400)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "FFmpeg is required to save non-GIF videos. Install ffmpeg or save as .gif"
                    ) from exc

            save_start = time.perf_counter()
            with writer.saving(fig, str(save_path), dpi=100):
                for i in frame_indices:
                    _draw_frame(int(i))
                    writer.grab_frame()
            save_time = time.perf_counter() - save_start
            print(f"Saved visualization to: {save_path} ({save_time:.3f}s)")
            plt.close(fig)
        else:
            plt.ion()
            plt.show(block=False)
            for i in frame_indices:
                _draw_frame(int(i))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if args.realtime:
                    time.sleep(dt)

        outcome = "found buoy" if found else "out of bounds" if out_of_bounds else "timed out" if timed_out else "ended"
        print(f"Episode outcome: {outcome}")
        print(f"Cumulative reward: {cumulative_rewards[-1]:.3f}")

        if not args.save:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
