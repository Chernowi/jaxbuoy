import argparse
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
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


def _build_eval_fn(
    env,
    network,
    params,
    is_rnn,
    obs_norm_stats,
    rnn_hidden_size,
    policy_mode,
):
    stochastic = policy_mode == "stochastic"

    def _rollout_single(key):
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
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(False),
            jnp.array(False),
            rnn_hidden,
        )

        def _scan_step(carry_in, _):
            (
                obs_t,
                state_t,
                key_t,
                done_t,
                cum_reward_t,
                length_t,
                found_t,
                oob_t,
                timeout_t,
                rnn_hidden_t,
            ) = carry_in

            def _active_step(args):
                (
                    obs_in,
                    state_in,
                    key_in,
                    cum_reward_in,
                    length_in,
                    _found_in,
                    _oob_in,
                    _timeout_in,
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
                return (
                    next_obs,
                    next_state,
                    key_out,
                    next_done,
                    cum_reward_in + reward,
                    length_in + 1,
                    found,
                    oob,
                    timed_out,
                    rnn_hidden_out,
                )

            def _inactive_step(args):
                (
                    obs_in,
                    state_in,
                    key_in,
                    cum_reward_in,
                    length_in,
                    found_in,
                    oob_in,
                    timeout_in,
                    rnn_hidden_in,
                ) = args
                return (
                    obs_in,
                    state_in,
                    key_in,
                    jnp.array(True),
                    cum_reward_in,
                    length_in,
                    found_in,
                    oob_in,
                    timeout_in,
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
                    cum_reward_t,
                    length_t,
                    found_t,
                    oob_t,
                    timeout_t,
                    rnn_hidden_t,
                ),
            )

            return carry_out, ()

        final_carry, _ = jax.lax.scan(_scan_step, carry, xs=None, length=env.max_steps)
        (
            _obs,
            _state,
            _key,
            _done,
            cum_reward,
            episode_length,
            found,
            out_of_bounds,
            timed_out,
            _h,
        ) = final_carry

        return {
            "episodic_return": cum_reward,
            "episodic_length": episode_length.astype(jnp.float32),
            "success": found.astype(jnp.float32),
            "out_of_bounds": out_of_bounds.astype(jnp.float32),
            "timed_out": timed_out.astype(jnp.float32),
        }

    return jax.jit(jax.vmap(_rollout_single))


def _summarize(values):
    values_np = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained buoy policy without visualization")
    parser.add_argument("--run", required=True, help="Run name (folder under output_dir)")
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Root output directory used during training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for evaluation episodes",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["deterministic", "stochastic"],
        default="stochastic",
        help="Action selection mode: deterministic uses policy mean, stochastic samples policy",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="JAX device used for evaluation (auto prefers GPU)",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save aggregate evaluation summary as JSON",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")

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
            "was not found. Evaluation policy inputs are unnormalized and may not match training behavior."
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
        rnn_hidden_size = 1
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

    eval_fn = _build_eval_fn(
        env,
        network,
        params,
        is_rnn,
        obs_norm_stats,
        rnn_hidden_size,
        args.policy_mode,
    )

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    with jax.default_device(device):
        result = jax.device_get(eval_fn(keys))

    summary = {
        "run": args.run,
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "policy_mode": args.policy_mode,
        "episodic_return": _summarize(result["episodic_return"]),
        "episodic_length": _summarize(result["episodic_length"]),
        "success_rate": float(np.mean(np.asarray(result["success"], dtype=np.float32))),
        "out_of_bounds_rate": float(np.mean(np.asarray(result["out_of_bounds"], dtype=np.float32))),
        "timeout_rate": float(np.mean(np.asarray(result["timed_out"], dtype=np.float32))),
    }

    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        out_path = Path(args.save_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
