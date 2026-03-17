from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class SpiralParams:
    a_m: float = 0.0
    b_m_per_rad: float = 1.5
    lookahead_rad: float = 0.45
    heading_gain: float = 1.2
    radial_gain: float = 0.8
    thruster_forward: float = 0.9


def angle_wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_archimedean_waypoints(
    env,
    spiral: SpiralParams,
    speed_mps: float = 2.0,
):
    radius_limit = float(env.radius_m) - 1e-3
    if radius_limit <= 0.0:
        return np.asarray([[0.0, 0.0]], dtype=np.float32), float(speed_mps)

    b = max(float(spiral.b_m_per_rad), 1e-6)

    theta_start = max(0.0, -float(spiral.a_m) / b)
    theta_max = max(theta_start, (radius_limit - float(spiral.a_m)) / b)

    if theta_max <= theta_start:
        theta_dense = np.asarray([theta_start], dtype=np.float32)
    else:
        theta_dense_step = 0.01
        n_dense = int(np.ceil((theta_max - theta_start) / theta_dense_step)) + 1
        theta_dense = np.linspace(theta_start, theta_max, n_dense, dtype=np.float32)

    r_dense = np.clip(float(spiral.a_m) + b * theta_dense, 0.0, radius_limit)
    x_dense = r_dense * np.cos(theta_dense)
    y_dense = r_dense * np.sin(theta_dense)

    dx = np.diff(x_dense)
    dy = np.diff(y_dense)
    seg_lengths = np.sqrt(dx * dx + dy * dy)
    cumulative_s = np.concatenate(
        [np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lengths, dtype=np.float32)]
    )

    total_s = float(cumulative_s[-1]) if cumulative_s.size > 0 else 0.0
    ds = max(float(speed_mps) * float(env.dt), 1e-6)

    if total_s <= 1e-6:
        sample_s = np.asarray([0.0], dtype=np.float32)
    else:
        n_samples = int(np.floor(total_s / ds)) + 1
        sample_s = np.arange(n_samples, dtype=np.float32) * np.float32(ds)
        if sample_s[-1] < np.float32(total_s):
            sample_s = np.concatenate([sample_s, np.asarray([total_s], dtype=np.float32)])

    x_values = np.interp(sample_s, cumulative_s, x_dense).astype(np.float32)
    y_values = np.interp(sample_s, cumulative_s, y_dense).astype(np.float32)
    waypoints = np.stack([x_values, y_values], axis=1).astype(np.float32)

    if waypoints.shape[0] == 0:
        waypoints = np.asarray([[0.0, 0.0]], dtype=np.float32)

    return waypoints, float(speed_mps)


def _spiral_target_xy(state, env, spiral: SpiralParams, waypoints, speed_mps: float):
    x = float(state.x)
    y = float(state.y)
    radius_now = np.sqrt(x * x + y * y)
    step_idx = int(max(0, float(state.step_count)))

    ds = max(float(env.dt) * max(float(speed_mps), 1e-6), 1e-6)
    lookahead_distance = max(radius_now, 1.0) * max(float(spiral.lookahead_rad), 0.0)
    lookahead_steps = int(
        np.round(lookahead_distance / ds)
    )
    target_idx = int(np.clip(step_idx + lookahead_steps, 0, waypoints.shape[0] - 1))

    target_x, target_y = waypoints[target_idx]
    return float(target_x), float(target_y), target_idx


def spiral_action(state, env, spiral: SpiralParams, waypoints=None, speed_mps: float = 2.0):
    if waypoints is None:
        waypoints, speed_mps = build_archimedean_waypoints(env, spiral, speed_mps=speed_mps)

    target_x, target_y, _ = _spiral_target_xy(
        state, env, spiral, waypoints=waypoints, speed_mps=speed_mps
    )

    x = float(state.x)
    y = float(state.y)
    heading = float(state.heading)

    desired_heading = np.arctan2(target_y - y, target_x - x)
    heading_error = angle_wrap(desired_heading - heading)

    r_now = np.sqrt(x * x + y * y)
    target_r = np.sqrt(target_x * target_x + target_y * target_y)
    radial_error = target_r - r_now

    turn_cmd = (
        float(spiral.heading_gain) * heading_error
        + float(spiral.radial_gain) * (radial_error / max(float(env.radius_m), 1e-6))
    )

    if env.action_mode == "simplified_rudder":
        max_heading_change = float(env._max_rudder_rad) * float(env.rudder_turn_rate)
        if max_heading_change <= 1e-6:
            steer = 0.5
        else:
            steer = 0.5 + 0.5 * np.clip(turn_cmd / max_heading_change, -1.0, 1.0)
        return jnp.asarray([np.clip(steer, 0.0, 1.0)], dtype=jnp.float32)

    yaw_scale = float(env.thruster_yaw_rate) * float(env.dt)
    if yaw_scale <= 1e-6:
        diff = 0.0
    else:
        diff = np.clip(turn_cmd / yaw_scale, -2.0, 2.0)

    forward = np.clip(float(spiral.thruster_forward), 0.0, 1.0)
    left = np.clip(forward - 0.5 * diff, -1.0, 1.0)
    right = np.clip(forward + 0.5 * diff, -1.0, 1.0)
    return jnp.asarray([left, right], dtype=jnp.float32)


def step_no_reset(
    env,
    state,
    action,
    ignore_buoy: bool = False,
    ignore_timeout: bool = False,
):
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

    drift_vx, drift_vy = env._boat_drift_velocity(state)
    buoy_drift_vx, buoy_drift_vy = env._buoy_drift_velocity(state)

    next_x = state.x + (speed * jnp.cos(next_heading) + drift_vx) * env.dt
    next_y = state.y + (speed * jnp.sin(next_heading) + drift_vy) * env.dt
    next_buoy_x = state.buoy_x + buoy_drift_vx * env.dt
    next_buoy_y = state.buoy_y + buoy_drift_vy * env.dt

    out_of_bounds = (next_x * next_x + next_y * next_y) > (env.radius_m * env.radius_m)
    if ignore_buoy:
        found = jnp.array(False)
    else:
        found = env._found_buoy(next_x, next_y, next_heading, next_buoy_x, next_buoy_y)
    next_step_count = state.step_count + 1
    timed_out = jnp.array(False) if ignore_timeout else (next_step_count >= env.max_steps)
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
        buoy_x=next_buoy_x.astype(jnp.float32),
        buoy_y=next_buoy_y.astype(jnp.float32),
        step_count=next_step_count,
        visited=next_visited,
    )
    next_obs = env._get_obs(next_state)

    return next_obs, next_state, reward.astype(jnp.float32), done, found, out_of_bounds, timed_out


def visited_fraction(env, visited):
    if not env.use_visited:
        return 0.0
    visited_mask = np.asarray(visited) >= 0.5
    area_mask = np.asarray(env._in_train_area_mask)
    denom = np.maximum(np.sum(area_mask), 1)
    return float(np.sum(visited_mask & area_mask) / denom)


def rollout_spiral(
    env,
    seed: int,
    spiral: SpiralParams,
    max_steps: Optional[int] = None,
    include_buoy: bool = True,
    ignore_timeout: bool = False,
):
    if max_steps is not None:
        max_steps = int(max_steps)
    elif not ignore_timeout:
        max_steps = int(env.max_steps)
    waypoints, speed_mps = build_archimedean_waypoints(env, spiral)
    key = jnp.array([0, 0], dtype=jnp.uint32)
    key = key.at[0].set(np.uint32(seed & 0xFFFFFFFF))
    obs, state = env.reset(key, None)

    x_hist: List[float] = [float(state.x)]
    y_hist: List[float] = [float(state.y)]
    buoy_x_hist: List[float] = [float(state.buoy_x)]
    buoy_y_hist: List[float] = [float(state.buoy_y)]
    heading_hist: List[float] = [float(state.heading)]
    reward_hist: List[float] = [0.0]
    visited_hist: List[np.ndarray] = [np.asarray(state.visited)] if env.use_visited else []

    cum_reward = 0.0
    done = False
    found = False
    out_of_bounds = False
    timed_out = False

    rollout_step = 0
    while True:
        if max_steps is not None and rollout_step >= max_steps:
            break

        action = spiral_action(
            state,
            env,
            spiral,
            waypoints=waypoints,
            speed_mps=speed_mps,
        )
        _, state, reward, done_jnp, found_jnp, oob_jnp, timeout_jnp = step_no_reset(
            env,
            state,
            action,
            ignore_buoy=not bool(include_buoy),
            ignore_timeout=ignore_timeout,
        )

        cum_reward += float(reward)
        x_hist.append(float(state.x))
        y_hist.append(float(state.y))
        buoy_x_hist.append(float(state.buoy_x))
        buoy_y_hist.append(float(state.buoy_y))
        heading_hist.append(float(state.heading))
        reward_hist.append(cum_reward)

        if env.use_visited:
            visited_hist.append(np.asarray(state.visited))

        done = bool(done_jnp)
        found = bool(found_jnp)
        out_of_bounds = bool(oob_jnp)
        timed_out = bool(timeout_jnp)
        rollout_step += 1

        if done:
            break

    steps = len(x_hist) - 1
    coverage = visited_fraction(env, state.visited)
    return {
        "x": np.asarray(x_hist, dtype=np.float32),
        "y": np.asarray(y_hist, dtype=np.float32),
        "buoy_x": np.asarray(buoy_x_hist, dtype=np.float32),
        "buoy_y": np.asarray(buoy_y_hist, dtype=np.float32),
        "heading": np.asarray(heading_hist, dtype=np.float32),
        "cumulative_reward": np.asarray(reward_hist, dtype=np.float32),
        "visited": visited_hist,
        "steps": int(steps),
        "time_s": float(steps * float(env.dt)),
        "done": bool(done),
        "found": bool(found),
        "out_of_bounds": bool(out_of_bounds),
        "timed_out": bool(timed_out),
        "coverage": float(coverage),
        "state": state,
        "obs": obs,
        "waypoints": np.asarray(waypoints, dtype=np.float32),
    }


def estimate_spiral_coverage_time(
    env,
    seed: int,
    spiral: SpiralParams,
    coverage_target: float = 0.995,
    max_steps: Optional[int] = None,
):
    max_steps = int(max_steps) if max_steps is not None else int(env.max_steps)
    waypoints, speed_mps = build_archimedean_waypoints(env, spiral)
    key = jnp.array([0, 0], dtype=jnp.uint32)
    key = key.at[0].set(np.uint32(seed & 0xFFFFFFFF))
    _, state = env.reset(key, None)

    reached_step = None
    done = False
    for step in range(max_steps + 1):
        coverage = visited_fraction(env, state.visited)
        if coverage >= coverage_target:
            reached_step = step
            break

        if step == max_steps:
            break

        action = spiral_action(
            state,
            env,
            spiral,
            waypoints=waypoints,
            speed_mps=speed_mps,
        )
        _, state, _, done_jnp, _, _, _ = step_no_reset(env, state, action)
        done = bool(done_jnp)
        if done:
            break

    final_coverage = visited_fraction(env, state.visited)
    return {
        "coverage_target": float(coverage_target),
        "reached": reached_step is not None,
        "steps_to_target": int(reached_step) if reached_step is not None else None,
        "time_to_target_s": (float(reached_step) * float(env.dt)) if reached_step is not None else None,
        "final_coverage": float(final_coverage),
        "ended_early": bool(done),
    }


def build_spiral_params_from_args(args):
    return SpiralParams(
        a_m=float(args.spiral_a_m),
        b_m_per_rad=float(args.spiral_b_m_per_rad),
        lookahead_rad=float(args.spiral_lookahead_rad),
        heading_gain=float(args.spiral_heading_gain),
        radial_gain=float(args.spiral_radial_gain),
        thruster_forward=float(args.spiral_thruster_forward),
    )


def spiral_params_dict(spiral: SpiralParams) -> Dict[str, float]:
    return {
        "a_m": float(spiral.a_m),
        "b_m_per_rad": float(spiral.b_m_per_rad),
        "lookahead_rad": float(spiral.lookahead_rad),
        "heading_gain": float(spiral.heading_gain),
        "radial_gain": float(spiral.radial_gain),
        "thruster_forward": float(spiral.thruster_forward),
    }
