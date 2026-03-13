import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from functools import partial
from gymnax.environments import spaces


@struct.dataclass
class BuoyEnvState:
    x: jnp.ndarray
    y: jnp.ndarray
    heading: jnp.ndarray
    buoy_x: jnp.ndarray
    buoy_y: jnp.ndarray
    step_count: jnp.ndarray
    visited: jnp.ndarray


class BuoySearchEnv:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.radius_m = float(self.cfg.get("radius_m", 100.0))
        self.max_steps = int(self.cfg.get("max_steps", 1200))
        self.dt = float(self.cfg.get("decision_dt_s", 1.0))

        self.action_mode = self.cfg.get("action_mode", "simplified_rudder")
        self.max_speed_mps = float(self.cfg.get("max_speed_mps", 2.0))
        self.constant_speed_mps = float(self.cfg.get("constant_speed_mps", 2.0))

        self.max_rudder_deg = float(self.cfg.get("max_rudder_deg", 45.0))
        self.rudder_turn_rate = float(self.cfg.get("rudder_turn_rate", 0.25))
        self.thruster_yaw_rate = float(self.cfg.get("thruster_yaw_rate", 0.6))

        self.find_range_m = float(self.cfg.get("find_range_m", 7.0))
        self.find_fov_deg = float(self.cfg.get("find_fov_deg", 90.0))

        self.step_reward = float(self.cfg.get("step_reward", 0.0))
        self.found_reward = float(self.cfg.get("found_reward", 100.0))
        self.out_of_bounds_penalty = float(self.cfg.get("out_of_bounds_penalty", 50.0))
        self.timeout_penalty = float(self.cfg.get("timeout_penalty", 10.0))

        self.use_visited = bool(self.cfg.get("use_visited_grid", True))
        self.grid_size = int(self.cfg.get("visited_grid_size", 64))
        self.sensor_range_m = float(self.cfg.get("sensor_range_m", 20.0))
        self.sensor_fov_deg = float(self.cfg.get("sensor_fov_deg", 90.0))
        self.explore_reward_per_cell = float(
            self.cfg.get("explore_reward_per_cell", 0.005)
        )
        self.directional_samples = int(self.cfg.get("directional_samples", 8))

        self._max_rudder_rad = jnp.deg2rad(self.max_rudder_deg)
        self._find_fov_half_rad = jnp.deg2rad(self.find_fov_deg / 2.0)
        self._sensor_fov_half_rad = jnp.deg2rad(self.sensor_fov_deg / 2.0)
        self._cell_size = (2.0 * self.radius_m) / self.grid_size

        coord = jnp.linspace(
            -self.radius_m + 0.5 * self._cell_size,
            self.radius_m - 0.5 * self._cell_size,
            self.grid_size,
        )
        self._grid_x, self._grid_y = jnp.meshgrid(coord, coord, indexing="xy")
        self._in_train_area_mask = (
            self._grid_x * self._grid_x + self._grid_y * self._grid_y
        ) <= (self.radius_m * self.radius_m)

    def _angle_wrap(self, angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _sample_buoy_position(self, key):
        k_r, k_theta = jax.random.split(key)
        r = self.radius_m * jnp.sqrt(jax.random.uniform(k_r, ()))
        theta = 2.0 * jnp.pi * jax.random.uniform(k_theta, ())
        return r * jnp.cos(theta), r * jnp.sin(theta)

    def _new_initial_state(self, key):
        k_heading, k_buoy = jax.random.split(key)
        heading = jax.random.uniform(k_heading, (), minval=-jnp.pi, maxval=jnp.pi)
        buoy_x, buoy_y = self._sample_buoy_position(k_buoy)

        visited = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.float32)
        state = BuoyEnvState(
            x=jnp.array(0.0, dtype=jnp.float32),
            y=jnp.array(0.0, dtype=jnp.float32),
            heading=heading.astype(jnp.float32),
            buoy_x=buoy_x.astype(jnp.float32),
            buoy_y=buoy_y.astype(jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            visited=visited,
        )
        if self.use_visited:
            updated, _ = self._update_visited(state.visited, state.x, state.y, state.heading)
            state = state.replace(visited=updated)
        return state

    def _visibility_mask(self, x, y, heading):
        dx = self._grid_x - x
        dy = self._grid_y - y
        dist = jnp.sqrt(dx * dx + dy * dy)
        bearing = jnp.arctan2(dy, dx)
        rel = self._angle_wrap(bearing - heading)
        return (
            self._in_train_area_mask
            & (dist <= self.sensor_range_m)
            & (jnp.abs(rel) <= self._sensor_fov_half_rad)
        )

    def _update_visited(self, visited, x, y, heading):
        visible = self._visibility_mask(x, y, heading)
        newly_seen = visible & (visited < 0.5)
        updated = jnp.where(visible, 1.0, visited)
        explore_reward = (
            jnp.sum(newly_seen.astype(jnp.float32)) * self.explore_reward_per_cell
        )
        return updated, explore_reward

    def _world_to_grid(self, x, y):
        gx = ((x + self.radius_m) / (2.0 * self.radius_m) * self.grid_size).astype(jnp.int32)
        gy = ((y + self.radius_m) / (2.0 * self.radius_m) * self.grid_size).astype(jnp.int32)
        gx = jnp.clip(gx, 0, self.grid_size - 1)
        gy = jnp.clip(gy, 0, self.grid_size - 1)
        return gx, gy

    def _directional_exploration_features(self, state):
        rel_angles = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
        sample_d = jnp.linspace(
            self._cell_size,
            self.sensor_range_m,
            self.directional_samples,
        )

        def score_for_direction(rel_angle):
            world_angle = state.heading + rel_angle
            px = state.x + sample_d * jnp.cos(world_angle)
            py = state.y + sample_d * jnp.sin(world_angle)
            inside = (px * px + py * py) <= (self.radius_m * self.radius_m)
            gx, gy = self._world_to_grid(px, py)
            values = state.visited[gy, gx] * inside.astype(jnp.float32)
            return jnp.sum(values) / (jnp.sum(inside.astype(jnp.float32)) + 1e-6)

        return jax.vmap(score_for_direction)(rel_angles)

    def _get_obs(self, state):
        base_obs = jnp.array(
            [
                state.x / self.radius_m,
                state.y / self.radius_m,
                state.heading / jnp.pi,
                state.step_count.astype(jnp.float32) / float(self.max_steps),
            ],
            dtype=jnp.float32,
        )
        if not self.use_visited:
            return base_obs
        directional = self._directional_exploration_features(state).astype(jnp.float32)
        return jnp.concatenate([base_obs, directional], axis=0)

    def _found_buoy(self, x, y, heading, buoy_x, buoy_y):
        dx = buoy_x - x
        dy = buoy_y - y
        dist = jnp.sqrt(dx * dx + dy * dy)
        bearing = jnp.arctan2(dy, dx)
        rel = self._angle_wrap(bearing - heading)
        return (dist <= self.find_range_m) & (jnp.abs(rel) <= self._find_fov_half_rad)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params=None):
        state = self._new_initial_state(key)
        obs = self._get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: BuoyEnvState, action, params=None):
        if self.action_mode == "simplified_rudder":
            steer_cmd = jnp.clip(action[0], 0.0, 1.0)
            rudder_angle = (2.0 * steer_cmd - 1.0) * self._max_rudder_rad
            next_heading = self._angle_wrap(state.heading + rudder_angle * self.rudder_turn_rate)
            speed = jnp.clip(self.constant_speed_mps, 0.0, self.max_speed_mps)
        else:
            clipped = jnp.clip(action, -1.0, 1.0)
            left = clipped[0]
            right = clipped[1]
            speed = jnp.clip(0.5 * (left + right) * self.max_speed_mps, 0.0, self.max_speed_mps)
            yaw = (right - left) * self.thruster_yaw_rate
            next_heading = self._angle_wrap(state.heading + yaw * self.dt)

        next_x = state.x + speed * jnp.cos(next_heading) * self.dt
        next_y = state.y + speed * jnp.sin(next_heading) * self.dt

        out_of_bounds = (next_x * next_x + next_y * next_y) > (self.radius_m * self.radius_m)
        found = self._found_buoy(next_x, next_y, next_heading, state.buoy_x, state.buoy_y)
        next_step_count = state.step_count + 1
        timed_out = next_step_count >= self.max_steps
        done = out_of_bounds | found | timed_out

        if self.use_visited:
            next_visited, explore_reward = self._update_visited(
                state.visited, next_x, next_y, next_heading
            )
        else:
            next_visited = state.visited
            explore_reward = jnp.array(0.0, dtype=jnp.float32)

        reward = (
            self.step_reward
            + explore_reward
            + found.astype(jnp.float32) * self.found_reward
            - out_of_bounds.astype(jnp.float32) * self.out_of_bounds_penalty
            - timed_out.astype(jnp.float32) * self.timeout_penalty
        )

        next_state = BuoyEnvState(
            x=next_x.astype(jnp.float32),
            y=next_y.astype(jnp.float32),
            heading=next_heading.astype(jnp.float32),
            buoy_x=state.buoy_x,
            buoy_y=state.buoy_y,
            step_count=next_step_count,
            visited=next_visited,
        )

        reset_key = jax.random.split(key, 2)[1]
        reset_state = self._new_initial_state(reset_key)
        next_obs = self._get_obs(next_state)
        reset_obs = self._get_obs(reset_state)

        state_out = jax.tree_util.tree_map(
            lambda current, reset: jnp.where(done, reset, current),
            next_state,
            reset_state,
        )
        obs_out = jnp.where(done, reset_obs, next_obs)

        info = {
            "success": found,
            "out_of_bounds": out_of_bounds,
            "timed_out": timed_out,
            "explore_reward": explore_reward,
            "x": next_x,
            "y": next_y,
            "heading": next_heading,
            "buoy_x": state.buoy_x,
            "buoy_y": state.buoy_y,
        }
        return obs_out, state_out, reward.astype(jnp.float32), done, info

    def observation_space(self, params=None):
        obs_dim = 4 + (8 if self.use_visited else 0)
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(obs_dim,),
            dtype=jnp.float32,
        )

    def action_space(self, params=None):
        if self.action_mode == "simplified_rudder":
            return spaces.Box(
                low=jnp.zeros((1,), dtype=jnp.float32),
                high=jnp.ones((1,), dtype=jnp.float32),
                shape=(1,),
                dtype=np.float32,
            )
        return spaces.Box(
            low=-jnp.ones((2,), dtype=jnp.float32),
            high=jnp.ones((2,), dtype=jnp.float32),
            shape=(2,),
            dtype=np.float32,
        )
