# ROS2 Agent Inference Integration (Minimal Runtime)

This folder is a drop-in bundle to integrate the trained buoy agent policy into a ROS2 control stack with minimal edge dependencies.

## What is included

- `export_checkpoint_to_npz.py`: one-time exporter from JAX/Flax checkpoint to NumPy weights.
- `model_assets/`: exported model and metadata for deployment.
- `buoy_agent_inference/`: ROS2 Python package (`ament_python`) with:
  - `minimal_policy.py`: pure NumPy RNN actor inference.
  - `inference_node.py`: ROS2 node that subscribes obs and publishes actions.
  - launch/config files for immediate use.

Runtime dependency on edge is only `numpy` (+ ROS2 core Python deps like `rclpy`).

## Current target model

Configured for `runs/wind_entropy_curiosity_high`.

- Observation dim: `12`
- Action dim: `2` (thruster left/right commands)
- RNN hidden size: `128`
- Action bounds expected by environment: `[-1, 1]`

## 1) Export model assets (one-time on training machine)

From repo root:

```bash
python ros2_agent_integration/export_checkpoint_to_npz.py \
  --run-dir runs/wind_entropy_curiosity_high \
  --output-model ros2_agent_integration/model_assets/wind_entropy_curiosity_high_policy.npz \
  --output-meta ros2_agent_integration/model_assets/wind_entropy_curiosity_high_metadata.json
```

## 2) Copy package into ROS2 workspace

Copy `ros2_agent_integration/buoy_agent_inference` into your ROS2 workspace `src/`.

## 3) Build and run

```bash
cd <your_ros2_ws>
colcon build --packages-select buoy_agent_inference
source install/setup.bash
ros2 launch buoy_agent_inference inference.launch.py
```

## ROS2 I/O contract

- Subscribe observations on `std_msgs/Float32MultiArray` topic `/agent/observation`
  - Must contain exactly 12 float values ordered as training obs.
- Publish actions on `std_msgs/Float32MultiArray` topic `/agent/action`
  - 2 floats: `[left_thruster, right_thruster]`
- Optional reset on `std_msgs/Bool` topic `/agent/reset_rnn`
  - `true` resets GRU hidden state at episode boundaries.

## Observation preprocessing (full spec)

This is the most important part of integration: the model is only as good as the observation vector you feed it.

For `runs/wind_entropy_curiosity_high`, the policy expects **exactly 12 floats** in this order:

1. `x_norm`
2. `y_norm`
3. `heading_norm`
4. `time_norm`
5. `dir_explore_0`
6. `dir_explore_1`
7. `dir_explore_2`
8. `dir_explore_3`
9. `dir_explore_4`
10. `dir_explore_5`
11. `dir_explore_6`
12. `dir_explore_7`

### 1) Coordinate frame and units

Use one consistent local 2D frame for all terms (`x`, `y`, heading, visited grid).

`x` and `y` are measured **with respect to the center of the training/search area**.
That center is the origin `(0, 0)`, where the episode starts in the original environment.

- Position unit: meters.
- Heading unit: radians.
- Training world radius: `radius_m = 50.0`.
- Episode length: `max_steps = 400`.

If your stack uses ENU/NED/body frames, convert once and keep this policy input in a fixed world frame.

### 2) Base features (indices 0..3)

Use:

- `x_norm = x / radius_m`
- `y_norm = y / radius_m`
- `heading_norm = wrap_pi(heading) / pi`
- `time_norm = step_count / max_steps`

So if the vessel is at the area center, `x_norm = 0` and `y_norm = 0`.
At the boundary along +X, `x_norm ~= +1`; along -Y boundary, `y_norm ~= -1`.

Where:

- `wrap_pi(a) = (a + pi) mod (2*pi) - pi`
- `step_count` starts at `0` and increments once per control decision.

Recommended safety clipping before publish:

- `x_norm, y_norm` to `[-1.2, 1.2]` (defensive against sensor spikes)
- `heading_norm` to `[-1.0, 1.0]`
- `time_norm` to `[0.0, 1.0]`

### 3) Directional exploration features (indices 4..11)

These 8 values encode how much area has been previously observed in 8 directions around current heading.

They are **not** direct sensor readings. You must maintain an internal `visited` grid map and compute features from it.

Training settings used by this model:

- `visited_grid_size = 64`
- `sensor_range_m = 10.0`
- `sensor_fov_deg = 90.0`
- `directional_samples = 8`

#### 3.1 Maintain visited grid

At each control step:

1. Build visibility mask from current pose:
   - For each grid cell center `(gx, gy)` in world coordinates:
     - `dist = sqrt((gx-x)^2 + (gy-y)^2)`
     - `bearing = atan2(gy-y, gx-x)`
     - `rel = wrap_pi(bearing - heading)`
     - Cell visible if:
       - inside training circle (`gx^2 + gy^2 <= radius_m^2`)
       - `dist <= sensor_range_m`
       - `abs(rel) <= sensor_fov_deg/2` in radians
2. Set `visited[cell] = 1.0` for all visible cells (never decay/reset during episode).

Reset `visited` to zeros when episode/mission resets.

#### 3.2 Compute 8 directional scores

For each relative direction angle:

- `rel_angles = [0, 45, 90, 135, 180, 225, 270, 315] deg` (i.e. `linspace(0, 2*pi, 8, endpoint=False)`)
- Build sample distances along ray:
  - `sample_d = linspace(cell_size, sensor_range_m, directional_samples)`
  - `cell_size = (2*radius_m)/visited_grid_size`

For each angle `a`:

1. `world_angle = heading + a`
2. For each distance `d` in `sample_d`:
   - `px = x + d*cos(world_angle)`
   - `py = y + d*sin(world_angle)`
   - Check `inside = (px^2 + py^2 <= radius_m^2)`
   - Map `(px, py)` to grid indices `(ix, iy)` in `[0, grid_size-1]`
   - Read `visited[iy, ix]` if inside, else 0
3. Direction score:
   - `score = sum(values) / (count_inside + 1e-6)`

Each score should be in `[0, 1]`.

### 4) Observation vector assembly

Final vector (dtype `float32`):

```text
obs = [
  x_norm,
  y_norm,
  heading_norm,
  time_norm,
  dir_explore_0,
  dir_explore_1,
  dir_explore_2,
  dir_explore_3,
  dir_explore_4,
  dir_explore_5,
  dir_explore_6,
  dir_explore_7,
]
```

Publish as `Float32MultiArray` length 12 on `/agent/observation`.

### 5) Heading conventions that usually break deployments

Common failure: wrong heading convention, causing the policy to turn incorrectly.

Before deployment, verify:

- Heading increases CCW in the chosen world frame.
- `heading=0` points along +X of the same frame used for `(x, y)`.
- `atan2(dy, dx)` uses the same axis order.
- Heading is wrapped to `[-pi, pi]` before normalization.

If your robot uses another convention, convert once before feature generation.

### 6) RNN state lifecycle

This policy is recurrent (GRU hidden size 128).

- Keep hidden state between consecutive steps of the same episode.
- Publish `Bool(data=true)` on `/agent/reset_rnn` when an episode/mission resets.
- Also reset your `step_count` and `visited` map at the same time.

Desynchronizing these resets is a major source of unstable behavior.

### 7) Thruster action postprocessing

Node output is `[left_thruster, right_thruster]`.

- Model is trained with action domain `[-1, 1]`.
- Node can clamp to this range (`clip_actions: true` default).
- If hardware expects another range (e.g. PWM), remap downstream:
  - `u_hw = remap(action, [-1,1] -> [u_min,u_max])`

### 8) Minimal ROS2 implementation checklist

When integrating into your ROS2 project, implement these blocks in order:

1. Pose/state estimator -> provide `(x, y, heading)` in consistent world frame.
2. Episode manager -> maintain `step_count` and reset events.
3. Visibility mapper -> update `visited` grid each step.
4. Feature builder -> compute 8 directional exploration scores.
5. Observation publisher -> publish 12-D float vector.
6. Action adapter -> consume 2-D action and map to actuators.

### 9) Quick sanity tests before sea trials

- Confirm vector length is always 12.
- Confirm no `NaN`/`inf` enters `/agent/observation`.
- Confirm directional features stay in `[0,1]`.
- Confirm reset event simultaneously resets:
  - RNN hidden state
  - visited grid
  - step counter
- Log first 100 observations/actions to verify smooth, bounded behavior.

## Notes

- Curiosity network is not needed for policy inference and is intentionally omitted.
- Node defaults to deterministic policy output; set `deterministic: false` to sample actions.
- Keep hidden state reset synchronized with your mission/episode reset logic.
