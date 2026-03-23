# Exploration Grid and Reward Mechanics (Technical)

This document explains how exploration tracking and exploration reward are implemented for `BoatAgentYOLO`, including the exact update logic, equations, and runtime parameters.

---

## 1) Where it is implemented

- Agent integration and reward application:
  - `Assets/Scripts/Training/BoatAgentYOLO.cs`
- Grid + sector exploration model used by the agent:
  - `Assets/Scripts/Training/SectorExplorationHeatmap.cs`
- Legacy heatmap (currently not instantiated anywhere):
  - `Assets/Scripts/Training/ExplorationHeatmap.cs`

Only `SectorExplorationHeatmap` is currently constructed in code (`new SectorExplorationHeatmap(...)` in `BoatAgentYOLO.InitializeExplorationHeatmap`).

---

## 2) High-level behavior

At each agent action step (`OnActionReceived`):

1. The agent applies movement action.
2. Standard rewards are computed (time penalty, approach reward, YOLO detection reward).
3. If exploration tracking is enabled:
   - The camera frustum is projected onto a 2D XZ grid.
   - Visible cells receive incremental visibility mass.
   - The sum of newly added visibility this step (`newVisibility`) is returned.
   - Exploration reward is added only if `newVisibility` exceeds a threshold.
4. Sector visibility cache (8 directions around agent) is updated and later appended to observations.

Important distinction:

- **Exploration reward** uses **newly added visibility mass** (`newVisibility`).
- **Exploration observations** use **8 sector visibility values** (`cachedSectorVisibility[0..7]`).

They are related, but not identical signals.

---

## 3) Grid geometry

Let:

- `R = areaRadius`
- `N = gridResolution`
- `cellSize = (2R)/N`
- Grid origin in world XZ:
  - `originX = centerX - R`
  - `originZ = centerZ - R`

A cell `(x, y)` center in world coordinates is:

- `wx = originX + (x + 0.5) * cellSize`
- `wz = originZ + (y + 0.5) * cellSize`

The grid stores scalar visibility in a flat array:

- `heatmap[y * N + x]`, with value range `[0, maxVisibilityValue]` (currently max is passed as `1.0`).

---

## 4) Per-step visibility update algorithm

The update is performed by:

- `SectorExplorationHeatmap.UpdateVisibilityAndSectors(agentPos, agentForward, horizontalFOV)`

### 4.1 Candidate cell window

To avoid iterating the full grid, it computes a bounded window around the camera/agent cell using:

- `cellRange = ceil(maxVisibilityDistance / cellSize)`

Only cells in `[agentCellX ± cellRange, agentCellY ± cellRange]` are evaluated.

### 4.2 Geometric gating

For each candidate cell:

1. Compute vector from agent to cell center on XZ plane:
   - `dx = cellX - agentX`
   - `dz = cellZ - agentZ`
2. Compute squared distance:
   - `distSq = dx^2 + dz^2`
3. Reject if:
   - `distSq > maxVisibilityDistance^2`
   - `distSq < 0.01` (near-zero safeguard)
4. Normalize direction and perform FOV check via dot product:
   - `dot = normalized(cellDir) · normalized(agentForwardXZ)`
   - Accept only if `dot >= cos(horizontalFOV/2)`

This gives a circular range clip + forward cone clip.

### 4.3 Distance falloff weight

For accepted cells, weight is:

- `ratio = dist / maxVisibilityDistance`
- `w = (1 - ratio)^p`, where `p = visibilityFalloffExponent`

Special-case optimization is used for `p == 2` (`w = (1-ratio)^2`) to avoid `pow` cost.

### 4.4 Visibility accumulation

For each accepted cell with current visibility `v`:

- Raw increment: `delta = w * visibilityUpdateRate`
- Saturation clamp: `v_new = min(maxVisibilityValue, v + delta)`
- Effective increment added this step:
  - `delta_eff = v_new - v`

Then:

- `heatmap[idx] = v_new`
- `totalNewVis += delta_eff`

Return value from update:

- `newVisibility = totalNewVis`

This is exactly what the agent uses for exploration reward.

---

## 5) Exploration reward formula

In `BoatAgentYOLO.CalculateRewards()`:

```csharp
float newVisibility = explorationHeatmap.UpdateVisibilityAndSectors(...);
if (explorationRewardScale > 0f && newVisibility > explorationMinNoveltyForReward)
{
    AddReward(newVisibility * explorationRewardScale);
}
```

So exploration reward at a step is:

\[
r_{explore} =
\begin{cases}
\text{newVisibility} \cdot s, & \text{if } \text{newVisibility} > \tau \text{ and } s > 0\\
0, & \text{otherwise}
\end{cases}
\]

where:

- `s = explorationRewardScale`
- `τ = explorationMinNoveltyForReward`

### Practical implications

1. **Thresholded sparse shaping**
   - Tiny updates below `τ` produce zero exploration reward.
2. **Saturation naturally reduces reward over time**
   - Previously seen cells approach `maxVisibilityValue` and stop contributing.
3. **Reward magnitude scales with total newly seen mass, not count of sectors**
   - Reward is computed before/independent of sector averaging.

---

## 6) Sector observations (state input, not direct reward)

After updating grid visibility, the system updates 8 directional sector values around the agent:

- Forward
- Forward-Right
- Right
- Back-Right
- Back
- Back-Left
- Left
- Forward-Left

Each sector stores a weighted average of nearby cell visibility (distance-weighted up to `sectorMaxDistance`).

These 8 values are appended to observations in `CollectObservations(...)` when exploration tracking is enabled.

### Why this matters

The policy receives local directional “already explored” structure while reward is generated from raw new visibility increments. This combination encourages discovering unseen space while preserving directional memory in observation space.

---

## 7) Metrics logged at episode end

When an episode ends and exploration tracking is active, the agent logs:

- `exploration/progress` = normalized average visibility over cells inside the circular training area
- `exploration/total_visibility_added`
- `Environment/ExplorationProgress`
- Per-sector final values:
  - `exploration/sector_forward`
  - `exploration/sector_forward_right`
  - `exploration/sector_right`
  - `exploration/sector_back_right`
  - `exploration/sector_back`
  - `exploration/sector_back_left`
  - `exploration/sector_left`
  - `exploration/sector_forward_left`

---

## 8) Current serialized parameter sets found in repository

There are multiple `BoatAgentYOLO` instances with different exploration settings.

### A) Prefab baseline
File: `Assets/Boats/simple_blueboat_no_antenna.prefab`

- `enableExplorationTracking: 1`
- `explorationGridResolution: 32`
- `explorationMaxVisibilityDistance: 40`
- `explorationVisibilityFalloff: 2`
- `explorationUpdateRate: 0.1`
- `explorationRewardScale: 0.001`
- `explorationMinNoveltyForReward: 0.01`

### B) Scene instance 1 (YOLO-focused)
File: `Assets/OutdoorsScene.unity` (first `BoatAgentYOLO` block)

- `detectionSource: YOLO (1)`
- `maxEpisodeTime: 180`
- `enableExplorationTracking: 1`
- `explorationGridResolution: 32`
- `explorationMaxVisibilityDistance: 20`
- `explorationVisibilityFalloff: 4`
- `explorationUpdateRate: 0.5`
- `explorationRewardScale: 0.005`
- `explorationMinNoveltyForReward: 0.001`

### C) Scene instance 2 (long-horizon sparse setup)
File: `Assets/OutdoorsScene.unity` (second `BoatAgentYOLO` block)

- `MaxStep: 60000`, `maxEpisodeTime: 1200`
- `enableExplorationTracking: 1`
- `explorationGridResolution: 64`
- `explorationMaxVisibilityDistance: 15`
- `explorationVisibilityFalloff: 2`
- `explorationUpdateRate: 0.1`
- `explorationRewardScale: 0.001`
- `explorationMinNoveltyForReward: 0.01`

---

## 9) Interaction with curiosity in trainer config

Your active trainer config (`config/YOLO/sparse_reward_experiments/ppo_curiosity_lstm_optimized_visor.yaml`) also enables intrinsic curiosity (`reward_signals.curiosity`).

So the policy can receive both:

1. **Environment-side exploration shaping** (this document: Unity C# reward from grid visibility).
2. **Trainer-side intrinsic reward** (ICM curiosity from ML-Agents trainer).

These are additive at training time and should be tuned jointly.

---

## 10) Tuning guidance (implementation-aware)

1. If exploration reward dominates behavior:
   - Decrease `explorationRewardScale`
   - Increase `explorationMinNoveltyForReward`
   - Or decrease curiosity `strength` in YAML

2. If exploration reward is too sparse/weak:
   - Increase `explorationUpdateRate`
   - Increase `explorationMaxVisibilityDistance`
   - Decrease `explorationMinNoveltyForReward`

3. If policy overfits near-field sweeping:
   - Increase `visibilityFalloffExponent` to bias near vs far contributions more strongly
   - Consider lowering `maxVisibilityDistance` if reward is too easy to farm from stationary scanning

4. If training becomes expensive:
   - Lower `explorationGridResolution` (quadratic impact in memory, linear/quasi-linear per-step depending on cell window)

---

## 11) Summary equation set

Per accepted cell at distance `d`:

\[
w(d) = \left(1 - \frac{d}{d_{max}}\right)^p
\]

\[
\Delta v = \min\left(v_{max} - v,\; w(d) \cdot \eta\right)
\]

Step visibility gain:

\[
\text{newVisibility} = \sum_{i \in \text{visible cells}} \Delta v_i
\]

Exploration reward:

\[
r_{explore} = \mathbb{1}[\text{newVisibility} > \tau] \cdot \text{newVisibility} \cdot s
\]

with:

- `d_max = explorationMaxVisibilityDistance`
- `p = explorationVisibilityFalloff`
- `η = explorationUpdateRate`
- `τ = explorationMinNoveltyForReward`
- `s = explorationRewardScale`
