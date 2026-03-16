import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Button, Slider

from buoy_env import BuoySearchEnv
from spiral_policy import (
    SpiralParams,
    build_archimedean_waypoints,
    estimate_spiral_coverage_time,
    rollout_spiral,
    spiral_params_dict,
)


def _load_env_cfg_from_run(run_name, output_dir):
    run_path = Path(output_dir).expanduser().resolve() / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    full_cfg = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    return full_cfg.get("environment", {})


def _load_env_cfg_from_config(config_path):
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    full_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "environment" in full_cfg:
        return full_cfg.get("environment", {})
    return full_cfg


def _build_spiral_from_sliders(sliders):
    return SpiralParams(
        a_m=float(sliders["a_m"].val),
        b_m_per_rad=float(sliders["b_m_per_rad"].val),
        lookahead_rad=float(sliders["lookahead_rad"].val),
        heading_gain=float(sliders["heading_gain"].val),
        radial_gain=float(sliders["radial_gain"].val),
        thruster_forward=float(sliders["thruster_forward"].val),
    )


def _load_spiral_params_from_yaml(path: Path):
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = payload.get("spiral", payload)
    return SpiralParams(
        a_m=float(data.get("a_m", 0.0)),
        b_m_per_rad=float(data.get("b_m_per_rad", 1.5)),
        lookahead_rad=float(data.get("lookahead_rad", 0.45)),
        heading_gain=float(data.get("heading_gain", 1.2)),
        radial_gain=float(data.get("radial_gain", 0.8)),
        thruster_forward=float(data.get("thruster_forward", 0.9)),
    )


def _save_spiral_params_to_yaml(path: Path, spiral: SpiralParams):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spiral": spiral_params_dict(spiral)}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Archimedean-spiral tuning for buoy-area coverage"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="Path to config YAML with environment section")
    src.add_argument("--run", type=str, help="Run directory name under --output-dir")

    parser.add_argument("--output-dir", default="runs", help="Root output directory for --run")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for buoy spawn and heading")
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=None,
        help="Max simulation steps for tuning rollout (default: environment max_steps)",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.995,
        help="Visited-area fraction target for max-time estimate",
    )
    parser.add_argument(
        "--max-steps-margin",
        type=float,
        default=1.1,
        help="Safety multiplier to suggest max_steps from target-coverage estimate",
    )

    parser.add_argument("--spiral-a-m", type=float, default=0.0)
    parser.add_argument("--spiral-b-m-per-rad", type=float, default=1.5)
    parser.add_argument("--spiral-lookahead-rad", type=float, default=0.45)
    parser.add_argument("--spiral-heading-gain", type=float, default=1.2)
    parser.add_argument("--spiral-radial-gain", type=float, default=0.8)
    parser.add_argument("--spiral-thruster-forward", type=float, default=0.9)
    parser.add_argument(
        "--spiral-params-file",
        type=str,
        default="configs/spiral_params.yaml",
        help="YAML file used to load/save spiral parameters",
    )

    args = parser.parse_args()

    if args.config:
        env_cfg = _load_env_cfg_from_config(args.config)
        source_label = Path(args.config).name
    else:
        env_cfg = _load_env_cfg_from_run(args.run, args.output_dir)
        source_label = f"run:{args.run}"

    env = BuoySearchEnv(env_cfg)
    sim_steps = int(args.sim_steps) if args.sim_steps is not None else int(env.max_steps)
    coverage_target = float(np.clip(args.coverage_target, 0.0, 1.0))
    margin = max(1.0, float(args.max_steps_margin))

    initial_spiral = SpiralParams(
        a_m=float(args.spiral_a_m),
        b_m_per_rad=float(args.spiral_b_m_per_rad),
        lookahead_rad=float(args.spiral_lookahead_rad),
        heading_gain=float(args.spiral_heading_gain),
        radial_gain=float(args.spiral_radial_gain),
        thruster_forward=float(args.spiral_thruster_forward),
    )
    spiral_params_path = Path(args.spiral_params_file).expanduser().resolve()
    yaml_spiral = _load_spiral_params_from_yaml(spiral_params_path)
    if yaml_spiral is not None:
        initial_spiral = yaml_spiral

    fig, (ax_path, ax_map) = plt.subplots(1, 2, figsize=(13, 7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.34, wspace=0.20)

    radius = float(env.radius_m)
    for ax in (ax_path, ax_map):
        ax.set_aspect("equal", "box")
        ax.set_xlim(-radius - 5.0, radius + 5.0)
        ax.set_ylim(-radius - 5.0, radius + 5.0)
        circle = plt.Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.4)
        ax.add_patch(circle)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    ax_path.set_title("Spiral Trajectory")
    ax_map.set_title("Visited Map")

    (preview_spiral_line,) = ax_path.plot(
        [],
        [],
        c="tab:gray",
        linewidth=1.2,
        linestyle="--",
        label="archimedean waypoints",
    )
    (path_line,) = ax_path.plot([], [], c="tab:blue", linewidth=1.6, label="generated path")
    boat_point = ax_path.scatter([], [], c="tab:blue", s=26)
    buoy_point = ax_path.scatter([], [], c="orange", s=45)
    heading_line, = ax_path.plot([], [], c="tab:blue", linewidth=2.2)
    ax_path.legend(loc="upper right")

    visited_im = ax_map.imshow(
        np.zeros((env.grid_size, env.grid_size), dtype=np.float32),
        extent=(-radius, radius, -radius, radius),
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=0.9,
    )
    map_boat = ax_map.scatter([], [], c="tab:blue", s=26)
    map_buoy = ax_map.scatter([], [], c="orange", s=45)
    buoy_point.set_visible(False)
    map_buoy.set_visible(False)

    info_text = fig.text(0.08, 0.94, "", ha="left", va="top", fontsize=10)

    slider_axes = {
        "a_m": plt.axes([0.12, 0.27, 0.78, 0.022]),
        "b_m_per_rad": plt.axes([0.12, 0.24, 0.78, 0.022]),
        "lookahead_rad": plt.axes([0.12, 0.21, 0.78, 0.022]),
        "heading_gain": plt.axes([0.12, 0.18, 0.78, 0.022]),
        "radial_gain": plt.axes([0.12, 0.15, 0.78, 0.022]),
        "thruster_forward": plt.axes([0.12, 0.12, 0.78, 0.022]),
    }

    sliders = {
        "a_m": Slider(slider_axes["a_m"], "a [m]", -10.0, radius, valinit=initial_spiral.a_m),
        "b_m_per_rad": Slider(
            slider_axes["b_m_per_rad"],
            "b [m/rad]",
            0.1,
            max(8.0, radius / 2.0),
            valinit=initial_spiral.b_m_per_rad,
        ),
        "lookahead_rad": Slider(
            slider_axes["lookahead_rad"],
            "lookahead [rad]",
            0.05,
            2.5,
            valinit=initial_spiral.lookahead_rad,
        ),
        "heading_gain": Slider(
            slider_axes["heading_gain"],
            "heading_gain",
            0.1,
            4.0,
            valinit=initial_spiral.heading_gain,
        ),
        "radial_gain": Slider(
            slider_axes["radial_gain"],
            "radial_gain",
            0.0,
            5.0,
            valinit=initial_spiral.radial_gain,
        ),
        "thruster_forward": Slider(
            slider_axes["thruster_forward"],
            "thruster_forward",
            0.0,
            1.0,
            valinit=initial_spiral.thruster_forward,
        ),
    }

    reset_ax = plt.axes([0.12, 0.05, 0.18, 0.04])
    reset_btn = Button(reset_ax, "Reset sliders")

    print_ax = plt.axes([0.34, 0.05, 0.22, 0.04])
    print_btn = Button(print_ax, "Print CLI params")

    generate_ax = plt.axes([0.58, 0.05, 0.14, 0.04])
    generate_btn = Button(generate_ax, "Generate")

    set_params_ax = plt.axes([0.74, 0.05, 0.18, 0.04])
    set_params_btn = Button(set_params_ax, "Set Params")

    def _update_preview(_=None):
        spiral = _build_spiral_from_sliders(sliders)
        waypoints, _ = build_archimedean_waypoints(env, spiral)
        preview_spiral_line.set_data(waypoints[:, 0], waypoints[:, 1])

        info_text.set_text(
            f"source={source_label} | action_mode={env.action_mode} | dt={env.dt:.2f}s | speed_max={env.max_speed_mps:.2f}m/s\n"
            f"preview updated | waypoints={len(waypoints)} | click 'Generate' to rollout an episode | params_file={spiral_params_path}"
        )
        fig.canvas.draw_idle()

    def _generate(_event):
        spiral = _build_spiral_from_sliders(sliders)
        rollout = rollout_spiral(
            env,
            args.seed,
            spiral,
            max_steps=sim_steps,
            include_buoy=False,
        )
        estimate = estimate_spiral_coverage_time(
            env,
            seed=args.seed,
            spiral=spiral,
            coverage_target=coverage_target,
            max_steps=sim_steps,
        )

        xs = rollout["x"]
        ys = rollout["y"]
        headings = rollout["heading"]
        visited = rollout["visited"][-1] if env.use_visited and rollout["visited"] else np.zeros((env.grid_size, env.grid_size))

        path_line.set_data(xs, ys)
        boat_point.set_offsets(np.array([[xs[-1], ys[-1]]]))

        heading_len = 5.0
        hx = xs[-1] + heading_len * np.cos(headings[-1])
        hy = ys[-1] + heading_len * np.sin(headings[-1])
        heading_line.set_data([xs[-1], hx], [ys[-1], hy])

        visited_im.set_data(np.asarray(visited))
        map_boat.set_offsets(np.array([[xs[-1], ys[-1]]]))

        suggested_max_steps = None
        if estimate["steps_to_target"] is not None:
            suggested_max_steps = int(np.ceil(estimate["steps_to_target"] * margin))

        outcome = (
            "found buoy"
            if rollout["found"]
            else "out of bounds"
            if rollout["out_of_bounds"]
            else "timed out"
            if rollout["timed_out"]
            else "running"
        )

        info_text.set_text(
            f"source={source_label} | action_mode={env.action_mode} | dt={env.dt:.2f}s | speed_max={env.max_speed_mps:.2f}m/s\n"
            f"steps={rollout['steps']} | episode lasted {rollout['time_s']:.1f}s | coverage={100.0 * rollout['coverage']:.2f}% | outcome={outcome}\n"
            f"target={100.0 * coverage_target:.2f}% | reached={estimate['reached']} | steps_to_target={estimate['steps_to_target']} | suggested_max_steps={suggested_max_steps}"
        )

        print(
            f"Generated episode with seed={args.seed}: steps={rollout['steps']} duration={rollout['time_s']:.1f}s outcome={outcome}"
        )
        fig.canvas.draw_idle()

    def _set_params(_event):
        spiral = _build_spiral_from_sliders(sliders)
        _save_spiral_params_to_yaml(spiral_params_path, spiral)
        info_text.set_text(
            f"source={source_label} | action_mode={env.action_mode} | dt={env.dt:.2f}s | speed_max={env.max_speed_mps:.2f}m/s\n"
            f"spiral parameters saved to {spiral_params_path}"
        )
        print(f"Saved spiral parameters to: {spiral_params_path}")
        fig.canvas.draw_idle()

    def _reset(_event):
        for slider in sliders.values():
            slider.reset()

    def _print_cli(_event):
        spiral = _build_spiral_from_sliders(sliders)
        params = spiral_params_dict(spiral)
        cli = (
            f"--spiral-a-m {params['a_m']:.4f} "
            f"--spiral-b-m-per-rad {params['b_m_per_rad']:.4f} "
            f"--spiral-lookahead-rad {params['lookahead_rad']:.4f} "
            f"--spiral-heading-gain {params['heading_gain']:.4f} "
            f"--spiral-radial-gain {params['radial_gain']:.4f} "
            f"--spiral-thruster-forward {params['thruster_forward']:.4f}"
        )
        print(cli)

    for slider in sliders.values():
        slider.on_changed(_update_preview)
    reset_btn.on_clicked(_reset)
    print_btn.on_clicked(_print_cli)
    generate_btn.on_clicked(_generate)
    set_params_btn.on_clicked(_set_params)

    _update_preview()
    plt.show()


if __name__ == "__main__":
    if any(arg == "--save" or arg.startswith("--save=") for arg in sys.argv):
        raise ValueError("This tuner is interactive only and does not support --save")
    main()
