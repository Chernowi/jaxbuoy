import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_REWARD_GRID = {
    "environment.action_mode": ["thruster", "simplified_rudder"],
    "environment.found_reward": [100.0, 125.0, 150.0, 175.0, 200.0, 225.0],
    "environment.out_of_bounds_penalty": [30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    "environment.timeout_penalty": [60.0, 80.0, 100.0, 120.0, 140.0, 160.0],
    "environment.step_reward": [-0.15, -0.10, -0.05],
    "environment.explore_reward_per_cell": [0.05, 0.10, 0.15],
}

OBJECTIVE_METRIC = "success_rate"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _set_nested(mapping, keys, value):
    cursor = mapping
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _format_value_for_name(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.8g}"
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace("-", "_")


def _to_tokens(path):
    if isinstance(path, str):
        return path.split(".")
    return list(path)


def _build_grid(grid_spec):
    items = list(grid_spec.items())
    keys = [key for key, _ in items]
    values = [vals for _, vals in items]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cleanup_run_artifacts(run_dir):
    keep_names = {"config.yaml", "summary.json"}
    for path in run_dir.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink(missing_ok=True)


def _write_results(results_path, rows, dimension_keys):
    fieldnames = [
        "trial_id",
        "run_name",
        "status",
        "returncode",
        "summary_path",
        "config_path",
        "global_step",
        "episodic_return",
        "success_rate",
        "timeout_rate",
        "out_of_bounds_rate",
        "episodic_length",
    ]
    fieldnames.extend(dimension_keys)

    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Extensive reward-parameter grid search for buoy training"
    )
    parser.add_argument(
        "--base-config",
        default="configs/buoy_thruster_ppo.yaml",
        help="Base YAML config for all runs",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=15_000_000,
        help="TOTAL_TIMESTEPS used for every trial",
    )
    parser.add_argument(
        "--search-name",
        default=None,
        help="Name for this search (defaults to timestamp)",
    )
    parser.add_argument(
        "--output-root",
        default="runs/grid_search",
        help="Directory where trial configs and aggregated results are saved",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1300,
        help="Optional cap on number of trials (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic shuffling before applying --max-runs",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep deterministic Cartesian order instead of shuffling trials",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top trials (by success_rate) to export in leaderboard.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate trial configs and print commands without launching training",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config does not exist: {base_config_path}")

    root_dir = Path(__file__).resolve().parents[1]
    train_script = Path(__file__).resolve().parent / "train_buoy.py"
    search_name = args.search_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = (root_dir / args.output_root / search_name).resolve()
    configs_dir = search_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_config_path)

    dimension_keys = list(DEFAULT_REWARD_GRID.keys())
    grid = list(_build_grid(DEFAULT_REWARD_GRID))

    if not args.no_shuffle:
        rng = random.Random(args.sample_seed)
        rng.shuffle(grid)

    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    if not grid:
        raise ValueError("No trial combinations generated from grid.")

    print(f"Base config: {base_config_path}")
    print(f"Search dir:  {search_dir}")
    print(f"Trials:      {len(grid)}")
    print(f"Objective:   {OBJECTIVE_METRIC}")
    print(f"Budget:      {args.budget}")

    results = []
    for idx, trial in enumerate(grid, start=1):
        action_mode = trial.get("environment.action_mode", "na")
        found_reward = trial.get("environment.found_reward", "na")
        explore_reward = trial.get("environment.explore_reward_per_cell", "na")
        run_name = (
            f"{search_name}_t{idx:05d}_"
            f"{_format_value_for_name(action_mode)}_"
            f"f-{_format_value_for_name(found_reward)}_"
            f"x-{_format_value_for_name(explore_reward)}"
        )

        cfg = json.loads(json.dumps(base_cfg))
        _set_nested(cfg, ["algorithm", "params", "TOTAL_TIMESTEPS"], int(args.budget))
        _set_nested(cfg, ["wandb", "mode"], "disabled")
        _set_nested(cfg, ["render", "enabled"], False)
        for hp_key, hp_value in trial.items():
            _set_nested(cfg, _to_tokens(hp_key), hp_value)

        config_path = configs_dir / f"trial_{idx:03d}.yaml"
        _save_yaml(config_path, cfg)

        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--run-name",
            run_name,
        ]

        print(f"\n[{idx}/{len(grid)}] {run_name}")
        print("Command:", " ".join(cmd))

        trial_result = {
            "trial_id": idx,
            "run_name": run_name,
            "config_path": str(config_path),
            **trial,
        }

        if args.dry_run:
            trial_result.update(
                {
                    "status": "dry_run",
                    "returncode": "",
                    "summary_path": "",
                }
            )
            results.append(trial_result)
            continue

        completed = subprocess.run(cmd, cwd=root_dir, check=False)

        run_summary = root_dir / "runs" / run_name / "summary.json"
        if completed.returncode == 0 and run_summary.exists():
            with open(run_summary, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            trial_result.update(metrics)
            trial_result.update(
                {
                    "status": "ok",
                    "returncode": completed.returncode,
                    "summary_path": str(run_summary),
                }
            )
            print(
                f"  -> {OBJECTIVE_METRIC}: "
                f"{trial_result.get(OBJECTIVE_METRIC, 'n/a'):.4f}" if _safe_float(trial_result.get(OBJECTIVE_METRIC)) is not None
                else f"  -> {OBJECTIVE_METRIC}: n/a"
            )
        else:
            trial_result.update(
                {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "summary_path": str(run_summary),
                }
            )

        run_dir = root_dir / "runs" / run_name
        if run_dir.exists() and run_dir.is_dir():
            _cleanup_run_artifacts(run_dir)

        results.append(trial_result)

    results_csv = search_dir / "results.csv"
    _write_results(results_csv, results, dimension_keys)

    results_json = search_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    successful = [
        row
        for row in results
        if row.get("status") == "ok" and _safe_float(row.get(OBJECTIVE_METRIC)) is not None
    ]
    ranked = sorted(
        successful,
        key=lambda row: (
            _safe_float(row.get(OBJECTIVE_METRIC)) or float("-inf"),
            _safe_float(row.get("episodic_return")) or float("-inf"),
        ),
        reverse=True,
    )

    leaderboard_csv = search_dir / "leaderboard.csv"
    _write_results(leaderboard_csv, ranked[: max(0, args.top_k)], dimension_keys)

    best_json = search_dir / "best_trial.json"
    best_trial = ranked[0] if ranked else None
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump({"objective_metric": OBJECTIVE_METRIC, "best_trial": best_trial}, f, indent=2)

    print("\nGrid search completed.")
    print(f"Results CSV:  {results_csv}")
    print(f"Results JSON: {results_json}")
    print(f"Leaderboard:  {leaderboard_csv}")
    print(f"Best trial:   {best_json}")
    if best_trial is not None:
        print(
            f"Best {OBJECTIVE_METRIC}: {best_trial.get(OBJECTIVE_METRIC)} "
            f"(trial {best_trial.get('trial_id')}, run {best_trial.get('run_name')})"
        )


if __name__ == "__main__":
    main()