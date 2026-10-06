"""Run the prespecified AdaTrustMSA ablation and three-seed stability study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    script = Path(__file__).with_name("train.py")
    plan = [
        ("full", 2026),
        ("no_reliability", 2026),
        ("no_conflict", 2026),
        ("no_uncertainty", 2026),
        ("full", 2027),
        ("full", 2028),
    ]
    args.runs_root.mkdir(parents=True, exist_ok=True)
    (args.runs_root / "experiment_plan.json").write_text(
        json.dumps(
            {
                "ablation_seed": 2026,
                "stability_seeds": [2026, 2027, 2028],
                "runs": [{"ablation": item, "seed": seed} for item, seed in plan],
                "selection_uses": "validation only",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    completed = []
    for ablation, seed in plan:
        name = f"{ablation}_s{seed}"
        output = args.runs_root / name
        summary = output / "summary.json"
        if summary.is_file():
            print(f"REUSE {name}", flush=True)
        else:
            command = [
                sys.executable,
                str(script),
                "--data-root", str(args.data_root),
                "--encoder", str(args.encoder),
                "--output-dir", str(output),
                "--ablation", ablation,
                "--seed", str(seed),
            ]
            if args.preprocessing:
                command.extend(("--preprocessing", str(args.preprocessing)))
            if args.warm_start:
                command.extend(("--warm-start", str(args.warm_start)))
            if args.config:
                command.extend(("--config", str(args.config)))
            print(f"START {name}", flush=True)
            subprocess.run(command, check=True)
        completed.append(json.loads(summary.read_text(encoding="utf-8")))
    (args.runs_root / "completed_runs.json").write_text(
        json.dumps(completed, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("AdaTrustMSA ablation and stability runs complete", flush=True)


if __name__ == "__main__":
    main()
