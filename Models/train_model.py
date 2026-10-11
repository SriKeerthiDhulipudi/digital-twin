from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
WEIGHTS_DIR = MODEL_DIR / "weights"
SAMPLE_PATH = ROOT / "Sample Upload" / "raw" / "sample_hour.csv"


def export_training_run(seed: int = 10) -> dict:
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if SAMPLE_PATH.exists():
        series = main.load_uci_hourly_demand(SAMPLE_PATH)
        result = main.run_experiment(seed=seed, series=series, data_source=f"UCI Bike Sharing hour.csv ({SAMPLE_PATH.name})")
    else:
        series = main.synthetic_demand(seed=seed)
        result = main.run_experiment(seed=seed, series=series, data_source="synthetic_hourly_demand")

    checkpoint_path = CHECKPOINTS_DIR / "last_run_metrics.json"
    checkpoint_payload = {
        "project": "project-10-digital-twin-enterprise",
        "seed": seed,
        "dataset": result.get("data_source", "synthetic_hourly_demand"),
        "test_hours": int(result.get("test_hours", 0)),
        "forecast_mae": float(result.get("forecast_mae", 0.0)),
        "decision_regret_vs_perfect_forecast": float(result.get("decision_regret_vs_perfect_forecast", 0.0)),
        "metrics": result,
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

    manifest_path = MODEL_DIR / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["project"] = "project-10-digital-twin-enterprise"
    manifest["model_type"] = "Forecasting and digital-twin action optimizer"
    manifest["status"] = "trained"
    manifest["last_run"] = {
        "seed": seed,
        "dataset": result.get("data_source", "synthetic_hourly_demand"),
        "checkpoint": str(checkpoint_path),
        "metrics": result,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "project": "project-10-digital-twin-enterprise",
        "model_type": manifest["model_type"],
        "status": "trained",
        "dataset": result.get("data_source", "synthetic_hourly_demand"),
        "seed": seed,
        "test_hours": int(result.get("test_hours", 0)),
        "forecast_mae": float(result.get("forecast_mae", 0.0)),
        "decision_regret_vs_perfect_forecast": float(result.get("decision_regret_vs_perfect_forecast", 0.0)),
        "checkpoint": str(checkpoint_path),
    }
    (WEIGHTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(export_training_run(), indent=2))
