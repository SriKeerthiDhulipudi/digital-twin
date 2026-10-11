"""Local API for synthetic demand scenarios in the digital-twin simulator."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import (
    forecast,
    load_uci_hourly_demand,
    load_csv_numeric_demand,
    operating_cost,
    recommend_staff,
    run_experiment,
    synthetic_demand,
)

app = FastAPI(title="Enterprise Digital Twin API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioRequest(BaseModel):
    seed: int = Field(default=10, ge=0, le=1_000_000)
    demand_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    replay_hours: int = Field(default=48, ge=12, le=240)


MAX_UPLOAD_BYTES = 500 * 1024 * 1024


def _scenario_result(series, data_source: str, is_synthetic: bool) -> dict:
    actual, predicted = forecast(series)
    result = run_experiment(series=series, data_source=data_source)
    result["is_synthetic"] = is_synthetic
    staffing = [recommend_staff(float(value)) for value in predicted]
    replay = [
        {
            "hour": index + 1,
            "actual": round(float(demand), 2),
            "forecast": round(float(predicted[index]), 2),
            "recommended_staff": staffing[index],
            "recommended_cost": round(operating_cost(float(demand), staffing[index]), 2),
            "fixed_cost": round(operating_cost(float(demand), 4), 2),
        }
        for index, demand in enumerate(actual)
    ]
    result["holdout_policy"] = "time-ordered 70% train / trailing 30% test; six prior rows supply lag features"
    return {"metrics": result, "replay": replay, "is_synthetic": is_synthetic}


@app.get("/")
def root() -> dict:
    return {
        "project": "Enterprise Digital Twin API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "model_status": "/api/model/status",
            "sample_metadata": "/api/sample",
            "run_scenario": "/api/scenario (POST)",
            "evaluate_dataset": "/api/dataset/evaluate (POST)",
        },
        "frontend": "http://localhost:5173",
    }


@app.get("/api")
def api_index() -> dict:
    return root()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "project": "digital-twin"}


@app.get("/api/model/status")
def model_status() -> dict:
    project_root = Path(__file__).resolve().parent
    manifest_path = project_root / "Models" / "model_manifest.json"
    summary_path = project_root / "Models" / "weights" / "model_summary.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload = {
        "project": manifest.get("project", "project-10-digital-twin-enterprise"),
        "model_type": manifest.get("model_type", "Forecasting and digital-twin action optimizer"),
        "status": manifest.get("status", "trained"),
        "last_run": manifest.get("last_run"),
    }
    payload.update({key: value for key, value in summary.items() if key not in payload})
    return payload


@app.get("/api/sample")
def sample_metadata() -> dict:
    project_root = Path(__file__).resolve().parent
    sample_path = project_root / "Sample Upload" / "raw" / "sample_hour.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample demand CSV not found.")
    series = load_uci_hourly_demand(sample_path)
    return {
        "dataset": "Bundled sample UCI Bike Sharing hour.csv",
        "path": str(sample_path),
        "row_count": int(len(series)),
        "min_demand": float(series.min()),
        "max_demand": float(series.max()),
        "mean_demand": float(series.mean()),
    }


@app.post("/api/scenario")
def run_scenario(request: ScenarioRequest) -> dict:
    # Produce a history whose 30% held-out split contains exactly the requested
    # number of replay points (forecast() reserves six initial lag values).
    history_length = (request.replay_hours * 10) // 3 + 6
    series = synthetic_demand(seed=request.seed, n=history_length) * request.demand_scale
    result = _scenario_result(
        series,
        "Synthetic demand generated locally by main.synthetic_demand(seed), then scaled.",
        True,
    )
    return {
        **result,
        "seed": request.seed,
        "demand_scale": request.demand_scale,
        "replay_hours": request.replay_hours,
    }


async def _read_upload_limited(upload: UploadFile) -> bytes:
    chunks = []
    total = 0
    while chunk := await upload.read(min(1024 * 1024, MAX_UPLOAD_BYTES + 1 - total)):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            await upload.close()
            raise HTTPException(status_code=413, detail="Upload exceeds the 500 MiB limit.")
        chunks.append(chunk)
    if total == 0:
        await upload.close()
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
    return b"".join(chunks)


@app.post("/api/dataset/evaluate")
async def evaluate_dataset(file: UploadFile = File(...)) -> dict:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    payload = await _read_upload_limited(file)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="bike-sharing-", suffix=".csv", delete=False) as temporary:
            temporary_path = temporary.name
            temporary.write(payload)
        try:
            with open(temporary_path, newline="", encoding="utf-8") as source:
                fieldnames = set(csv.DictReader(source).fieldnames or [])
            if {"instant", "dteday", "hr", "cnt"}.issubset(fieldnames):
                selected_column = "cnt"
                series = load_uci_hourly_demand(temporary_path)
                dataset_description = "UCI Bike Sharing-format"
            else:
                selected_column, series = load_csv_numeric_demand(temporary_path)
                dataset_description = "generic"
            result = _scenario_result(
                series,
                f"Uploaded {dataset_description} CSV ({filename}); numeric column '{selected_column}' evaluated in input row order.",
                False,
            )
        except (OSError, UnicodeError, ValueError, csv.Error) as error:
            status_code = 413 if "evaluation limit" in str(error) else 400
            raise HTTPException(status_code=status_code, detail=f"Invalid CSV dataset: {error}") from error
        result["mode"] = "uploaded_dataset"
        result["provenance"] = (
            f"User-uploaded CSV; numeric column '{selected_column}' was selected "
            f"({'the cnt column for UCI Bike Sharing data' if selected_column == 'cnt' else 'automatically as the first complete numeric column'}) "
            "and evaluated in supplied row order."
        )
        result["evaluation_assumptions"] = (
            "Trailing 30% time-ordered holdout; six preceding rows are lag features. "
            "The first complete finite, non-negative numeric column is used; other columns are not modeled."
        )
        return result
    finally:
        await file.close()
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
