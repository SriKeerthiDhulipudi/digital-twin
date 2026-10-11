"""Forecast-to-action discrete demand twin with a clairvoyant regret baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def synthetic_demand(seed=10, n=240):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    demand = 18 + 5 * np.sin(2 * np.pi * t / 24) + 0.025 * t
    demand += rng.normal(0, 2.0, n)
    return np.maximum(0, demand)


def load_uci_hourly_demand(path, max_rows=1_000_000):
    """Read the UCI Bike Sharing hour.csv file's cnt column."""
    values = []
    previous_instant = None
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_fields = {"instant", "dteday", "hr", "cnt"}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Expected UCI Bike Sharing hour.csv columns: instant, dteday, hr, cnt.")
        for line_number, row in enumerate(reader, start=2):
            try:
                instant_value = float(row["instant"])
                value = float(row["cnt"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Bike Sharing instant and 'cnt' values must be numeric on CSV line {line_number}.") from error
            if not np.isfinite(instant_value) or not instant_value.is_integer():
                raise ValueError(f"Bike Sharing 'instant' must be a finite integer on CSV line {line_number}.")
            instant = int(instant_value)
            if previous_instant is not None and instant <= previous_instant:
                raise ValueError(
                    "Bike Sharing 'instant' values must be strictly increasing in input row order "
                    f"(CSV line {line_number})."
                )
            previous_instant = instant
            if not np.isfinite(value) or value < 0:
                raise ValueError("Bike Sharing 'cnt' values must be finite and non-negative.")
            if len(values) >= max_rows:
                raise ValueError(f"Bike Sharing CSV exceeds the {max_rows}-row evaluation limit.")
            values.append(value)
    if not values:
        raise ValueError("Bike Sharing CSV contains no hourly demand rows.")
    return np.asarray(values, dtype=float)


def load_csv_numeric_demand(path, max_rows=1_000_000):
    """Read the first complete, finite, non-negative numeric CSV column."""
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row.")
        columns = {field: [] for field in reader.fieldnames if field}
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if row_count > max_rows:
                raise ValueError(f"CSV exceeds the {max_rows}-row evaluation limit.")
            for field in columns:
                raw_value = row.get(field, "")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    columns[field].append(None)
                    continue
                columns[field].append(value if np.isfinite(value) and value >= 0 else None)
    valid_columns = {field: values for field, values in columns.items() if values and all(v is not None for v in values)}
    if not valid_columns:
        raise ValueError("CSV does not contain a complete finite, non-negative numeric column.")
    skip_names = {"time", "timestamp", "datetime", "date", "id", "instant", "index", "row", "unnamed: 0", "step"}
    for field, values in valid_columns.items():
        if field.strip().lower() not in skip_names:
            return field, np.asarray(values, dtype=float)
    field, values = next(iter(valid_columns.items()))
    return field, np.asarray(values, dtype=float)


def make_features(series, lag=6):
    x = np.array([series[i - lag:i] for i in range(lag, len(series))])
    y = series[lag:]
    return x, y


def forecast(series, train_fraction=0.7):
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("Demand series must be a finite one-dimensional array.")
    if np.any(series < 0):
        raise ValueError("Demand series must contain non-negative values.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    x, y = make_features(series)
    split = int(len(y) * train_fraction)
    if split < 1 or split >= len(y):
        raise ValueError("Demand series is too short for non-empty train and test splits.")
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]
    design = np.column_stack((np.ones(len(train_x)), train_x))
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    predictions = np.maximum(0, np.column_stack((np.ones(len(test_x)), test_x)) @ coefficients)
    return test_y, predictions


def operating_cost(demand: float, staff: int, capacity_per_staff=5.0) -> float:
    if isinstance(demand, bool) or not isinstance(demand, (int, float, np.number)):
        raise ValueError("demand must be finite and non-negative.")
    if not np.isfinite(demand) or demand < 0:
        raise ValueError("demand must be finite and non-negative.")
    if isinstance(staff, bool) or not isinstance(staff, (int, np.integer)) or staff < 0:
        raise ValueError("staff must be a non-negative integer.")
    if (
        isinstance(capacity_per_staff, bool)
        or not isinstance(capacity_per_staff, (int, float, np.number))
        or not np.isfinite(capacity_per_staff)
        or capacity_per_staff <= 0
    ):
        raise ValueError("capacity_per_staff must be finite and positive.")
    unmet = max(0.0, demand - staff * capacity_per_staff)
    return 0.7 * staff + 4.0 * unmet


def recommend_staff(forecast_demand: float, max_staff=8) -> int:
    if isinstance(max_staff, bool) or not isinstance(max_staff, (int, np.integer)) or max_staff < 0:
        raise ValueError("max_staff must be a non-negative integer.")
    return min(range(max_staff + 1),
               key=lambda staff: operating_cost(forecast_demand, staff))


def run_experiment(seed=10, series=None, data_source=None):
    source = data_source or ("synthetic_hourly_demand" if series is None else "provided_series")
    series = synthetic_demand(seed) if series is None else np.asarray(series, dtype=float)
    actual, predicted = forecast(series)
    chosen = np.array([recommend_staff(value) for value in predicted])
    oracle = np.array([recommend_staff(value) for value in actual])
    baseline = np.full(len(actual), 4)
    chosen_cost = np.array([operating_cost(d, s) for d, s in zip(actual, chosen)])
    oracle_cost = np.array([operating_cost(d, s) for d, s in zip(actual, oracle)])
    fixed_cost = np.array([operating_cost(d, s) for d, s in zip(actual, baseline)])
    return {
        "data_source": source,
        "is_synthetic": source == "synthetic_hourly_demand",
        "test_hours": int(len(actual)),
        "forecast_mae": float(np.mean(np.abs(actual - predicted))),
        "recommended_staffing_cost": float(chosen_cost.mean()),
        "fixed_four_staff_cost": float(fixed_cost.mean()),
        "decision_regret_vs_perfect_forecast": float(np.mean(chosen_cost - oracle_cost)),
        "mean_recommended_staff": float(chosen.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data", help="Optional UCI Bike Sharing hour.csv path")
    args = parser.parse_args()
    series = load_uci_hourly_demand(args.data) if args.data else None
    source = f"UCI Bike Sharing hour.csv ({Path(args.data).name})" if args.data else None
    result = run_experiment(args.seed, series, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
