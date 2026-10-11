import unittest
import asyncio
from io import BytesIO
from unittest.mock import mock_open, patch

import numpy as np
from fastapi import HTTPException, UploadFile

from main import (
    forecast, load_csv_numeric_demand, load_uci_hourly_demand, operating_cost, recommend_staff, run_experiment,
    synthetic_demand,
)

try:
    import api
except RuntimeError as error:
    if "python-multipart" not in str(error):
        raise
    api = None


class DigitalTwinTests(unittest.TestCase):
    def test_demand_reproducible_and_nonnegative(self):
        self.assertTrue(np.array_equal(synthetic_demand(), synthetic_demand()))
        self.assertTrue((synthetic_demand() >= 0).all())

    def test_forecaster_returns_finite_heldout_predictions(self):
        actual, predicted = forecast(synthetic_demand())
        self.assertEqual(actual.shape, predicted.shape)
        self.assertTrue(np.isfinite(predicted).all())

    def test_forecaster_rejects_negative_demand(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            forecast(np.array([1, 2, 3, 4, 5, 6, 7, -1], dtype=float))

    def test_action_optimizer_and_regret(self):
        self.assertLessEqual(operating_cost(12, recommend_staff(12)), operating_cost(12, 0))
        result = run_experiment()
        self.assertGreater(result["test_hours"], 0)
        self.assertGreaterEqual(result["forecast_mae"], 0)
        self.assertGreaterEqual(result["decision_regret_vs_perfect_forecast"], -1e-9)

    def test_cost_and_staff_optimizer_reject_invalid_inputs(self):
        for demand in (-1.0, float("nan"), float("inf")):
            with self.subTest(demand=demand), self.assertRaisesRegex(
                ValueError, "demand must be finite and non-negative"
            ):
                operating_cost(demand, 1)
        for staff in (-1, 1.5, True):
            with self.subTest(staff=staff), self.assertRaisesRegex(
                ValueError, "staff must be a non-negative integer"
            ):
                operating_cost(10, staff)
        for capacity in (0, float("nan"), float("inf")):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(
                ValueError, "capacity_per_staff must be finite and positive"
            ):
                operating_cost(10, 1, capacity)
        for max_staff in (-1, 1.5, True):
            with self.subTest(max_staff=max_staff), self.assertRaisesRegex(
                ValueError, "max_staff must be a non-negative integer"
            ):
                recommend_staff(10, max_staff)

    def test_experiment_accepts_provided_series(self):
        result = run_experiment(series=synthetic_demand(n=120))
        self.assertEqual(result["data_source"], "provided_series")

    def test_uci_hour_csv_loader_reads_cnt_and_rejects_invalid_rows(self):
        csv_text = "instant,dteday,hr,cnt\n1,2011-01-01,0,16\n2,2011-01-01,1,40\n"
        with patch("builtins.open", mock_open(read_data=csv_text)):
            values = load_uci_hourly_demand("hour.csv")
        self.assertEqual(values.tolist(), [16.0, 40.0])
        out_of_order = "instant,dteday,hr,cnt\n2,2011-01-01,1,40\n1,2011-01-01,0,16\n"
        with patch("builtins.open", mock_open(read_data=out_of_order)):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_uci_hourly_demand("hour.csv")
        fractional_instant = "instant,dteday,hr,cnt\n1.5,2011-01-01,0,16\n"
        with patch("builtins.open", mock_open(read_data=fractional_instant)):
            with self.assertRaisesRegex(ValueError, "finite integer"):
                load_uci_hourly_demand("hour.csv")
        with patch("builtins.open", mock_open(read_data="cnt\n\n")):
            with self.assertRaises(ValueError):
                load_uci_hourly_demand("hour.csv")

    def test_generic_csv_loader_selects_first_complete_numeric_column(self):
        csv_text = "category,amount,description\nA,10,one\nB,20,two\nC,30,three\n"
        with patch("builtins.open", mock_open(read_data=csv_text)):
            column, values = load_csv_numeric_demand("data.csv")
        self.assertEqual(column, "amount")
        self.assertEqual(values.tolist(), [10.0, 20.0, 30.0])


@unittest.skipIf(api is None, "python-multipart is required to import the upload API")
class DatasetUploadTests(unittest.TestCase):

    def test_hour_csv_upload_runs_time_ordered_forecast_evaluation(self):
        rows = ["instant,dteday,hr,cnt"]
        rows.extend(f"{index},2011-01-01,{index % 24},{20 + index}" for index in range(30))
        result = asyncio.run(api.evaluate_dataset(
            UploadFile(filename="hour.csv", file=BytesIO("\n".join(rows).encode()))
        ))
        self.assertFalse(result["is_synthetic"])
        self.assertIn("UCI Bike Sharing", result["provenance"])
        self.assertIn("time-ordered", result["evaluation_assumptions"])
        self.assertEqual(result["metrics"]["test_hours"], 8)
        self.assertEqual(len(result["replay"]), 8)

    def test_csv_upload_rejects_invalid_schema(self):
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(api.evaluate_dataset(UploadFile(filename="demand.csv", file=BytesIO(b"demand\n10\n"))))
        self.assertEqual(invalid.exception.status_code, 400)

    def test_hour_csv_upload_rejects_non_chronological_rows(self):
        rows = "instant,dteday,hr,cnt\n2,2011-01-01,1,21\n1,2011-01-01,0,20\n"
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(api.evaluate_dataset(
                UploadFile(filename="hour.csv", file=BytesIO(rows.encode()))
            ))
        self.assertEqual(invalid.exception.status_code, 400)
        self.assertIn("strictly increasing", invalid.exception.detail)


if __name__ == "__main__":
    unittest.main()
