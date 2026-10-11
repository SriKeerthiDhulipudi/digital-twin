# Real-Time Digital Twin for Enterprise Analytics

## Purpose
An end-to-end small demand-to-action twin: train a lag-based linear forecast, select staffing levels by minimizing operating cost, and compare the decision against a fixed-staff baseline and a perfect-forecast oracle.

## Limitations
The default hourly demand is reproducibly generated and is not a real enterprise stream. The simulator is a simple per-hour capacity/cost model, not SimPy, a production system, or a validated operational policy. Forecast and cost metrics depend on the synthetic process and chosen cost constants.

## Setup
Python 3.10+; install the API and simulation dependencies from `requirements.txt`.

```powershell
cd project-10-digital-twin-enterprise
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data acquisition
The default is synthetic. Download and extract the official UCI Bike Sharing archive from <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>; no data is bundled or downloaded automatically. Pass its `hour.csv` to `--data`. The loader requires the `cnt` header, accepts finite non-negative demand values, and preserves the file's existing row order. The current model uses only this ordered count series; timestamps and other weather/context columns are not modeled. Cite UCI when reporting results.

## Use

```powershell
python main.py
python main.py --seed 31
python main.py --data path\to\hour.csv
```

## Evaluation
`forecast_mae` is held-out mean absolute error. Recommended and fixed-policy cost use the same test hours; `decision_regret_vs_perfect_forecast` compares the forecast-driven policy against the cost-minimizing action with actual demand. Run the unit tests to check deterministic data, finite forecasts, and nonnegative oracle regret.
The cost and staffing helpers require finite non-negative demand, a
non-negative integer staffing level, and finite positive capacity per staff.

## Interactive local app

The React 19 + TypeScript + Vite frontend is in `web/`. Its FastAPI service
invokes the existing synthetic demand generator, forecaster, staffing
optimizer, and cost functions. Data and metrics are synthetic; they are not a
validated enterprise forecast or staffing recommendation.

From this project directory, install the Python API dependencies and start it:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload
```

In a second terminal:

```powershell
cd web
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). Adjust
the seed and demand scale, run the scenario, and play or scrub the held-out
hourly forecast and staffing metrics. Validate with
`python -m unittest discover -s tests -v` from the project directory and
`npm run build` from `web/`.

### Browser upload evaluation

The synthetic scenario remains the default. Choose **Evaluate uploaded data**
to submit any CSV by multipart field `file`. For UCI Bike Sharing `hour.csv`,
the required `instant`, `dteday`, `hr`, and `cnt` columns are validated and
`cnt` is modeled. For other CSV files, the first complete, finite,
non-negative numeric column is selected automatically and modeled in the
supplied row order. The result identifies the selected column.

Forecast evaluation uses a time-ordered 70% train / trailing 30% holdout with
six preceding rows for lag features. Uploads are capped at 500 MiB while read
and one million rows, evaluated from a temporary file, and removed after the
request. Nothing is persisted in the source tree. This local API has no
authentication and is not a privacy boundary; do not expose it to untrusted
networks or upload data you are not authorized to process.
