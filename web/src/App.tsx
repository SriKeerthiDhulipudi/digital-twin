import { useEffect, useState, type FormEvent } from "react";

type ReplayPoint = { hour: number; actual: number; forecast: number; recommended_staff: number; recommended_cost: number; fixed_cost: number };
type Metrics = { test_hours: number; forecast_mae: number; recommended_staffing_cost: number; fixed_four_staff_cost: number; decision_regret_vs_perfect_forecast: number; mean_recommended_staff: number; data_source: string; holdout_policy?: string };
type TwinResult = { metrics: Metrics; replay: ReplayPoint[]; seed?: number; demand_scale?: number; replay_hours?: number; is_synthetic: boolean; provenance?: string; evaluation_assumptions?: string };

export default function App() {
  const [seed, setSeed] = useState(10);
  const [scale, setScale] = useState(1);
  const [hours, setHours] = useState(72);
  const [result, setResult] = useState<TwinResult | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetResult, setDatasetResult] = useState<TwinResult | null>(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState("");
  const [dataMode, setDataMode] = useState<"synthetic" | "uploaded">("synthetic");

  async function runScenario() {
    setBusy(true); setError(""); setPlaying(false);
    setDataMode("synthetic");
    try {
      const response = await fetch("/api/scenario", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, demand_scale: scale, replay_hours: hours }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Scenario request failed.");
      setResult(body); setIndex(0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reach the digital-twin API. Start FastAPI and try again.");
    } finally { setBusy(false); }
  }

  async function evaluateDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetFile) { setDatasetError("Choose a CSV dataset."); return; }
    setDatasetBusy(true); setDatasetError(""); setPlaying(false);
    try {
      const form = new FormData();
      form.append("file", datasetFile);
      const response = await fetch("/api/dataset/evaluate", { method: "POST", body: form });
      const text = await response.text();
      let body: any;
      try {
        body = JSON.parse(text);
      } catch {
        throw new Error(response.status === 413 ? "File is too large for serverless upload (Vercel limit is 4.5 MB). Try a smaller CSV." : text || `Server error (${response.status})`);
      }
      if (!response.ok) throw new Error(body.detail ?? "Dataset evaluation failed.");
      setDatasetResult(body);
      setIndex(0);
      setDataMode("uploaded");
    } catch (cause) {
      setDatasetError(cause instanceof Error ? cause.message : "Could not reach the local dataset evaluator.");
    } finally { setDatasetBusy(false); }
  }

  const activeResult = dataMode === "uploaded" ? datasetResult : result;
  useEffect(() => {
    if (!playing || !activeResult) return;
    const timer = window.setInterval(() => {
      setIndex(current => {
        if (current >= activeResult.replay.length - 1) { setPlaying(false); return current; }
        return current + 1;
      });
    }, 420);
    return () => window.clearInterval(timer);
  }, [playing, activeResult]);

  const current = activeResult?.replay[index];
  const maxValue = activeResult?.replay.reduce(
    (maximum, point) => Math.max(maximum, point.actual, point.forecast),
    1,
  ) ?? 1;
  const chartPoints = (field: "actual" | "forecast") => activeResult?.replay.map((point, i) => `${16 + i / Math.max(1, activeResult.replay.length - 1) * 568},${190 - point[field] / maxValue * 156}`).join(" ") ?? "";
  return <main className="twin-shell">
    <header className="twin-nav"><a href="#dashboard" className="twin-brand"><span className="brand-icon">S</span> SIGNAL<span className="brand-tag">TWIN LAB</span></a><div className="nav-right"><span className="synthetic-tag"><i /> {dataMode === "uploaded" ? "UPLOADED UCI DATA" : "SYNTHETIC SANDBOX"}</span><span className="project-tag">PROJECT 10</span></div></header>
    <section className="twin-title" id="dashboard"><div className="twin-kicker"><span>ENTERPRISE ANALYTICS</span><b>/</b><span>DEMAND → ACTION</span></div><h1>See the next shift<br /><em>before it starts.</em></h1><p>Explore how a lag-based demand forecast translates into staffing decisions—and replay the held-out hours, one step at a time.</p></section>
    <section className="twin-layout">
      <aside className="scenario-card"><div className="scenario-head"><span>SCENARIO CONTROLS</span><span className="control-number">01</span></div><h2>Shape the demand</h2><p>Each run rebuilds a seeded synthetic demand series and evaluates the existing forecast-and-staffing pipeline.</p>
        <label htmlFor="seed">Scenario seed <span>REPEATABLE</span></label><input id="seed" type="number" min="0" max="1000000" value={seed} onChange={event => setSeed(Number(event.target.value))} />
        <label htmlFor="demand">Demand intensity <strong>{scale.toFixed(1)}×</strong></label><input id="demand" type="range" min="0.5" max="2" step="0.1" value={scale} onChange={event => setScale(Number(event.target.value))} /><div className="scale-ends"><span>Lower</span><span>Higher</span></div>
        <label htmlFor="hours">Simulation window <span>HOURLY</span></label><select id="hours" value={hours} onChange={event => setHours(Number(event.target.value))}><option value="48">48 hours</option><option value="72">72 hours</option><option value="120">120 hours</option><option value="240">240 hours</option></select>
        <button className="run-button" onClick={runScenario} disabled={busy} aria-busy={busy}>{busy ? <><span className="spinner" /> Running model…</> : <>Run twin scenario <span>↗</span></>}</button>
        <form className="twin-upload" onSubmit={evaluateDataset}>
          <label htmlFor="bike-file">Real-data mode <span>CSV · MAX 500 MiB</span></label>
          <input id="bike-file" type="file" accept=".csv" onChange={event => setDatasetFile(event.target.files?.[0] ?? null)} />
          <p>Upload any CSV. The first complete numeric column is evaluated in file order with a trailing 30% holdout.</p>
          <button className="run-button upload-run" type="submit" disabled={datasetBusy || !datasetFile} aria-busy={datasetBusy}>{datasetBusy ? "Evaluating upload…" : "Evaluate uploaded data"}</button>
        </form>
        <div className="simulation-stamp"><span>MODEL</span><strong>Lag regression + cost optimizer</strong><span className="stamp-status">LOCAL · READ-ONLY</span></div>
      </aside>
      <div className="twin-results">
        {error && <div className="twin-error" role="alert"><strong>API unavailable</strong> {error}</div>}
        {datasetError && <div className="twin-error" role="alert"><strong>Dataset evaluation failed</strong> {datasetError}</div>}
        {!activeResult && !error && <div className="twin-empty"><div className="pulse-icon">⌁</div><h2>Awaiting scenario</h2><p>Configure demand, then run a scenario to reveal forecast accuracy, staffing cost, and replay metrics.</p></div>}
        {activeResult && <div aria-live="polite">
          <div className="data-mode-switch" role="group" aria-label="Select demand dataset"><button type="button" className={dataMode === "synthetic" ? "active" : ""} onClick={() => { setDataMode("synthetic"); setIndex(0); }}>Synthetic demo</button><button type="button" className={dataMode === "uploaded" ? "active" : ""} disabled={!datasetResult} onClick={() => { setDataMode("uploaded"); setIndex(0); }}>Uploaded UCI data</button></div>
          <div className="metric-strip">
            <TwinMetric label="Forecast MAE" value={activeResult.metrics.forecast_mae.toFixed(2)} unit="DEMAND / HR" />
            <TwinMetric label="Staffing cost" value={activeResult.metrics.recommended_staffing_cost.toFixed(1)} unit="PER TEST HOUR" />
            <TwinMetric label="Fixed 4-staff cost" value={activeResult.metrics.fixed_four_staff_cost.toFixed(1)} unit="PER TEST HOUR" />
            <TwinMetric label="Decision regret" value={activeResult.metrics.decision_regret_vs_perfect_forecast.toFixed(2)} unit="VS. PERFECT FORECAST" />
          </div>
          <section className="chart-card"><div className="chart-header"><div><span className="micro-label">HELD-OUT DEMAND</span><h2>Actual vs. forecast</h2></div><div className="chart-legend"><span><i className="actual-swatch" /> Actual</span><span><i className="forecast-swatch" /> Forecast</span></div></div>
            <div className="chart-area"><svg viewBox="0 0 600 210" role="img" aria-label={`Actual and forecast ${activeResult.is_synthetic ? "synthetic" : "UCI Bike Sharing"} hourly demand chart`}><line x1="16" y1="35" x2="584" y2="35" /><line x1="16" y1="87" x2="584" y2="87" /><line x1="16" y1="139" x2="584" y2="139" /><polyline className="forecast-line" points={chartPoints("forecast")} /><polyline className="actual-line" points={chartPoints("actual")} />{current && <line className="replay-cursor" x1={16 + index / Math.max(1, activeResult.replay.length - 1) * 568} x2={16 + index / Math.max(1, activeResult.replay.length - 1) * 568} y1="24" y2="192" />}</svg><div className="chart-xlabels"><span>Hour 1</span><span>Hour {Math.round(activeResult.replay.length / 2)}</span><span>Hour {activeResult.replay.length}</span></div></div>
            <div className="replay-controls"><button className="play-button" onClick={() => { if (index >= activeResult.replay.length - 1) setIndex(0); setPlaying(!playing); }} aria-label={playing ? "Pause replay" : "Play replay"}>{playing ? "Ⅱ" : "▶"}</button><div className="replay-slider"><label htmlFor="replay">REPLAY <strong>HOUR {current?.hour ?? 0}</strong><span> / {activeResult.replay.length}</span></label><input id="replay" type="range" min="0" max={activeResult.replay.length - 1} value={index} onChange={event => { setPlaying(false); setIndex(Number(event.target.value)); }} /></div><span className="replay-live">{playing ? "PLAYING" : "PAUSED"}</span></div>
          </section>
          {current && <section className="hour-card"><div className="hour-top"><div><span className="micro-label">REPLAY INSPECTOR</span><h2>Hour {current.hour}</h2></div><span className="staff-pill">{current.recommended_staff} staff recommended</span></div><div className="hour-values"><Value label="Actual demand" value={current.actual.toFixed(1)} /><Value label="Model forecast" value={current.forecast.toFixed(1)} /><Value label="Recommended cost" value={current.recommended_cost.toFixed(1)} /><Value label="Fixed policy cost" value={current.fixed_cost.toFixed(1)} /></div></section>}
          <div className="data-disclaimer"><span>i</span><p><strong>{activeResult.is_synthetic ? "Synthetic demo:" : "Uploaded UCI data:"}</strong> {activeResult.metrics.data_source} {activeResult.provenance} {activeResult.evaluation_assumptions} Metrics represent this simple simulator only, not a validated enterprise policy.</p></div>
        </div>}
      </div>
    </section>
    <footer className="twin-footer"><span>SIGNAL TWIN LAB <i>·</i> PROJECT 10</span><span>ALL METRICS ARE SIMULATED</span></footer>
  </main>;
}

function TwinMetric({ label, value, unit }: { label: string; value: string; unit: string }) {
  return <div className="twin-metric"><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>;
}
function Value({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
