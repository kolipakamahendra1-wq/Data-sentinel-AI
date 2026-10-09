import { useCallback, useEffect, useState } from "react";
import { api, type Alert, type Dataset, type Health, type Report } from "./api";

const GENERATORS = ["loan", "churn", "fraud", "demand"];
const FAILURES: Record<string, string> = {
  missing_values: "Missing values",
  distribution_shift: "Distribution shift",
  label_imbalance: "Label imbalance",
  schema_change: "Schema change",
  outliers: "Outliers",
  data_leakage: "Data leakage",
};
const PSI_MAX = 1; // bar scale; values above are clipped
const PSI_THRESHOLD = 0.2;

const tone = (s: "ok" | "warn" | "bad") => ({ ok: "text-ok", warn: "text-warn", bad: "text-bad" })[s];
const scoreTone = (n: number | null) => (n === null ? "text-muted" : n >= 90 ? "text-ok" : n >= 60 ? "text-warn" : "text-bad");
const plain = (text: string) => text.replaceAll("`", "");
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (select?: number) => {
    try {
      const [ds, al] = await Promise.all([api.datasets(), api.alerts()]);
      setDatasets(ds);
      setAlerts(al);
      setSelected((cur) => select ?? cur ?? ds[0]?.id ?? null);
      setError(null);
    } catch (e) {
      setError(`Can't reach the DataSentinel API (${(e as Error).message}). Start it with: uvicorn backend.main:app`);
    }
  }, []);

  useEffect(() => void refresh(), [refresh]);
  useEffect(() => {
    if (selected === null) return;
    api.health(selected).then(setHealth, () => setHealth(null));
  }, [selected, alerts]);

  const datasetAlerts = alerts.filter((a) => a.dataset_id === selected);
  const current = datasetAlerts[0] ?? null;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-16 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-rule py-5">
        <div>
          <h1 className="text-xl font-bold tracking-tight">DataSentinel</h1>
          <p className="text-sm text-muted">Data quality and drift monitoring for production models</p>
        </div>
        <DemoControl onDone={(id) => refresh(id)} onError={setError} />
      </header>

      {error && <p role="alert" className="mt-4 border-l-4 border-bad bg-panel px-4 py-3 text-sm">{error}</p>}

      <div className="mt-6 grid gap-8 lg:grid-cols-[15rem_1fr]">
        <nav aria-label="Datasets">
          <h2 className="mb-2 text-sm font-semibold text-muted">Datasets</h2>
          {datasets.length === 0 && !error && (
            <p className="text-sm text-muted">No datasets yet. Run a demo failure to create one.</p>
          )}
          <ul className="space-y-1">
            {datasets.map((d) => (
              <li key={d.id}>
                <button
                  onClick={() => setSelected(d.id)}
                  aria-current={d.id === selected}
                  className={`flex w-full items-baseline justify-between rounded px-3 py-2 text-left text-sm ${
                    d.id === selected ? "bg-panel font-semibold" : "hover:bg-panel/60"
                  }`}
                >
                  <span className="truncate">{d.name} <span className="text-muted">#{d.id}</span></span>
                  <span className={`font-semibold ${scoreTone(d.health_score)}`}>{d.health_score ?? "–"}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <main className="min-w-0 space-y-8">
          {health ? <HealthView health={health} /> : selected !== null && (
            <p className="text-sm text-muted">This dataset has no validation runs yet.</p>
          )}
          {current && <Incident alert={current} onChange={() => refresh()} />}
          <History alerts={alerts} datasets={datasets} onSelect={setSelected} />
        </main>
      </div>
    </div>
  );
}

function DemoControl({ onDone, onError }: { onDone: (id: number) => void; onError: (e: string) => void }) {
  const [generator, setGenerator] = useState("loan");
  const [failure, setFailure] = useState("missing_values");
  const [useLlm, setUseLlm] = useState(true);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      onDone((await api.demo(generator, failure, useLlm)).dataset_id);
    } catch (e) {
      onError(`Demo failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const field = "rounded border border-rule bg-panel px-2 py-1.5 text-sm";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label className="sr-only" htmlFor="gen">Dataset</label>
      <select id="gen" className={field} value={generator} onChange={(e) => setGenerator(e.target.value)}>
        {GENERATORS.map((g) => <option key={g} value={g}>{g} model</option>)}
      </select>
      <label className="sr-only" htmlFor="fail">Failure to inject</label>
      <select id="fail" className={field} value={failure} onChange={(e) => setFailure(e.target.value)}>
        {Object.entries(FAILURES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
      <label className="flex items-center gap-1.5 text-sm">
        <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
        Use local LLM
      </label>
      <button
        onClick={run}
        disabled={busy}
        className="rounded bg-act px-3 py-1.5 text-sm font-semibold text-paper disabled:opacity-60"
      >
        {busy ? (useLlm ? "Investigating with Ollama…" : "Running…") : "Inject failure"}
      </button>
    </div>
  );
}

function HealthView({ health }: { health: Health }) {
  const features = [...health.drift.features].sort((a, b) => b.psi - a.psi);
  const m = health.model;
  return (
    <section aria-labelledby="health-h">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <h2 id="health-h" className="text-lg font-semibold">{health.name}</h2>
        <span className="text-sm text-muted">Last checked {when(health.checked_at)}</span>
      </div>

      <div className="mt-4 grid gap-6 md:grid-cols-[auto_1fr]">
        <div className="pr-6 md:border-r md:border-rule">
          <p className="text-sm text-muted">Dataset health</p>
          <p className={`text-6xl font-bold leading-none ${scoreTone(health.health_score)}`}>{health.health_score}</p>
          <p className="mt-1 text-sm text-muted">% of features passing</p>
          <p className="mt-4 text-sm text-muted">Max drift (PSI)</p>
          <p className="text-2xl font-semibold">{health.drift.max_psi.toFixed(2)}</p>
          {health.drift.label.ref_rate !== null && (
            <>
              <p className="mt-4 text-sm text-muted">Label positive rate</p>
              <p className={`text-base font-semibold ${health.drift.label.drifted ? "text-bad" : ""}`}>
                {(health.drift.label.ref_rate * 100).toFixed(1)}% → {((health.drift.label.cur_rate ?? 0) * 100).toFixed(1)}%
              </p>
            </>
          )}
        </div>

        <div>
          <h3 className="mb-3 text-sm font-semibold">Feature drift against reference</h3>
          <ul className="space-y-2" aria-label="Feature drift">
            {features.map((f) => {
              const state = health.feature_health[f.feature] ?? "ok";
              return (
                <li key={f.feature} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 text-sm">
                  <span className={`truncate ${state !== "ok" ? `font-semibold ${tone(state)}` : ""}`}>{f.feature}</span>
                  <span className="relative h-2.5 rounded-sm bg-rule/60" aria-hidden>
                    <span
                      className={`absolute inset-y-0 left-0 rounded-sm ${f.drifted ? "bg-bad" : "bg-ok"}`}
                      style={{ width: `${Math.max(1, Math.min(f.psi / PSI_MAX, 1) * 100)}%` }}
                    />
                    <span className="absolute -inset-y-1 w-px bg-ink" style={{ left: `${(PSI_THRESHOLD / PSI_MAX) * 100}%` }} />
                  </span>
                  <span className="text-right">{f.psi.toFixed(2)}</span>
                </li>
              );
            })}
            {Object.entries(health.feature_health)
              .filter(([name]) => !features.some((f) => f.feature === name))
              .map(([name, state]) => (
                <li key={name} className="grid grid-cols-[9rem_1fr_3.5rem] gap-3 text-sm">
                  <span className={`truncate font-semibold ${tone(state)}`}>{name}</span>
                  <span className="text-muted">Not comparable in this batch</span>
                </li>
              ))}
          </ul>
          <p className="mt-2 text-xs text-muted">The vertical line marks the drift threshold (PSI 0.2).</p>
          {health.findings.length > 0 && (
            <>
              <h3 className="mt-5 mb-2 text-sm font-semibold">Quality check findings</h3>
              <ul className="space-y-1 text-sm">
                {health.findings.map((f, i) => (
                  <li key={i} className={f.severity === "high" ? "text-bad" : "text-warn"}>{plain(f.message)}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>

      {m && (
        <div className="mt-6">
          <h3 className="mb-2 text-sm font-semibold">
            Model health <span className={m.degraded ? "text-bad" : "text-ok"}>{m.degraded ? "degraded" : "stable"}</span>
          </h3>
          <dl className="grid grid-cols-3 gap-x-6 gap-y-3 sm:grid-cols-6">
            {(["accuracy", "precision", "recall", "f1", "roc_auc", "ece"] as const).map((k) => {
              const d = m.deltas[k];
              const worse = k === "ece" ? d > 0.02 : d < -0.02;
              return (
                <div key={k}>
                  <dt className="text-xs text-muted">{k === "roc_auc" ? "ROC-AUC" : k === "ece" ? "Calibration error" : k[0].toUpperCase() + k.slice(1)}</dt>
                  <dd className="text-lg font-semibold">{m.current[k].toFixed(3)}</dd>
                  <dd className={`text-xs ${worse ? "text-bad" : "text-muted"}`}>{d >= 0 ? "+" : ""}{d.toFixed(3)} vs baseline</dd>
                </div>
              );
            })}
          </dl>
        </div>
      )}
    </section>
  );
}

function Incident({ alert, onChange }: { alert: Alert; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const r: Report | null = alert.report;

  async function investigate(useLlm: boolean) {
    setBusy(true);
    setErr(null);
    try {
      await api.investigate(alert.id, useLlm);
      onChange();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="incident-h" className={`border-l-4 bg-panel px-5 py-5 sm:px-7 ${alert.severity === "high" ? "border-bad" : "border-warn"}`}>
      <p className="text-sm text-muted">
        Alert #{alert.id} raised {when(alert.created_at)} · {alert.severity} severity
      </p>
      <h2 id="incident-h" className="mt-1 font-serif text-2xl font-semibold leading-snug">{plain(alert.title)}</h2>

      {!r && (
        <div className="mt-4 flex flex-wrap gap-2">
          <button disabled={busy} onClick={() => investigate(true)} className="rounded bg-act px-3 py-1.5 text-sm font-semibold text-paper disabled:opacity-60">
            {busy ? "Investigating…" : "Investigate with local LLM"}
          </button>
          <button disabled={busy} onClick={() => investigate(false)} className="rounded border border-rule px-3 py-1.5 text-sm">
            Investigate with rules
          </button>
          {err && <p role="alert" className="text-sm text-bad">{err}</p>}
        </div>
      )}

      {r && (
        <div className="mt-4 max-w-3xl font-serif text-[1.05rem] leading-relaxed">
          <p><strong>Likely cause:</strong> {plain(r.likely_cause)}</p>
          <p className="mt-3 font-semibold">Evidence</p>
          <ul className="list-disc pl-5">
            {r.evidence.map((e, i) => <li key={i}>{plain(e)}</li>)}
          </ul>
          <p className="mt-3"><strong>Impact:</strong> {plain(r.impact)}</p>
          <p className="mt-3 border-t border-rule pt-3"><strong>Recommended action:</strong> {plain(r.recommended_action)}</p>
          <p className="mt-4 font-sans text-xs text-muted">
            Root cause classified as {r.root_cause.replace("_", " ")} with {Math.round(r.confidence * 100)}% confidence.
            Written by {r.source === "llm" ? "the local LLM (Ollama)" : "the rule-based investigator"}
            {alert.latency_ms !== null && ` in ${(alert.latency_ms / 1000).toFixed(1)} s`}.
          </p>
        </div>
      )}
    </section>
  );
}

function History({ alerts, datasets, onSelect }: { alerts: Alert[]; datasets: Dataset[]; onSelect: (id: number) => void }) {
  if (alerts.length === 0) return null;
  const name = (id: number) => datasets.find((d) => d.id === id)?.name ?? `#${id}`;
  return (
    <section aria-labelledby="history-h">
      <h2 id="history-h" className="mb-2 text-sm font-semibold">Incident history</h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[36rem] text-left text-sm">
          <thead className="border-b border-rule text-muted">
            <tr>
              <th className="py-2 pr-4 font-medium">Raised</th>
              <th className="py-2 pr-4 font-medium">Dataset</th>
              <th className="py-2 pr-4 font-medium">Alert</th>
              <th className="py-2 pr-4 font-medium">Root cause</th>
              <th className="py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a) => (
              <tr key={a.id} className="border-b border-rule/60">
                <td className="py-2 pr-4 whitespace-nowrap text-muted">{when(a.created_at)}</td>
                <td className="py-2 pr-4">
                  <button className="underline decoration-rule underline-offset-2 hover:decoration-ink" onClick={() => onSelect(a.dataset_id)}>
                    {name(a.dataset_id)}
                  </button>
                </td>
                <td className={`py-2 pr-4 ${a.severity === "high" ? "text-bad" : "text-warn"}`}>{plain(a.title)}</td>
                <td className="py-2 pr-4">{a.report?.root_cause.replace("_", " ") ?? "–"}</td>
                <td className="py-2">{a.status === "open" ? "Open" : "Investigated"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
