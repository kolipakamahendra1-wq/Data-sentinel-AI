const BASE = import.meta.env.VITE_API_URL ?? "/api";
const KEY = import.meta.env.VITE_API_KEY ?? "demo-key";

export type Dataset = { id: number; name: string; generator: string | null; created_at: string; health_score: number | null };
export type DriftFeature = { feature: string; psi: number; ks_p: number | null; drifted: boolean };
export type Finding = { type: string; feature: string | null; severity: string; message: string };
export type ModelResult = { degraded: boolean; deltas: Record<string, number>; current: Record<string, number> };
export type Health = {
  dataset_id: number;
  name: string;
  checked_at: string;
  health_score: number;
  feature_health: Record<string, "ok" | "warn" | "bad">;
  drift: { features: DriftFeature[]; label: { ref_rate: number | null; cur_rate: number | null; drifted: boolean }; max_psi: number };
  model: ModelResult | null;
  baseline: Record<string, number> | null;
  findings: Finding[];
};
export type Report = {
  alert: string;
  root_cause: string;
  likely_cause: string;
  evidence: string[];
  impact: string;
  recommended_action: string;
  confidence: number;
  source: "llm" | "rules";
};
export type Alert = {
  id: number;
  dataset_id: number;
  title: string;
  severity: "high" | "medium";
  status: "open" | "investigated";
  report: Report | null;
  latency_ms: number | null;
  created_at: string;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "X-API-Key": KEY, ...(init?.body ? { "Content-Type": "application/json" } : {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${(await res.json().catch(() => ({}))).detail ?? res.statusText}`);
  return res.json();
}

export const api = {
  datasets: () => call<Dataset[]>("/datasets"),
  health: (id: number) => call<Health>(`/datasets/${id}/health`),
  alerts: () => call<Alert[]>("/alerts"),
  investigate: (alert_id: number, use_llm: boolean) =>
    call<Report>("/investigations", { method: "POST", body: JSON.stringify({ alert_id, use_llm }) }),
  demo: (generator: string, failure: string, use_llm: boolean) =>
    call<{ dataset_id: number; alert_id: number | null }>("/demo/run", {
      method: "POST",
      body: JSON.stringify({ generator, failure, use_llm }),
    }),
};
