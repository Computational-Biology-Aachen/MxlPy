// Types and requests of the lab's API (mxlpy.lab.server, mxlpy.lab.views).

export type Status = "pending" | "passed" | "failed" | "done" | "skipped";
export type Gate = "user" | "auto" | "implicit" | null;

export interface StepSummary {
  step: string;
  number: number;
  title: string;
  status: Status;
  current: boolean;
  duration: number | null;
  messages: string[];
  gate: Gate;
  has_options: boolean;
}

export interface StepProgress {
  step: string;
  started: number;
  finished: number | null;
  elapsed: number;
  expected: [number, number, number] | null;
  eta: [number, number] | null;
  passed: boolean | null;
  n_evaluations: number;
  n_finished: number;
  best: Record<string, number>;
  traces: Record<string, [number, number][]>;
  waterfalls: Record<string, number[]>;
  n_failed: Record<string, number>;
  version: number;
}

export interface RunState {
  running: StepProgress | null;
  gate: string | null;
  error: { step: string; message: string } | null;
  finished: boolean;
}

export interface Overview {
  run_dir: string;
  steps: StepSummary[];
  /** First step with a result but no decision. */
  open_gate: string | null;
  /** Every step is up to date and decided. */
  finished: boolean;
  paused: boolean;
  driven: boolean;
  driving: boolean;
  /** The server may run the steps (not watch-only). */
  can_run: boolean;
  driver_error: string | null;
  active_time: number;
  time_budget: number | null;
  config: Record<string, unknown>;
  progress: RunState;
}

export interface Axis {
  label: string;
  log: boolean;
}

export interface Series {
  label: string;
  type: "points" | "line" | "band" | "bar";
  x: (number | string | null)[];
  y: (number | null)[];
  lower?: (number | null)[];
  upper?: (number | null)[];
  color: number;
  dashed: boolean;
}

export interface ChartSpec {
  id: string;
  title: string;
  kind: "xy" | "category";
  x: Axis;
  y: Axis;
  series: Series[];
  hlines: { y: number; label: string }[];
}

export interface TableSpec {
  id: string;
  title: string;
  columns: string[];
  rows: unknown[][];
}

export interface Option {
  kind: "subset" | "choice" | "flag" | "mapping";
  choices: unknown[];
  description: string;
}

export interface Decision {
  values: Record<string, unknown>;
  source: Exclude<Gate, null>;
  basis: string | null;
}

export interface StepView {
  step: string;
  number: number;
  title: string;
  status: Status;
  current: boolean;
  result: {
    passed: boolean | null;
    summary: Record<string, unknown>;
    proposal: Record<string, unknown>;
    options: Record<string, Option>;
    messages: string[];
    duration: number;
    skipped: boolean;
  } | null;
  decision: Decision | null;
  charts: ChartSpec[];
  tables: TableSpec[];
}

export interface Live {
  group: string;
  best: number;
  charts: ChartSpec[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  fetcher: typeof fetch = fetch,
): Promise<T> {
  const res = await fetcher(`/api/${path}`, {
    credentials: "same-origin",
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(body.error ?? res.statusText, res.status);
  }
  return body as T;
}

export const api = {
  overview: (f?: typeof fetch) => request<Overview>("overview", {}, f),
  step: (step: string, f?: typeof fetch) =>
    request<StepView>(`steps/${step}`, {}, f),
  progress: (step: string) => request<StepProgress | null>(`progress/${step}`),
  live: (step: string) => request<Live | null>(`live/${step}`),
  decide: (step: string, values: Record<string, unknown>) =>
    request(`steps/${step}/decision`, {
      method: "POST",
      body: JSON.stringify({ values }),
    }),
  reopen: (step: string) =>
    request(`steps/${step}/decision`, { method: "DELETE" }),
  pause: () => request("pause", { method: "POST" }),
  unpause: () => request("unpause", { method: "POST" }),
  run: (auto?: boolean) =>
    request("run", { method: "POST", body: JSON.stringify({ auto }) }),
};

export function formatSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "–";
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 90) return `${s.toFixed(1)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)
      ? v.toExponential(3)
      : v.toPrecision(4);
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
