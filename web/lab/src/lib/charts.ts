// Chart specs of mxlpy.lab.views → props of the design system's charts.

import type {
  PlotRule,
  PlotSeries,
} from "@computational-biology-aachen/design/PlotChart.svelte";
import type { ChartSpec, Series } from "./api.js";

/**
 * RWTH palette, starting with the design's chart colours --color-c1..c4
 * (petrol, violet, blue, magenta); index 7 is neutral (e.g. parameters that
 * stay fixed).
 */
export const PALETTE = [
  "rgb(0, 97, 101)",
  "rgb(97, 33, 88)",
  "rgb(0, 84, 159)",
  "rgb(227, 0, 102)",
  "rgb(246, 168, 0)",
  "rgb(87, 171, 39)",
  "rgb(161, 16, 53)",
  "rgb(148, 163, 184)",
  "rgb(0, 152, 161)",
  "rgb(122, 111, 172)",
];

export function color(i: number): string {
  return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

type Point = [number, number, number?, number?];

function finitePoints(s: Series): Point[] {
  const out: Point[] = [];
  s.x.forEach((x, i) => {
    const y = s.y[i];
    if (typeof x !== "number" || y === null || !Number.isFinite(y)) return;
    if (s.type === "band") {
      const lo = s.lower?.[i];
      const hi = s.upper?.[i];
      if (lo === null || lo === undefined || hi === null || hi === undefined)
        return;
      out.push([x, y, lo, hi]);
    } else {
      out.push([x, y]);
    }
  });
  return out.sort((a, b) => a[0] - b[0]);
}

/** Log axes only help (and get tick labels) over at least a decade. */
function spansDecade(values: number[]): boolean {
  if (values.length === 0 || values.some((v) => !(v > 0))) return false;
  return Math.max(...values) / Math.min(...values) >= 10;
}

/**
 * Relative spread below which the lab draws an axis as flat: objectives that
 * differ in the seventh digit are the same fit, and an axis over such a range
 * (live traces of refits that start at the optimum span ~1e-14) has no
 * readable ticks. Stricter than PlotChart's own guard (1e-9).
 */
const FLAT = 1e-6;

/** Limits around the value of flat data (± 5 %), undefined otherwise. */
export function flatLimits(values: number[]): [number, number] | undefined {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!Number.isFinite(lo)) return undefined;
  const size = Math.max(Math.abs(lo), Math.abs(hi));
  if (hi - lo > FLAT * size) return undefined;
  const mid = (lo + hi) / 2;
  const pad = size > 0 ? 0.05 * size : 1;
  return [mid - pad, mid + pad];
}

export interface XYProps {
  series: PlotSeries[];
  xLabel: string;
  yLabel: string;
  logX: boolean;
  logY: boolean;
  /** Fixed limits of flat data (see `flatLimits`) */
  xLim?: [number, number];
  yLim?: [number, number];
  hLines: PlotRule[];
  legend: boolean;
}

export function xyProps(spec: ChartSpec): XYProps {
  const series: PlotSeries[] = [];
  const labelled = new Set<string>();
  for (const s of spec.series) {
    const pts = finitePoints(s);
    if (pts.length === 0) continue;
    const c = color(s.color);
    // One legend entry per label (data and prediction share a colour)
    const label = labelled.has(s.label) ? undefined : s.label;
    labelled.add(s.label);
    if (s.type === "band") {
      series.push({
        label: `${s.label} (interval)`,
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[3] as number),
        fillTo: pts.map((p) => p[2] as number),
        color: c,
        alpha: 0.25,
        lineWidth: 0,
        zorder: 0,
      });
      series.push({
        label,
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        color: c,
        zorder: 1,
      });
    } else if (s.type === "points") {
      series.push({
        label,
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        style: "markers",
        marker: "o",
        markerSize: 3,
        color: c,
        alpha: 0.7,
        zorder: 2,
      });
    } else {
      series.push({
        label,
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        color: c,
        dash: s.dashed ? "dashed" : "solid",
        zorder: 1,
      });
    }
  }
  const xs = series.flatMap((s) => s.x);
  const ys = [
    ...series.flatMap((s) => [...s.y, ...(s.fillTo ?? [])]),
    ...spec.hlines.map((h) => h.y),
  ];
  const xLim = flatLimits(xs);
  const yLim = flatLimits(ys);
  const logX = spec.x.log && !xLim && spansDecade(xs);
  const logY = spec.y.log && !yLim && spansDecade(series.flatMap((s) => s.y));
  return {
    series,
    xLabel: spec.x.label,
    yLabel: spec.y.label,
    logX,
    logY,
    xLim,
    yLim,
    hLines: spec.hlines.map((h) => ({
      value: h.y,
      color: "rgb(100 116 139)",
      dash: "dashed",
    })),
    legend: labelled.size > 1,
  };
}

export interface BarProps {
  categories: string[];
  stacks: { label: string; values: number[]; color: string }[];
  yLabel: string;
}

export function barProps(spec: ChartSpec): BarProps {
  const categories: string[] = [];
  for (const s of spec.series) {
    for (const x of s.x) {
      const key = String(x);
      if (!categories.includes(key)) categories.push(key);
    }
  }
  const stacks = spec.series.map((s) => {
    const values = categories.map((cat) => {
      const i = s.x.findIndex((x) => String(x) === cat);
      return i < 0 ? 0 : (s.y[i] ?? 0);
    });
    return { label: s.label, values, color: color(s.color) };
  });
  return { categories, stacks, yLabel: spec.y.label };
}

/** Loss traces of a running step as a chart spec (best objective over time). */
export function traceChart(
  traces: Record<string, [number, number][]>,
): ChartSpec {
  const series: Series[] = Object.entries(traces).map(([group, t], i) => ({
    label: group,
    type: "line",
    x: t.map((p) => p[0]),
    y: t.map((p) => p[1]),
    color: i,
    dashed: false,
  }));
  const positive = series.every((s) => s.y.every((v) => v === null || v > 0));
  return {
    id: "trace",
    title: "Best objective",
    kind: "xy",
    x: { label: "seconds", log: false },
    y: { label: "objective", log: positive },
    series,
    hlines: [],
  };
}

/** Sorted final objectives of the finished starts of every fit group. */
export function waterfallChart(
  waterfalls: Record<string, number[]>,
): ChartSpec {
  const series: Series[] = Object.entries(waterfalls)
    .filter(([, v]) => v.length > 0)
    .map(([group, v], i) => ({
      label: group,
      type: "points",
      x: v.map((_, j) => j + 1),
      y: v,
      color: i,
      dashed: false,
    }));
  const positive = series.every((s) => s.y.every((v) => v === null || v > 0));
  return {
    id: "waterfall",
    title: "Finished starts",
    kind: "xy",
    x: { label: "start (sorted)", log: false },
    y: { label: "objective", log: positive },
    series,
    hlines: [],
  };
}
