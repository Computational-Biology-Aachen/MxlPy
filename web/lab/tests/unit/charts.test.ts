import { describe, expect, it } from "vitest";
import type { ChartSpec } from "../../src/lib/api.js";
import {
  barProps,
  color,
  PALETTE,
  traceChart,
  waterfallChart,
  xyProps,
} from "../../src/lib/charts.js";

const base: Omit<ChartSpec, "series"> = {
  id: "c",
  title: "c",
  kind: "xy",
  x: { label: "time", log: false },
  y: { label: "value", log: true },
  hlines: [{ y: 1.9, label: "threshold" }],
};

describe("xyProps", () => {
  it("draws data as markers and predictions as lines, one legend entry each", () => {
    const props = xyProps({
      ...base,
      series: [
        {
          label: "x",
          type: "points",
          x: [2, 1, 3],
          y: [0.5, null, 0.2],
          color: 0,
          dashed: false,
        },
        {
          label: "x",
          type: "line",
          x: [1, 2],
          y: [5, 0.5],
          color: 0,
          dashed: true,
        },
      ],
    });
    expect(props.series).toHaveLength(2);
    const [points, line] = props.series;
    expect(points.style).toBe("markers");
    // non-finite points are dropped, the rest sorted by x
    expect(points.x).toEqual([2, 3]);
    expect(points.label).toBe("x");
    expect(line.label).toBeUndefined();
    expect(line.dash).toBe("dashed");
    expect(props.legend).toBe(false);
    // positive over more than a decade (0.2 to 5)
    expect(props.logY).toBe(true);
    expect(props.hLines[0].value).toBe(1.9);
  });

  it("fills bands between lower and upper and drops log scales for non-positive data", () => {
    const props = xyProps({
      ...base,
      series: [
        {
          label: "band",
          type: "band",
          x: [0, 1],
          y: [0, 1],
          lower: [-1, 0],
          upper: [1, 2],
          color: 2,
          dashed: false,
        },
      ],
    });
    expect(props.series[0].fillTo).toEqual([-1, 0]);
    expect(props.series[0].y).toEqual([1, 2]);
    expect(props.series[1].y).toEqual([0, 1]);
    expect(props.logY).toBe(false);
  });
});

describe("barProps", () => {
  it("stacks series over the union of categories", () => {
    const props = barProps({
      ...base,
      kind: "category",
      series: [
        {
          label: "estimated",
          type: "bar",
          x: ["k1", "k2"],
          y: [3, 2],
          color: 0,
          dashed: false,
        },
        {
          label: "fixed",
          type: "bar",
          x: ["k3"],
          y: [0.1],
          color: 7,
          dashed: false,
        },
      ],
    });
    expect(props.categories).toEqual(["k1", "k2", "k3"]);
    expect(props.stacks[0].values).toEqual([3, 2, 0]);
    expect(props.stacks[1].values).toEqual([0, 0, 0.1]);
    expect(props.stacks[1].color).toBe(PALETTE[7]);
  });
});

describe("progress charts", () => {
  it("builds traces and waterfalls per fit group", () => {
    const trace = traceChart({
      start: [
        [0, 10],
        [1, 2],
      ],
      "a/start": [[0.5, 1]],
    });
    expect(trace.series.map((s) => s.label)).toEqual(["start", "a/start"]);
    expect(trace.y.log).toBe(true);
    const fall = waterfallChart({ start: [1, 2, 5], empty: [] });
    expect(fall.series).toHaveLength(1);
    expect(fall.series[0].x).toEqual([1, 2, 3]);
  });

  it("wraps colour indices", () => {
    expect(color(PALETTE.length)).toBe(PALETTE[0]);
    expect(color(-1)).toBe(PALETTE[PALETTE.length - 1]);
  });
});
