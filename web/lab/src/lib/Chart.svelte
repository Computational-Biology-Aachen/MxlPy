<!--
  @component
  A chart of mxlpy.lab.views: x-y charts with PlotChart, categories with
  StackedBarChart.
-->
<script lang="ts">
  import PlotChart from "@computational-biology-aachen/design/PlotChart.svelte";
  import StackedBarChart from "@computational-biology-aachen/design/StackedBarChart.svelte";
  import type { ChartSpec } from "./api.js";
  import { barProps, xyProps } from "./charts.js";

  let { spec }: { spec: ChartSpec } = $props();

  let xy = $derived(spec.kind === "xy" ? xyProps(spec) : null);
  let bars = $derived(spec.kind === "category" ? barProps(spec) : null);
</script>

<figure>
  <figcaption>{spec.title}</figcaption>
  <div class="plot">
    {#if xy && xy.series.length > 0}
      <PlotChart
        series={xy.series}
        xLabel={xy.xLabel}
        yLabel={xy.yLabel}
        logX={xy.logX}
        logY={xy.logY}
        xLim={xy.xLim}
        yLim={xy.yLim}
        hLines={xy.hLines}
        legend={xy.legend}
        margin={0.05}
      />
    {:else if bars && bars.categories.length > 0}
      <StackedBarChart
        categories={bars.categories}
        stacks={bars.stacks}
        yLabel={bars.yLabel}
      />
    {:else}
      <p class="empty">No data yet</p>
    {/if}
  </div>
</figure>

<style>
  figure {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    margin: 0;
    box-shadow: var(--shadow-sm);
    border-radius: var(--radius-lg);
    background: var(--color-surface);
    padding: var(--space-4) var(--space-4) var(--space-3);
    min-width: 0;
  }
  figcaption {
    font-weight: var(--weight-medium);
    font-size: var(--text-sm);
  }
  .plot {
    position: relative;
    min-height: 300px;
  }
  .empty {
    display: grid;
    place-items: center;
    margin: 0;
    height: 300px;
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
</style>
