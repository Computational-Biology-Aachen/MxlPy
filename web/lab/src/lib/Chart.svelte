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
    margin: 0;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    padding: var(--space-3);
    min-width: 0;
  }
  figcaption {
    margin-bottom: var(--space-2);
    font-weight: 600;
  }
  .plot {
    position: relative;
    height: 260px;
  }
  .empty {
    color: var(--color-text-muted);
  }
</style>
