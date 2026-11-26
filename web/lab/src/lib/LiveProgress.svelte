<!--
  @component
  Progress of the running step: time, ETA, loss traces, finished starts and
  (for fits) the data with the best fit so far.
-->
<script lang="ts">
  import {
    api,
    formatSeconds,
    formatValue,
    type Live,
    type StepProgress,
  } from "./api.js";
  import Chart from "./Chart.svelte";
  import Panel from "./Panel.svelte";
  import { traceChart, waterfallChart } from "./charts.js";
  import { run } from "./run.svelte.js";

  let { step }: { step: string } = $props();

  let progress = $state<StepProgress | null>(null);
  let live = $state<Live | null>(null);
  let now = $state(Date.now() / 1000);

  $effect(() => {
    void run.progress;
    void run.version;
    void api
      .progress(step)
      .then((p) => (progress = p))
      .catch(() => {});
  });

  // The best fit so far: polled every 3 s while the step runs, fetched
  // only when the best objective changed
  $effect(() => {
    let last = "";
    const timer = setInterval(() => {
      if (progress === null || progress.finished) return;
      const best = JSON.stringify(progress.best);
      if (best === last) return;
      last = best;
      void api
        .live(step)
        .then((l) => (live = l))
        .catch(() => {});
    }, 3000);
    return () => clearInterval(timer);
  });

  $effect(() => {
    const timer = setInterval(() => (now = Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  });

  let elapsed = $derived(
    progress === null ? null : (progress.finished ?? now) - progress.started,
  );
  let eta = $derived.by(() => {
    if (progress?.expected == null || progress.finished || elapsed === null)
      return null;
    const [low, , high] = progress.expected;
    return [Math.max(low - elapsed, 0), Math.max(high - elapsed, 0)];
  });
  // Elapsed over the expected (median) duration, capped below 1 until done
  let fraction = $derived.by(() => {
    if (progress === null || elapsed === null) return null;
    if (progress.finished) return 1;
    if (progress.expected == null) return null;
    return Math.min(elapsed / Math.max(progress.expected[1], 1e-9), 0.97);
  });
</script>

<Panel
  title="Progress"
  titleId="progress-title"
>
  {#if progress}
    {#if fraction !== null}
      <div
        class="bar"
        role="progressbar"
        aria-label="Progress of the step"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
      >
        <div
          class="fill"
          style:transform={`scaleX(${fraction})`}
        ></div>
      </div>
    {/if}
    <dl>
      <div>
        <dt>Elapsed</dt>
        <dd>{formatSeconds(elapsed)}</dd>
      </div>
      <div>
        <dt>Remaining</dt>
        <dd>
          {#if eta}
            {formatSeconds(eta[0])} – {formatSeconds(eta[1])}
          {:else if progress.finished}
            done
          {:else}
            unknown
          {/if}
        </dd>
      </div>
      <div>
        <dt>Evaluations</dt>
        <dd>{progress.n_evaluations}</dd>
      </div>
      <div>
        <dt>Finished fits</dt>
        <dd>{progress.n_finished}</dd>
      </div>
      {#each Object.entries(progress.best) as [group, best] (group)}
        <div>
          <dt>Best {group}</dt>
          <dd>{formatValue(best)}</dd>
        </div>
      {/each}
    </dl>
  {:else}
    <p class="muted">Starting…</p>
  {/if}
</Panel>
{#if progress}
  <div class="charts">
    {#if Object.keys(progress.traces).length > 0}
      <Chart spec={traceChart(progress.traces)} />
    {/if}
    {#if Object.values(progress.waterfalls).some((w) => w.length > 0)}
      <Chart spec={waterfallChart(progress.waterfalls)} />
    {/if}
    {#if live}
      {#each live.charts as spec (spec.id)}
        <Chart spec={{ ...spec, title: `${spec.title}: best so far` }} />
      {/each}
    {/if}
  </div>
{/if}

<style>
  .bar {
    border-radius: var(--radius-full);
    background: var(--slate-50);
    height: 0.5rem;
    overflow: hidden;
  }
  .fill {
    transform-origin: left;
    transition: transform 200ms ease-out;
    background: var(--color-primary);
    height: 100%;
  }
  dl {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr));
    gap: var(--space-4);
    margin: 0;
  }
  dt {
    color: var(--color-text-muted);
    font-size: var(--text-callout);
  }
  dd {
    margin: 0;
    font-weight: var(--weight-medium);
    font-size: var(--text-h3);
    font-variant-numeric: tabular-nums;
  }
  .muted {
    margin: 0;
    color: var(--color-text-muted);
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 24rem), 1fr));
    gap: var(--space-4);
  }
  @media (prefers-reduced-motion: reduce) {
    .fill {
      transition: none;
    }
  }
</style>
