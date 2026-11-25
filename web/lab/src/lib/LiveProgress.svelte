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
</script>

<section aria-label="Progress">
  {#if progress}
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
  {:else}
    <p>Starting…</p>
  {/if}
</section>

<style>
  dl {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
    gap: var(--space-2);
    margin: 0 0 var(--space-3);
  }
  dt {
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
  dd {
    margin: 0 0 var(--space-1);
    font-weight: 600;
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 22rem), 1fr));
    gap: var(--space-3);
  }
</style>
