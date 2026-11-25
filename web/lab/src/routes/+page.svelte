<script lang="ts">
  import { formatSeconds } from "#lib/api.js";
  import LiveProgress from "#lib/LiveProgress.svelte";
  import { run } from "#lib/run.svelte.js";
  import StepList from "#lib/StepList.svelte";

  let overview = $derived(run.overview);
  let running = $derived(overview?.progress.running?.step ?? null);
  let gate = $derived(overview?.open_gate ?? null);
  let gateTitle = $derived(
    overview?.steps.find((s) => s.step === gate)?.title ?? gate,
  );
</script>

<svelte:head><title>Fitting lab</title></svelte:head>

{#if overview}
  <div class="layout">
    <aside>
      <StepList overview={overview} />
      <p class="muted">
        Computing time {formatSeconds(overview.active_time)}
        {#if overview.time_budget !== null}
          of a budget of {formatSeconds(overview.time_budget)}
        {/if}
      </p>
    </aside>
    <section>
      {#if running}
        <h1>
          Running: {overview.steps.find((s) => s.step === running)?.title}
        </h1>
        {#key running}
          <LiveProgress step={running} />
        {/key}
      {:else if gate}
        <h1>{gateTitle} waits for your decision</h1>
        <p>
          <a href={`/steps/${gate}`}>Review the results and decide</a>, or
          decide from Python with <code>run.decide("{gate}", ...)</code>.
        </p>
      {:else if overview.finished}
        <h1>The run is finished</h1>
        <p>
          Every step is decided. <a href="/api/report">Download the report</a>
          or open a step to review it.
        </p>
      {:else}
        <h1>Overview</h1>
        <p class="muted">
          {#if overview.driving || overview.driven}
            Waiting for the next step…
          {:else}
            No process runs the steps. Press Run, or run the lab from Python.
          {/if}
        </p>
      {/if}
    </section>
  </div>
{:else if !run.error}
  <p>Loading…</p>
{/if}

<style>
  .layout {
    display: grid;
    grid-template-columns: minmax(16rem, 22rem) 1fr;
    align-items: start;
    gap: var(--space-6);
  }
  section {
    min-width: 0;
  }
  h1 {
    margin-top: 0;
    font-size: var(--text-h2);
  }
  .muted {
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
  @media (max-width: 800px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
</style>
