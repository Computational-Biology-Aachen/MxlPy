<script lang="ts">
  import { page } from "$app/state";
  import { api, formatSeconds, type StepView } from "#lib/api.js";
  import Chart from "#lib/Chart.svelte";
  import DataTable from "#lib/DataTable.svelte";
  import GatePanel from "#lib/GatePanel.svelte";
  import LiveProgress from "#lib/LiveProgress.svelte";
  import { run } from "#lib/run.svelte.js";
  import StepList from "#lib/StepList.svelte";

  let step = $derived(page.params.step ?? "");
  let view = $state<StepView | null>(null);
  let error = $state<string | null>(null);

  $effect(() => {
    void run.version;
    const name = step;
    api
      .step(name)
      .then((v) => {
        if (name === step) {
          view = v;
          error = null;
        }
      })
      .catch((e) => (error = e instanceof Error ? e.message : String(e)));
  });

  let running = $derived(run.overview?.progress.running?.step === view?.step);
</script>

<svelte:head>
  <title>{view ? `${view.number} · ${view.title}` : "Step"} – Fitting lab</title
  >
</svelte:head>

<div class="layout">
  <aside>
    {#if run.overview}
      <StepList
        overview={run.overview}
        selected={view?.step ?? null}
      />
    {/if}
  </aside>
  <section>
    {#if error}
      <p class="error">{error}</p>
    {:else if view}
      <h1>{view.number} · {view.title}</h1>
      <p class="status">
        <span class={view.status}>{running ? "running" : view.status}</span>
        {#if view.result}· {formatSeconds(view.result.duration)}{/if}
        {#if view.result && !view.current}· outdated: recomputed when the run
          gets here{/if}
      </p>
      {#if running}
        <LiveProgress step={view.step} />
      {/if}
      {#if view.result && !view.result.skipped}
        <ul class="messages">
          {#each view.result.messages as message, i (i)}
            <li>{message}</li>
          {/each}
        </ul>
        <GatePanel view={view} />
        <div class="charts">
          {#each view.charts as spec (spec.id)}
            <Chart spec={spec} />
          {/each}
        </div>
        <div class="tables">
          {#each view.tables as spec (spec.id)}
            <DataTable spec={spec} />
          {/each}
        </div>
      {:else if view.result?.skipped}
        <p class="muted">Not needed in this run.</p>
      {:else if !running}
        <p class="muted">Not run yet.</p>
      {/if}
    {/if}
  </section>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: minmax(16rem, 22rem) 1fr;
    align-items: start;
    gap: var(--space-6);
  }
  section {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  h1 {
    margin: 0;
    font-size: var(--text-h2);
  }
  .status {
    margin: 0;
    color: var(--color-text-muted);
  }
  .passed {
    color: var(--rwth-green);
    font-weight: 600;
  }
  .failed {
    color: var(--color-danger);
    font-weight: 600;
  }
  .messages {
    margin: 0;
    padding-left: var(--space-4);
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 24rem), 1fr));
    gap: var(--space-3);
  }
  .tables {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 28rem), 1fr));
    gap: var(--space-4);
  }
  .muted {
    color: var(--color-text-muted);
  }
  .error {
    color: var(--color-danger);
  }
  @media (max-width: 800px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
</style>
