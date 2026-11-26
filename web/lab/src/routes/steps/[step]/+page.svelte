<script lang="ts">
  import { page } from "$app/state";
  import { api, formatSeconds, stepBadge, type StepView } from "#lib/api.js";
  import Chart from "#lib/Chart.svelte";
  import DataTable from "#lib/DataTable.svelte";
  import GatePanel from "#lib/GatePanel.svelte";
  import LabLayout from "#lib/LabLayout.svelte";
  import LiveProgress from "#lib/LiveProgress.svelte";
  import Panel from "#lib/Panel.svelte";
  import { run } from "#lib/run.svelte.js";
  import StatusBadge from "#lib/StatusBadge.svelte";
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
  let badge = $derived(
    view
      ? stepBadge(
          view,
          running,
          !running && view.result !== null && view.decision === null,
        )
      : null,
  );
</script>

<svelte:head>
  <title>{view ? `${view.number} · ${view.title}` : "Step"} – Fitting lab</title
  >
</svelte:head>

<LabLayout>
  {#snippet rail()}
    {#if run.overview}
      <StepList
        overview={run.overview}
        selected={view?.step ?? null}
      />
    {/if}
  {/snippet}
  {#if error}
    <p class="error">{error}</p>
  {:else if view}
    <hgroup>
      <h1>{view.number} · {view.title}</h1>
      <p class="status">
        {#if badge}
          <StatusBadge tone={badge.tone}>{badge.text}</StatusBadge>
        {/if}
        {#if badge?.tone === "waiting" && view.status === "failed"}
          <StatusBadge tone="failed">failed</StatusBadge>
        {/if}
        {#if view.result}<span>{formatSeconds(view.result.duration)}</span>{/if}
        {#if view.result && !view.current}
          <span>outdated: recomputed when the run gets here</span>
        {/if}
      </p>
    </hgroup>
    {#if running}
      <LiveProgress step={view.step} />
    {/if}
    {#if view.result && !view.result.skipped}
      {#if view.result.messages.length > 0}
        <Panel
          title="Summary"
          titleId="summary-title"
        >
          <ul class="messages">
            {#each view.result.messages as message, i (i)}
              <li>{message}</li>
            {/each}
          </ul>
        </Panel>
      {/if}
      <GatePanel view={view} />
      {#if view.charts.length > 0}
        <div class="charts">
          {#each view.charts as spec (spec.id)}
            <Chart spec={spec} />
          {/each}
        </div>
      {/if}
      {#if view.tables.length > 0}
        <div class="tables">
          {#each view.tables as spec (spec.id)}
            <DataTable spec={spec} />
          {/each}
        </div>
      {/if}
    {:else if view.result?.skipped}
      <p class="muted">Not needed in this run.</p>
    {:else if !running}
      <p class="muted">Not run yet.</p>
    {/if}
  {/if}
</LabLayout>

<style>
  hgroup {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  h1 {
    margin: 0;
    font-size: var(--text-h1);
  }
  .status {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-3);
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
  .messages {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin: 0;
    padding-left: var(--space-4);
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 24rem), 1fr));
    gap: var(--space-4);
  }
  .tables {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 28rem), 1fr));
    align-items: start;
    gap: var(--space-4);
  }
  .muted {
    color: var(--color-text-muted);
  }
  .error {
    color: var(--color-danger);
  }
</style>
