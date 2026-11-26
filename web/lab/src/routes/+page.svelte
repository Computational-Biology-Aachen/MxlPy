<script lang="ts">
  import Button from "@computational-biology-aachen/design/Button.svelte";
  import LabLayout from "#lib/LabLayout.svelte";
  import LiveProgress from "#lib/LiveProgress.svelte";
  import Panel from "#lib/Panel.svelte";
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
  <LabLayout>
    {#snippet rail()}
      <StepList overview={overview} />
    {/snippet}
    {#if running}
      <h1>
        Running: {overview.steps.find((s) => s.step === running)?.title}
      </h1>
      {#key running}
        <LiveProgress step={running} />
      {/key}
    {:else if gate}
      <Panel label="Next">
        <h1>{gateTitle} waits for your decision</h1>
        <p>
          Check its results, then accept the proposal or edit it. From Python:
          <code>run.decide("{gate}", ...)</code>.
        </p>
        <div class="cta">
          <Button
            variant="primary"
            href={`/steps/${gate}`}
          >
            Review the results and decide
          </Button>
        </div>
      </Panel>
    {:else if overview.finished}
      <Panel label="Next">
        <h1>The run is finished</h1>
        <p>Every step is decided. Open a step to review it.</p>
        <div class="cta">
          <Button
            variant="primary"
            href="/api/report"
          >
            Download the report
          </Button>
        </div>
      </Panel>
    {:else}
      <Panel label="Next">
        <h1>Overview</h1>
        <p class="muted">
          {#if overview.driving || overview.driven}
            Waiting for the next step…
          {:else}
            No process runs the steps. Press Run, or run the lab from Python.
          {/if}
        </p>
      </Panel>
    {/if}
  </LabLayout>
{:else if !run.error}
  <p class="muted">Loading…</p>
{/if}

<style>
  h1 {
    margin: 0;
    font-size: var(--text-h1);
  }
  p {
    margin: 0;
  }
  .muted {
    color: var(--color-text-muted);
  }
  .cta :global(:is(a, button)) {
    width: auto;
  }
</style>
