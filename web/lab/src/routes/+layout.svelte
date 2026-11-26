<script lang="ts">
  import "../app.css";
  import Button from "@computational-biology-aachen/design/Button.svelte";
  import InfoBox from "@computational-biology-aachen/design/InfoBox.svelte";
  import Navbar from "@computational-biology-aachen/design/Navbar.svelte";
  import SectionMain from "@computational-biology-aachen/design/SectionMain.svelte";
  import { api, formatSeconds, type Tone } from "#lib/api.js";
  import { run } from "#lib/run.svelte.js";
  import StatusBadge from "#lib/StatusBadge.svelte";

  let { children } = $props();

  $effect(() => run.connect());

  let overview = $derived(run.overview);
  let running = $derived(overview?.progress.running?.step ?? null);
  let runningTitle = $derived(
    overview?.steps.find((s) => s.step === running)?.title ?? running,
  );
  let gateTitle = $derived(
    overview?.steps.find((s) => s.step === overview?.open_gate)?.title ??
      overview?.open_gate,
  );

  let state = $derived.by((): { tone: Tone; text: string } | null => {
    if (!overview) return null;
    if (overview.paused)
      return {
        tone: "neutral",
        text: running ? `pausing after ${runningTitle}…` : "paused",
      };
    if (running) return { tone: "running", text: `running ${runningTitle}` };
    if (overview.finished) return { tone: "passed", text: "finished" };
    if (overview.open_gate)
      return { tone: "neutral", text: `waiting at ${gateTitle}` };
    if (!overview.driving && !overview.driven)
      return { tone: "neutral", text: "idle" };
    return null;
  });
</script>

<Navbar>
  {#snippet brand()}
    <a
      class="brand"
      href="/"
    >
      <span class="wordmark">MxlPy</span>
      <span class="product">Fitting lab</span>
    </a>
  {/snippet}
  <div
    class="state"
    role="status"
  >
    {#if state}
      <StatusBadge tone={state.tone}>{state.text}</StatusBadge>
    {/if}
    {#if overview && !run.connected}
      <StatusBadge tone="neutral">reconnecting</StatusBadge>
    {/if}
  </div>
</Navbar>

<SectionMain
  pad="tight"
  styleVars={{ innerAlign: "stretch" }}
>
  {#if overview}
    <div class="toolbar">
      <div class="run">
        <span
          class="run-dir"
          title={overview.run_dir}
        >
          {overview.run_dir}
        </span>
        <span class="muted">
          Computing time {formatSeconds(overview.active_time)}
          {#if overview.time_budget !== null}
            of {formatSeconds(overview.time_budget)}
          {/if}
        </span>
      </div>
      <div class="actions">
        {#if overview.can_run && !overview.driving && !overview.driven}
          <Button
            variant="primary"
            onclick={() => void run.act(() => api.run())}
          >
            Run
          </Button>
        {/if}
        {#if overview.paused}
          <Button
            variant="secondary"
            onclick={() => void run.act(api.unpause)}
          >
            Unpause
          </Button>
        {:else}
          <Button
            variant="secondary"
            onclick={() => void run.act(api.pause)}
          >
            Pause
          </Button>
        {/if}
        <Button
          variant="secondary"
          href="/api/report"
        >
          Report
        </Button>
      </div>
    </div>
  {/if}
  {#if run.unauthorized}
    <InfoBox
      header="Access token needed"
      variant="warning"
    >
      Open the address with the token that <code>mxlpy lab serve</code> or
      <code>run.serve()</code> printed.
    </InfoBox>
  {:else}
    {#if run.error}
      <InfoBox
        header="Error"
        variant="error"
      >
        {run.error}
      </InfoBox>
    {/if}
    {#if overview?.progress.error}
      <InfoBox
        header={`Step ${overview.progress.error.step} failed`}
        variant="error"
      >
        <pre>{overview.progress.error.message}</pre>
      </InfoBox>
    {/if}
    {#if overview?.driver_error}
      <InfoBox
        header="Not running the steps"
        variant="warning"
      >
        {overview.driver_error}
      </InfoBox>
    {/if}
    {@render children()}
  {/if}
</SectionMain>

<style>
  .brand {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
    color: var(--color-text);
    text-decoration: none;
    white-space: nowrap;
  }
  .wordmark {
    color: var(--color-primary);
    font-weight: var(--weight-bold);
    font-size: var(--text-h1);
  }
  .product {
    color: var(--color-text-muted);
    font-weight: var(--weight-medium);
  }
  .state {
    display: flex;
    flex-wrap: wrap;
    justify-content: end;
    gap: var(--space-1);
    min-width: 0;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-2) var(--space-4);
    width: 100%;
  }
  .run {
    display: flex;
    flex: 1 1 16rem;
    flex-direction: column;
    min-width: 0;
    font-size: var(--text-sm);
  }
  .run-dir {
    overflow: hidden;
    font-family: var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .muted {
    color: var(--color-text-muted);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .actions :global(:is(button, a)) {
    width: auto;
  }
  pre {
    margin: 0;
    white-space: pre-wrap;
  }
</style>
