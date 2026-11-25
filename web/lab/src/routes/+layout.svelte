<script lang="ts">
  import "../app.css";
  import Button from "@computational-biology-aachen/design/Button.svelte";
  import InfoBox from "@computational-biology-aachen/design/InfoBox.svelte";
  import { api } from "#lib/api.js";
  import { run } from "#lib/run.svelte.js";

  let { children } = $props();

  $effect(() => run.connect());

  let overview = $derived(run.overview);
  let running = $derived(overview?.progress.running?.step ?? null);
</script>

<header>
  <div class="inner">
    <a
      class="brand"
      href="/"
    >
      Fitting lab
    </a>
    {#if overview}
      <span
        class="run-dir"
        title={overview.run_dir}
      >
        {overview.run_dir}
      </span>
      <span
        class="state"
        role="status"
      >
        {#if overview.paused}
          {running ? `pausing after ${running}…` : "paused"}
        {:else if running}
          running {running}
        {:else if overview.finished}
          finished
        {:else if overview.open_gate}
          waiting at {overview.open_gate}
        {:else if !overview.driving && !overview.driven}
          idle
        {/if}
        {#if !run.connected}· reconnecting{/if}
      </span>
      <div class="actions">
        {#if overview.paused}
          <Button
            variant="inverted"
            onclick={() => void run.act(api.unpause)}
          >
            Unpause
          </Button>
        {:else}
          <Button
            variant="inverted"
            onclick={() => void run.act(api.pause)}
          >
            Pause
          </Button>
        {/if}
        {#if overview.can_run && !overview.driving && !overview.driven}
          <Button
            variant="inverted"
            onclick={() => void run.act(() => api.run())}
          >
            Run
          </Button>
        {/if}
        <Button
          variant="inverted"
          href="/api/report"
        >
          Report
        </Button>
      </div>
    {/if}
  </div>
</header>

<main>
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
</main>

<style>
  header {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--color-primary);
    color: var(--color-text-inverse);
  }
  .inner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-4);
    margin: 0 auto;
    padding: var(--space-2) var(--space-4);
    max-width: 1400px;
  }
  .brand {
    color: inherit;
    font-weight: 700;
    font-size: var(--text-h3);
    text-decoration: none;
  }
  .run-dir {
    flex: 1 1 10rem;
    min-width: 0;
    overflow: hidden;
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  main {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    margin: 0 auto;
    padding: var(--space-4);
    max-width: 1400px;
  }
  pre {
    white-space: pre-wrap;
  }
  @media (max-width: 600px) {
    .inner,
    main {
      padding: var(--space-2) var(--space-4);
    }
  }
</style>
