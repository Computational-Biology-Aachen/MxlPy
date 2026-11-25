<!--
  @component
  The steps of the run with their status and gate.
-->
<script lang="ts">
  import { formatSeconds, type Overview } from "./api.js";

  let {
    overview,
    selected = null,
  }: { overview: Overview; selected?: string | null } = $props();

  let running = $derived(overview.progress.running?.step ?? null);

  function gateLabel(
    step: Overview["steps"][number],
  ): { text: string; open: boolean } | null {
    if (step.status === "pending") return null;
    if (step.gate === null) return { text: "waiting for you", open: true };
    return { text: `decided (${step.gate})`, open: false };
  }
</script>

<nav aria-label="Steps">
  <ol>
    {#each overview.steps as step (step.step)}
      {@const gate = gateLabel(step)}
      <li>
        <a
          href={`/steps/${step.step}`}
          aria-current={selected === step.step ? "page" : undefined}
        >
          <span class="number">{step.number}</span>
          <span class="title">{step.title}</span>
          <span class="status {step.status}">
            {running === step.step ? "running" : step.status}
            {#if step.status !== "pending" && !step.current && running !== step.step}
              · outdated
            {/if}
          </span>
          <span class="meta">
            {#if step.duration !== null}{formatSeconds(step.duration)}{/if}
            {#if gate}
              <span class:open={gate.open}>{gate.text}</span>
            {/if}
          </span>
        </a>
      </li>
    {/each}
  </ol>
</nav>

<style>
  ol {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  a {
    display: grid;
    grid-template-columns: 2rem 1fr auto;
    gap: 0 var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    padding: var(--space-2) var(--space-3);
    color: inherit;
    text-decoration: none;
  }
  a:hover,
  a:focus-visible {
    border-color: var(--color-primary);
  }
  a[aria-current="page"] {
    box-shadow: inset 4px 0 0 var(--color-primary);
  }
  .number {
    grid-row: span 2;
    align-self: center;
    color: var(--color-text-muted);
    font-weight: 600;
    font-size: var(--text-h3);
  }
  .title {
    font-weight: 600;
  }
  .status {
    font-size: var(--text-sm);
    text-align: right;
  }
  .status.passed {
    color: var(--rwth-green);
  }
  .status.failed {
    color: var(--color-danger);
  }
  .status.pending,
  .status.skipped {
    color: var(--color-text-muted);
  }
  .meta {
    display: flex;
    grid-column: 2 / span 2;
    justify-content: space-between;
    gap: var(--space-2);
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
  .meta .open {
    color: var(--color-text);
    font-weight: 600;
  }
</style>
