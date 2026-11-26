<!--
  @component
  The steps of the run with their status and gate, as a vertical stepper.
-->
<script lang="ts">
  import { formatSeconds, stepBadge, type Overview } from "./api.js";
  import StatusBadge from "./StatusBadge.svelte";

  let {
    overview,
    selected = null,
  }: { overview: Overview; selected?: string | null } = $props();

  let running = $derived(overview.progress.running?.step ?? null);
</script>

<nav aria-label="Steps">
  <ol>
    {#each overview.steps as step (step.step)}
      {@const isRunning = running === step.step}
      {@const waiting =
        !isRunning && step.status !== "pending" && step.gate === null}
      {@const badge = stepBadge(step, isRunning, waiting)}
      <li
        class={badge.tone}
        class:muted={step.status === "pending" || step.status === "skipped"}
      >
        <a
          href={`/steps/${step.step}`}
          aria-current={selected === step.step ? "page" : undefined}
          {@attach (node) => {
            // keeps the current step visible in the horizontal mobile strip
            if (selected === step.step)
              node.scrollIntoView({ block: "nearest", inline: "nearest" });
          }}
        >
          <span
            class="number"
            aria-hidden="true"
          >
            {step.number}
          </span>
          <span class="title">{step.title}</span>
          <span class="meta">
            <StatusBadge tone={badge.tone}>{badge.text}</StatusBadge>
            {#if waiting && step.status === "failed"}
              <StatusBadge tone="failed">failed</StatusBadge>
            {/if}
            {#if step.duration !== null}
              <span>{formatSeconds(step.duration)}</span>
            {/if}
            {#if step.gate !== null && step.status !== "pending"}
              <span>decided ({step.gate})</span>
            {/if}
            {#if step.status !== "pending" && !step.current && !isRunning}
              <span>outdated</span>
            {/if}
          </span>
        </a>
      </li>
    {/each}
  </ol>
</nav>

<style>
  nav {
    box-shadow: var(--shadow-sm);
    border-radius: var(--radius-lg);
    background: var(--color-surface);
    padding: var(--space-2);
  }
  ol {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  li {
    position: relative;
  }
  /* the connector between the step circles */
  li:not(:last-child)::after {
    position: absolute;
    top: calc(var(--space-3) + 1.75rem);
    bottom: calc(-1 * var(--space-3));
    left: calc(var(--space-3) + 0.875rem - 1px);
    background: var(--color-border);
    width: 2px;
    content: "";
  }
  li.passed:not(:last-child)::after {
    background: var(--color-primary);
  }
  a {
    display: grid;
    grid-template-columns: 1.75rem minmax(0, 1fr);
    align-items: center;
    gap: var(--space-1) var(--space-3);
    transition: background-color var(--transition);
    border-radius: var(--radius-md);
    padding: var(--space-3);
    color: inherit;
    text-decoration: none;
  }
  a:hover {
    background: var(--color-bg);
  }
  a[aria-current="page"] {
    background: rgb(from var(--color-primary) r g b / 8%);
  }
  a[aria-current="page"] .title {
    color: var(--color-primary);
  }
  .number {
    display: grid;
    position: relative;
    place-items: center;
    z-index: 1;
    box-sizing: border-box;
    border: 2px solid var(--color-border);
    border-radius: var(--radius-full);
    background: var(--color-surface);
    width: 1.75rem;
    height: 1.75rem;
    color: var(--color-text-muted);
    font-weight: var(--weight-medium);
    font-size: var(--text-sm);
  }
  .passed .number {
    border-color: var(--color-primary);
    background: var(--color-primary);
    color: var(--color-text-inverse);
  }
  .failed .number {
    border-color: var(--color-danger);
    background: var(--color-danger);
    color: var(--color-text-inverse);
  }
  .running .number {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }
  .waiting .number {
    border-color: var(--color-accent);
    color: var(--color-text);
  }
  .title {
    font-weight: var(--weight-medium);
  }
  .muted .title {
    color: var(--color-text-muted);
  }
  .meta {
    display: flex;
    grid-column: 2;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    color: var(--color-text-muted);
    font-size: var(--text-callout);
  }
  /* On narrow screens the rail is a horizontal strip above the page */
  @media (max-width: 768px) {
    ol {
      display: flex;
      gap: var(--space-1);
      overflow-x: auto;
      scroll-snap-type: x proximity;
    }
    li {
      flex: 0 0 auto;
      scroll-snap-align: start;
    }
    li:not(:last-child)::after,
    .meta {
      display: none;
    }
    a {
      grid-template-columns: 1.75rem auto;
      gap: var(--space-2);
      padding: var(--space-2);
    }
    .title {
      font-size: var(--text-sm);
      white-space: nowrap;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    a {
      transition: none;
    }
  }
</style>
