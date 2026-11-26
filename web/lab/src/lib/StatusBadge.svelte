<!--
  @component
  A status pill: running, passed, failed, waiting for a decision, …
-->
<script lang="ts">
  import type { Snippet } from "svelte";
  import type { Tone } from "./api.js";

  let { tone, children }: { tone: Tone; children: Snippet } = $props();
</script>

<span class="badge {tone}">{@render children()}</span>

<style>
  .badge {
    --badge-color: var(--slate-500);
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    border-radius: var(--radius-full);
    background: rgb(from var(--badge-color) r g b / 12%);
    padding: 0.125rem var(--space-2);
    color: var(--badge-color);
    font-weight: var(--weight-medium);
    font-size: var(--text-callout);
    line-height: 1.4;
    white-space: nowrap;
  }
  .running {
    --badge-color: var(--color-primary);
  }
  .running::before {
    animation: pulse 1.2s ease-in-out infinite;
    border-radius: var(--radius-full);
    background: currentColor;
    width: 0.4rem;
    height: 0.4rem;
    content: "";
  }
  .passed {
    --badge-color: color-mix(in srgb, var(--rwth-green) 65%, black);
  }
  .failed {
    --badge-color: var(--color-danger);
  }
  .waiting {
    --badge-color: color-mix(in srgb, var(--color-accent) 55%, black);
    background: rgb(from var(--color-accent) r g b / 22%);
  }
  @keyframes pulse {
    50% {
      opacity: 0.3;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .running::before {
      animation: none;
    }
  }
</style>
