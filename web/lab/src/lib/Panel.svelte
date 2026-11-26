<!--
  @component
  A surface panel of the lab (mxl-web's dashboard panels): an optional title
  row with actions, then the content.
-->
<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    title,
    titleId,
    level = 2,
    label,
    actions,
    children,
  }: {
    title?: string;
    titleId?: string;
    level?: 2 | 3;
    /** Accessible name of the region when it has no title */
    label?: string;
    actions?: Snippet;
    children: Snippet;
  } = $props();
</script>

<section
  class="panel"
  aria-labelledby={title ? titleId : undefined}
  aria-label={title ? undefined : label}
>
  {#if title || actions}
    <header>
      {#if title}
        <svelte:element
          this={`h${level}`}
          id={titleId}
        >
          {title}
        </svelte:element>
      {/if}
      {#if actions}
        <div class="actions">{@render actions()}</div>
      {/if}
    </header>
  {/if}
  {@render children()}
</section>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    box-shadow: var(--shadow-sm);
    border-radius: var(--radius-lg);
    background: var(--color-surface);
    padding: var(--space-6);
    min-width: 0;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-2);
  }
  header :global(h2),
  header :global(h3) {
    margin: 0;
    font-size: var(--text-h3);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  @media (max-width: 768px) {
    .panel {
      padding: var(--space-4);
    }
  }
</style>
