<!--
  @component
  The lab's page layout: the step rail on the left, the page on the right.
-->
<script lang="ts">
  import type { Snippet } from "svelte";

  let { rail, children }: { rail: Snippet; children: Snippet } = $props();
</script>

<div class="layout">
  <aside>{@render rail()}</aside>
  <div class="content">{@render children()}</div>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: minmax(15rem, 19rem) minmax(0, 1fr);
    align-items: start;
    gap: var(--space-6);
    width: 100%;
  }
  aside {
    position: sticky;
    top: calc(var(--nav-height) + var(--space-4));
    max-height: calc(100dvh - var(--nav-height) - var(--space-8));
    overflow-y: auto;
  }
  .content {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  @media (max-width: 768px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    aside {
      position: static;
      max-height: none;
    }
  }
</style>
