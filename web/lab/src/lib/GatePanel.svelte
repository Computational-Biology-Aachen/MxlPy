<!--
  @component
  The review gate of a step: its proposal, the decision so far, and a form
  built from the step's options to override it.
-->
<script lang="ts">
  import Button from "@computational-biology-aachen/design/Button.svelte";
  import { api, formatValue, type StepView } from "./api.js";
  import { run } from "./run.svelte.js";

  let { view }: { view: StepView } = $props();

  let result = $derived(view.result!);
  let options = $derived(Object.entries(result.options));
  let decided = $derived(view.decision?.values ?? null);

  // The form starts from the decision, or else the proposal
  let values = $state<Record<string, unknown>>({});
  let mappings = $state<Record<string, string>>({});
  let mappingErrors = $state<Record<string, string>>({});
  let busy = $state(false);

  $effect(() => {
    const start = structuredClone(
      $state.snapshot(decided ?? result.proposal),
    ) as Record<string, unknown>;
    values = start;
    mappings = Object.fromEntries(
      options
        .filter(([, o]) => o.kind === "mapping")
        .map(([k]) => [k, JSON.stringify(start[k] ?? {}, null, 1)]),
    );
    mappingErrors = {};
  });

  function toggle(key: string, choice: unknown, on: boolean): void {
    const current = (values[key] as unknown[]) ?? [];
    values[key] = on
      ? [...current, choice]
      : current.filter((c) => c !== choice);
  }

  function parsed(): Record<string, unknown> | null {
    const out: Record<string, unknown> = { ...values };
    let ok = true;
    for (const [key, text] of Object.entries(mappings)) {
      try {
        out[key] = JSON.parse(text);
        mappingErrors[key] = "";
      } catch (e) {
        mappingErrors[key] = e instanceof Error ? e.message : "Invalid JSON";
        ok = false;
      }
    }
    return ok ? out : null;
  }

  async function submit(override: boolean): Promise<void> {
    const out = override ? parsed() : {};
    if (out === null) return;
    busy = true;
    await run.act(() => api.decide(view.step, out));
    busy = false;
  }

  async function reopen(): Promise<void> {
    busy = true;
    await run.act(() => api.reopen(view.step));
    busy = false;
  }
</script>

<section
  class="gate"
  class:open={view.decision === null}
  aria-labelledby="gate-title"
>
  <header>
    <h2 id="gate-title">Review gate</h2>
    {#if view.decision}
      <span class="badge">decided ({view.decision.source})</span>
    {:else}
      <span class="badge open">waiting for a decision</span>
    {/if}
  </header>

  {#if Object.keys(result.proposal).length > 0}
    <dl class="proposal">
      {#each Object.entries(result.proposal) as [key, value] (key)}
        <dt>Proposed {key}</dt>
        <dd>{formatValue(value)}</dd>
        {#if decided && JSON.stringify(decided[key]) !== JSON.stringify(value)}
          <dt>Decided {key}</dt>
          <dd class="changed">{formatValue(decided[key])}</dd>
        {/if}
      {/each}
    </dl>
  {:else}
    <p class="muted">Nothing to decide: continue to the next step.</p>
  {/if}

  {#if options.length > 0}
    <form
      onsubmit={(e) => {
        e.preventDefault();
        void submit(true);
      }}
    >
      {#each options as [key, option] (key)}
        <fieldset>
          <legend>{key}</legend>
          {#if option.description}
            <p class="muted">{option.description}</p>
          {/if}
          {#if option.kind === "subset"}
            <div class="choices">
              {#each option.choices as choice (String(choice))}
                <label>
                  <input
                    type="checkbox"
                    checked={((values[key] as unknown[]) ?? []).includes(
                      choice,
                    )}
                    onchange={(e) =>
                      toggle(key, choice, e.currentTarget.checked)}
                  />
                  {String(choice)}
                </label>
              {/each}
            </div>
          {:else if option.kind === "choice"}
            <select
              aria-label={key}
              value={String(values[key])}
              onchange={(e) => {
                values[key] = option.choices.find(
                  (c) => String(c) === e.currentTarget.value,
                );
              }}
            >
              {#each option.choices as choice (String(choice))}
                <option value={String(choice)}>{String(choice)}</option>
              {/each}
            </select>
          {:else if option.kind === "flag"}
            <label>
              <input
                type="checkbox"
                checked={Boolean(values[key])}
                onchange={(e) => (values[key] = e.currentTarget.checked)}
              />
              {key}
            </label>
          {:else}
            <textarea
              aria-label={key}
              rows="4"
              bind:value={mappings[key]}></textarea>
            {#if mappingErrors[key]}
              <p class="error">{mappingErrors[key]}</p>
            {/if}
          {/if}
        </fieldset>
      {/each}
      <div class="actions">
        <Button
          variant="primary"
          disabled={busy}
          onclick={() => void submit(false)}
        >
          Accept proposal
        </Button>
        <Button
          variant="secondary"
          type="submit"
          disabled={busy}
        >
          Decide as edited
        </Button>
        {#if view.decision}
          <Button
            variant="secondary"
            disabled={busy}
            onclick={() => void reopen()}
          >
            Reopen
          </Button>
        {/if}
      </div>
    </form>
  {:else}
    <div class="actions">
      <Button
        variant="primary"
        disabled={busy || view.decision !== null}
        onclick={() => void submit(false)}
      >
        Continue
      </Button>
      {#if view.decision}
        <Button
          variant="secondary"
          disabled={busy}
          onclick={() => void reopen()}
        >
          Reopen
        </Button>
      {/if}
    </div>
  {/if}
</section>

<style>
  .gate {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    padding: var(--space-4);
  }
  .gate.open {
    box-shadow: inset 4px 0 0 var(--color-accent);
    border-color: var(--color-accent);
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  h2 {
    margin: 0;
    font-size: var(--text-h3);
  }
  .badge {
    border-radius: var(--radius-full);
    background: var(--slate-50);
    padding: 0 var(--space-2);
    font-size: var(--text-sm);
  }
  .badge.open {
    background: var(--color-accent);
    color: var(--color-text);
  }
  .proposal {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: var(--space-1) var(--space-3);
  }
  dt {
    color: var(--color-text-muted);
  }
  dd {
    margin: 0;
    font-family: var(--font-mono);
    overflow-wrap: anywhere;
  }
  dd.changed {
    color: var(--color-primary);
    font-weight: 600;
  }
  fieldset {
    margin: 0 0 var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
  }
  legend {
    padding: 0 var(--space-1);
    font-weight: 600;
  }
  .choices {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
  }
  label {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  select,
  textarea {
    box-sizing: border-box;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    padding: var(--space-1) var(--space-2);
    max-width: 100%;
    color: inherit;
    font: inherit;
  }
  textarea {
    width: 100%;
    font-family: var(--font-mono);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .muted {
    margin: 0 0 var(--space-2);
    color: var(--color-text-muted);
  }
  .error {
    color: var(--color-danger);
  }
</style>
