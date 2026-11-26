<!--
  @component
  The review gate of a step: its proposal, the decision so far, and a form
  built from the step's options to override it.
-->
<script lang="ts">
  import Button from "@computational-biology-aachen/design/Button.svelte";
  import InfoBox from "@computational-biology-aachen/design/InfoBox.svelte";
  import InputCheckbox from "@computational-biology-aachen/design/InputCheckbox.svelte";
  import InputChoice from "@computational-biology-aachen/design/InputChoice.svelte";
  import { api, formatValue, type StepView } from "./api.js";
  import Panel from "./Panel.svelte";
  import { run } from "./run.svelte.js";
  import StatusBadge from "./StatusBadge.svelte";

  let { view }: { view: StepView } = $props();

  let result = $derived(view.result!);
  let options = $derived(Object.entries(result.options));
  let decided = $derived(view.decision?.values ?? null);

  // The form starts from the decision, or else the proposal
  let values = $state<Record<string, unknown>>({});
  let mappings = $state<Record<string, string>>({});
  let mappingErrors = $state<Record<string, string>>({});
  let busy = $state<"accept" | "edit" | "reopen" | null>(null);

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

  const same = (a: unknown, b: unknown): boolean =>
    JSON.stringify(a) === JSON.stringify(b);

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
    busy = override ? "edit" : "accept";
    await run.act(() => api.decide(view.step, out));
    busy = null;
  }

  async function reopen(): Promise<void> {
    busy = "reopen";
    await run.act(() => api.reopen(view.step));
    busy = null;
  }
</script>

<Panel
  title="Review gate"
  titleId="gate-title"
>
  {#snippet actions()}
    {#if view.decision}
      <StatusBadge tone="neutral">decided ({view.decision.source})</StatusBadge>
    {/if}
  {/snippet}

  {#if view.decision === null}
    <InfoBox
      header="Waiting for a decision"
      variant="warning"
    >
      {#if options.length > 0}
        Accept the proposal, or edit the values below and decide as edited.
      {:else}
        Continue when the results look right.
      {/if}
    </InfoBox>
  {/if}

  {#if Object.keys(result.proposal).length > 0}
    <h3 class="caption">{decided ? "Proposal and decision" : "Proposal"}</h3>
    <dl class="proposal">
      {#each Object.entries(result.proposal) as [key, value] (key)}
        <dt>{key}</dt>
        <dd>
          {#if decided && !same(decided[key], value)}
            <span class="old">{formatValue(value)}</span>
            <span aria-hidden="true">→</span>
            <span class="changed">{formatValue(decided[key])}</span>
          {:else}
            {formatValue(value)}
          {/if}
        </dd>
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
            <div class="chips">
              {#each option.choices as choice (String(choice))}
                <label class="chip">
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
            <div class="control">
              <InputChoice
                id={`gate-${key}`}
                label={key}
                bind:value={
                  () => String(values[key]),
                  (v) =>
                    (values[key] = option.choices.find((c) => String(c) === v))
                }
              >
                {#each option.choices as choice (String(choice))}
                  <option value={String(choice)}>{String(choice)}</option>
                {/each}
              </InputChoice>
            </div>
          {:else if option.kind === "flag"}
            <div class="control">
              <InputCheckbox
                id={`gate-${key}`}
                label={key}
                bind:checked={
                  () => Boolean(values[key]), (v) => (values[key] = v)
                }
              />
            </div>
          {:else}
            <textarea
              aria-label={key}
              aria-invalid={mappingErrors[key] ? "true" : undefined}
              rows="4"
              bind:value={mappings[key]}></textarea>
            {#if mappingErrors[key]}
              <p class="error">{mappingErrors[key]}</p>
            {/if}
          {/if}
        </fieldset>
      {/each}
      <div class="actions">
        {#if view.decision}
          <Button
            variant="secondary"
            disabled={busy !== null}
            loading={busy === "reopen"}
            onclick={() => void reopen()}
          >
            Reopen
          </Button>
        {/if}
        <Button
          variant="secondary"
          type="submit"
          disabled={busy !== null}
          loading={busy === "edit"}
        >
          Decide as edited
        </Button>
        <Button
          variant="primary"
          disabled={busy !== null}
          loading={busy === "accept"}
          onclick={() => void submit(false)}
        >
          Accept proposal
        </Button>
      </div>
    </form>
  {:else}
    <div class="actions">
      {#if view.decision}
        <Button
          variant="secondary"
          disabled={busy !== null}
          loading={busy === "reopen"}
          onclick={() => void reopen()}
        >
          Reopen
        </Button>
      {/if}
      <Button
        variant="primary"
        disabled={busy !== null || view.decision !== null}
        loading={busy === "accept"}
        onclick={() => void submit(false)}
      >
        Continue
      </Button>
    </div>
  {/if}
</Panel>

<style>
  .caption {
    margin: 0;
    color: var(--color-text-muted);
    font-weight: var(--weight-medium);
    font-size: var(--text-callout);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .proposal {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: var(--space-2) var(--space-4);
    margin: 0;
    font-size: var(--text-sm);
  }
  dt {
    color: var(--color-text-muted);
  }
  dd {
    margin: 0;
    font-family: var(--font-mono);
    overflow-wrap: anywhere;
  }
  .old {
    color: var(--color-text-muted);
    text-decoration: line-through;
  }
  .changed {
    color: var(--color-primary);
    font-weight: var(--weight-medium);
  }
  form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  fieldset {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin: 0;
    border: none;
    border-top: var(--border);
    padding: var(--space-4) 0 0;
    min-width: 0;
  }
  legend {
    float: left;
    margin-bottom: var(--space-1);
    padding: 0;
    font-weight: var(--weight-medium);
  }
  legend + * {
    clear: left;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    transition:
      background-color var(--transition),
      border-color var(--transition);
    cursor: pointer;
    border: var(--border);
    border-radius: var(--radius-full);
    padding: var(--space-1) var(--space-3);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
  }
  .chip:has(input:checked) {
    border-color: var(--color-primary);
    background: rgb(from var(--color-primary) r g b / 8%);
    color: var(--color-primary);
  }
  .chip:has(input:focus-visible) {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
  }
  .chip input {
    margin: 0;
    accent-color: var(--color-primary);
  }
  .control {
    max-width: 24rem;
  }
  textarea {
    box-sizing: border-box;
    border: var(--border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    padding: var(--space-2) var(--space-3);
    width: 100%;
    color: inherit;
    font-size: var(--text-body);
    font-family: var(--font-mono);
  }
  textarea[aria-invalid="true"] {
    border-color: var(--color-danger);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: end;
    gap: var(--space-2);
  }
  .actions :global(:is(button, a)) {
    width: auto;
  }
  .muted {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-sm);
  }
  .error {
    margin: 0;
    color: var(--color-danger);
    font-size: var(--text-callout);
  }
  @media (prefers-reduced-motion: reduce) {
    .chip {
      transition: none;
    }
  }
</style>
