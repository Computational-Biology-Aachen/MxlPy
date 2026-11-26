<!--
  @component
  A table of mxlpy.lab.views.
-->
<script lang="ts">
  import Table from "@computational-biology-aachen/design/Table.svelte";
  import Tbody from "@computational-biology-aachen/design/Tbody.svelte";
  import Td from "@computational-biology-aachen/design/Td.svelte";
  import Th from "@computational-biology-aachen/design/Th.svelte";
  import Thead from "@computational-biology-aachen/design/Thead.svelte";
  import Tr from "@computational-biology-aachen/design/Tr.svelte";
  import { formatValue, type TableSpec } from "./api.js";
  import Panel from "./Panel.svelte";

  let { spec }: { spec: TableSpec } = $props();
</script>

<Panel
  title={spec.title}
  titleId={`table-${spec.id}`}
  level={3}
>
  <div class="table">
    <Table label={spec.title}>
      <Thead>
        <Tr>
          {#each spec.columns as column, i (i)}
            <Th>{column}</Th>
          {/each}
        </Tr>
      </Thead>
      <Tbody>
        {#each spec.rows as row, i (i)}
          <Tr>
            {#each row as cell, j (j)}
              <Td>
                <span class:number={typeof cell === "number"}>
                  {formatValue(cell)}
                </span>
              </Td>
            {/each}
          </Tr>
        {/each}
      </Tbody>
    </Table>
  </div>
</Panel>

<style>
  .table :global(.table-wrap) {
    margin-bottom: 0;
  }
  .table :global(th) {
    background: var(--color-bg);
    color: var(--color-text-muted);
    font-weight: var(--weight-medium);
    font-size: var(--text-callout);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .number {
    font-size: var(--text-body);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
</style>
