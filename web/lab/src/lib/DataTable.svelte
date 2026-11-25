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

  let { spec }: { spec: TableSpec } = $props();
</script>

<section>
  <h3>{spec.title}</h3>
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
            <Td>{formatValue(cell)}</Td>
          {/each}
        </Tr>
      {/each}
    </Tbody>
  </Table>
</section>

<style>
  section {
    min-width: 0;
  }
  h3 {
    margin: 0 0 var(--space-2);
    font-size: var(--text-body);
  }
</style>
