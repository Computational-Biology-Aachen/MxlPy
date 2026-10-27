# API surface after #189

Function arguments on PR #189 (`feat/builder-fit-scan`), and whether each one makes sense per model builder.
Same layout as [api-surface.md](api-surface.md), which shows `main` before the PR.
Cells are taken from each routine's per-builder `@overload`, so ⚪ also marks an argument that a builder's overload drops (e.g. `y0` for `SteadyStateModelBuilder`).
Every cell holds three symbols, one per builder, always in this order:

1. `KineticModelBuilder`
2. `OdeModelBuilder`
3. `SteadyStateModelBuilder`

| symbol | meaning                                                           |
| ------ | ----------------------------------------------------------------- |
| 🟢      | applicable                                                        |
| 🟡      | might work, only potentially useful                               |
| 🔴      | doesn't work                                                      |
| ⚪      | not in this function's signature, and not needed there            |
| ➖      | routine doesn't apply to this builder at all (🔴 on its model row) |

**Bold** rows are the `runner=` argument, which replaces `integrator`, `tolerance`, `rel_norm`, `oscillation_detector` and `worker`.
It takes a `KineticRunner`, `OdeRunner` or `EvaluationRunner` respectively (`mxlpy.runners`), and defaults to the builder's own runner.

## scan

| args                   | `scan.steady_state` | `scan.time_course` | `scan.protocol` | `scan.protocol_time_course` |
| ---------------------- | ------------------- | ------------------ | --------------- | --------------------------- |
| `model`                | 🟢🟢🟢                 | 🟢🟢🔴                | 🟢🟢🔴             | 🟢🟢🔴                         |
| `to_scan`              | 🟢🟢🟢                 | 🟢🟢➖                | 🟢🟢➖             | 🟢🟢➖                         |
| `time_points`          | ⚪⚪⚪                 | 🟢🟢➖                | ⚪⚪➖             | 🟢🟢➖                         |
| `protocol`             | ⚪⚪⚪                 | ⚪⚪➖                | 🟢🟢➖             | 🟢🟢➖                         |
| `time_points_per_step` | ⚪⚪⚪                 | ⚪⚪➖                | 🟢🟢➖             | ⚪⚪➖                         |
| `y0`                   | 🟢🟢⚪                 | 🟢🟢➖                | 🟢🟢➖             | 🟢🟢➖                         |
| **`runner`**           | 🟢🟢🟢                 | 🟢🟢➖                | 🟢🟢➖             | 🟢🟢➖                         |
| `parallel`             | 🟢🟢🟡                 | 🟢🟢➖                | 🟢🟢➖             | 🟢🟢➖                         |
| `cache`                | 🟢🟢🟡                 | 🟢🟢➖                | 🟢🟢➖             | 🟢🟢➖                         |

## monte carlo (mc)

| args                   | `mc.steady_state` | `mc.time_course` | `mc.protocol` | `mc.protocol_time_course` |
| ---------------------- | ----------------- | ---------------- | ------------- | ------------------------- |
| `model`                | 🟢🟢🟢               | 🟢🟢🔴              | 🟢🟢🔴           | 🟢🟢🔴                       |
| `mc_to_scan`           | 🟢🟢🟢               | 🟢🟢➖              | 🟢🟢➖           | 🟢🟢➖                       |
| `time_points`          | ⚪⚪⚪               | 🟢🟢➖              | ⚪⚪➖           | 🟢🟢➖                       |
| `protocol`             | ⚪⚪⚪               | ⚪⚪➖              | 🟢🟢➖           | 🟢🟢➖                       |
| `time_points_per_step` | ⚪⚪⚪               | ⚪⚪➖              | 🟢🟢➖           | ⚪⚪➖                       |
| `y0`                   | 🟢🟢⚪               | 🟢🟢➖              | 🟢🟢➖           | 🟢🟢➖                       |
| **`runner`**           | 🟢🟢🟢               | 🟢🟢➖              | 🟢🟢➖           | 🟢🟢➖                       |
| `max_workers`          | 🟢🟢🟡               | 🟢🟢➖              | 🟢🟢➖           | 🟢🟢➖                       |
| `cache`                | 🟢🟢🟡               | 🟢🟢➖              | 🟢🟢➖           | 🟢🟢➖                       |

| args           | `mc.parameter_elasticities` | `mc.variable_elasticities` | `mc.response_coefficients` |
| -------------- | --------------------------- | -------------------------- | -------------------------- |
| `model`        | 🟢🔴🔴                         | 🟢🔴🔴                        | 🟢🟢🟢                        |
| `to_scan`      | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟢                        |
| `mc_to_scan`   | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟢                        |
| `variables`    | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢⚪                        |
| `time`         | 🟢➖➖                         | 🟢➖➖                        | ⚪⚪⚪                        |
| `normalized`   | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟢                        |
| `displacement` | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟢                        |
| **`runner`**   | ⚪➖➖                         | ⚪➖➖                        | 🟢🟢🟢                        |
| `max_workers`  | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟡                        |
| `cache`        | 🟢➖➖                         | 🟢➖➖                        | 🟢🟢🟡                        |
| `disable_tqdm` | ⚪➖➖                         | ⚪➖➖                        | 🟢🟢🟢                        |

| args          | `mc.scan_steady_state` |
| ------------- | ---------------------- |
| `model`       | 🟢🟢🟢                    |
| `to_scan`     | 🟢🟢🟢                    |
| `mc_to_scan`  | 🟢🟢🟢                    |
| `y0`          | 🟢🟢⚪                    |
| **`runner`**  | 🟢🟢🟢                    |
| `max_workers` | 🟢🟢🟡                    |
| `cache`       | 🟢🟢🟡                    |

## metabolic control analysis (mca)

| args           | `mca.parameter_elasticities` | `mca.variable_elasticities` | `mca.response_coefficients` |
| -------------- | ---------------------------- | --------------------------- | --------------------------- |
| `model`        | 🟢🔴🔴                          | 🟢🔴🔴                         | 🟢🟢🟢                         |
| `to_scan`      | 🟢➖➖                          | 🟢➖➖                         | 🟢🟢🟢                         |
| `variables`    | 🟢➖➖                          | 🟢➖➖                         | 🟢🟢⚪                         |
| `time`         | 🟢➖➖                          | 🟢➖➖                         | ⚪⚪⚪                         |
| `normalized`   | 🟢➖➖                          | 🟢➖➖                         | 🟢🟢🟢                         |
| `displacement` | 🟢➖➖                          | 🟢➖➖                         | 🟢🟢🟢                         |
| **`runner`**   | ⚪➖➖                          | ⚪➖➖                         | 🟢🟢🟢                         |
| `parallel`     | ⚪➖➖                          | ⚪➖➖                         | 🟢🟢🟡                         |
| `max_workers`  | ⚪➖➖                          | ⚪➖➖                         | 🟢🟢🟡                         |
| `disable_tqdm` | ⚪➖➖                          | ⚪➖➖                         | 🟢🟢🟢                         |

| args           | `mca.steady_state_stability` | `mca.rate_characteristics` |
| -------------- | ---------------------------- | -------------------------- |
| `model`        | 🟢🔴🔴                          | 🟢🔴🔴                        |
| `steady_state` | 🟢➖➖                          | ⚪➖➖                        |
| `variable`     | ⚪➖➖                          | 🟢➖➖                        |
| `threshold`    | 🟢➖➖                          | ⚪➖➖                        |
| `min_factor`   | ⚪➖➖                          | 🟢➖➖                        |
| `max_factor`   | ⚪➖➖                          | 🟢➖➖                        |
| `n_points`     | ⚪➖➖                          | 🟢➖➖                        |
| **`runner`**   | ⚪➖➖                          | 🟢➖➖                        |
| `parallel`     | ⚪➖➖                          | 🟢➖➖                        |

## fit

| args             | `fit.steady_state` | `fit.time_course` | `fit.protocol_time_course` |
| ---------------- | ------------------ | ----------------- | -------------------------- |
| `model`          | 🟢🟢🟢                | 🟢🟢🔴               | 🟢🟢🔴                        |
| `p0`             | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `data`           | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `minimizer`      | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `residual_fn`    | 🟢🟢🟡                | 🟢🟢➖               | 🟢🟢➖                        |
| `loss_fn`        | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `bounds`         | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `standard_scale` | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `protocol`       | ⚪⚪⚪                | ⚪⚪➖               | 🟢🟢➖                        |
| `y0`             | 🟢🟢⚪                | 🟢🟢➖               | 🟢🟢➖                        |
| **`runner`**     | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |
| `as_deepcopy`    | 🟢🟢🟢                | 🟢🟢➖               | 🟢🟢➖                        |

| args          | `fit.group_steady_state` | `fit.group_time_course` | `fit.group_protocol_time_course` |
| ------------- | ------------------------ | ----------------------- | -------------------------------- |
| `model`       | 🟢🟢🟢                      | 🟢🟢🔴                     | 🟢🟢🔴                              |
| `p0`          | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `data`        | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `minimizer`   | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `residual_fn` | 🟢🟢🟡                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `loss_fn`     | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `bounds`      | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `protocol`    | ⚪⚪⚪                      | ⚪⚪➖                     | 🟢🟢➖                              |
| `y0`          | 🟢🟢⚪                      | 🟢🟢➖                     | 🟢🟢➖                              |
| **`runner`**  | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `timeout`     | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |
| `as_deepcopy` | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              |

| args          | `fit.ensemble_steady_state` | `fit.ensemble_time_course` | `fit.ensemble_protocol_time_course` |
| ------------- | --------------------------- | -------------------------- | ----------------------------------- |
| `ensemble`    | 🟢🟢🟢                         | 🟢🟢🔴                        | 🟢🟢🔴                                 |
| `p0`          | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `data`        | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `minimizer`   | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `residual_fn` | 🟢🟢🟡                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `loss_fn`     | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `bounds`      | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `protocol`    | ⚪⚪⚪                         | ⚪⚪➖                        | 🟢🟢➖                                 |
| `y0`          | 🟢🟢⚪                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| **`runner`**  | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `timeout`     | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |
| `as_deepcopy` | 🟢🟢🟢                         | 🟢🟢➖                        | 🟢🟢➖                                 |

`to_fit` is a `Sequence[FitSettings]` for the first three and a `Sequence[MixedSettings]` for `joint_mixed`.
`to_fit[i].*` rows are their fields; ⚪ there means the field exists but the routine ignores it.
Both are typed `FitSettings[Any]`, so a `SteadyStateModelBuilder` in a joint time course type-checks, but fails at runtime: `EvaluationRunner` has no `time_course`.
For a `SteadyStateModelBuilder` entry, `y0` is accepted but silently ignored.

| args                    | `fit.joint_steady_state` | `fit.joint_time_course` | `fit.joint_protocol_time_course` | `fit.joint_mixed` |
| ----------------------- | ------------------------ | ----------------------- | -------------------------------- | ----------------- |
| `to_fit`                | 🟢🟢🟢                      | 🟢🟢🔴                     | 🟢🟢🔴                              | 🟢🟢🟢               |
| `p0`                    | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `minimizer`             | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `loss_fn`               | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `bounds`                | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `standard_scale`        | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `y0`                    | 🟢🟢🔴                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🔴               |
| **`runner`**            | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `max_workers`           | 🟢🟢🟡                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟡               |
| `as_deepcopy`           | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `to_fit[i].model`       | 🟢🟢🟢                      | 🟢🟢🔴                     | 🟢🟢🔴                              | 🟢🟢🟢               |
| `to_fit[i].data`        | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `to_fit[i].y0`          | 🟢🟢🔴                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🔴               |
| **`to_fit[i].runner`**  | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `to_fit[i].loss_fn`     | 🟢🟢🟢                      | 🟢🟢➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `to_fit[i].protocol`    | ⚪⚪⚪                      | ⚪⚪➖                     | 🟢🟢➖                              | 🟢🟢🟢               |
| `to_fit[i].residual_fn` | ⚪⚪⚪                      | ⚪⚪➖                     | ⚪⚪➖                              | 🟢🟢🟡               |

The carousel routines live in `mxlpy.carousel`, not `mxlpy.fit`.

| args          | `carousel.carousel_steady_state` | `carousel.carousel_time_course` | `carousel.carousel_protocol_time_course` |
| ------------- | -------------------------------- | ------------------------------- | ---------------------------------------- |
| `carousel`    | 🟢🔴🔴                              | 🟢🔴🔴                             | 🟢🔴🔴                                      |
| `p0`          | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `data`        | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `minimizer`   | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `residual_fn` | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `loss_fn`     | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `bounds`      | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `protocol`    | ⚪➖➖                              | ⚪➖➖                             | 🟢➖➖                                      |
| `y0`          | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| **`runner`**  | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `timeout`     | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
| `as_deepcopy` | 🟢➖➖                              | 🟢➖➖                             | 🟢➖➖                                      |
