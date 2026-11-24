# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Model selection with information criteria and structure search.

Compare candidate model structures by the log-likelihood of their best
multistart fit and their number of estimated parameters.

- `ModelSpace`: a base model and switchable features (`Switch`,
  `free_parameter`), whose combinations are the candidates
- `run`: forward, backward, adaptive (FAMoS-like) or brute-force search
  over a model space
- `compare`: fit and rank explicitly given model variants
- `Selection`: ranked table, selected model, search path and
  likelihood-ratio tests between candidates

Condition-specific parameters

- `condition_specific`: which parameters have to differ between conditions
  (mutants, treatments, ...), from an L1-regularised joint fit of all
  conditions and unpenalised refits of the structures on its path

The criteria are also available on their own, e.g. to compare other fits:
`aic`, `aicc`, `bic`, `model_weights` and the likelihood-ratio test `lrt`.

Examples
--------
    >>> from mxlpy import fit, select
    >>> space = select.ModelSpace(
    ...     base=model,
    ...     switches={"feedback": select.free_parameter("k_fb", fixed_value=0.0)},
    ... )
    >>> selection = select.run(space, data, {"k1": 1.0, "k_fb": 0.1},
    ...                        loss_fn=fit.GaussianNll(sigma=0.05))
    >>> selection.table

"""

from __future__ import annotations

from mxlpy.select._conditions import (
    ConditionFit,
    ConditionSpecific,
    Regularisation,
    Structure,
    condition_specific,
)
from mxlpy.select._criteria import (
    Criterion,
    InformationCriterion,
    LikelihoodRatio,
    aic,
    aicc,
    bic,
    information_criterion,
    lrt,
    model_weights,
)
from mxlpy.select._select import CandidateFit, Method, Selection, compare, run
from mxlpy.select._space import BASE, ModelSpace, Move, Switch, free_parameter

__all__ = [
    "BASE",
    "CandidateFit",
    "ConditionFit",
    "ConditionSpecific",
    "Criterion",
    "InformationCriterion",
    "LikelihoodRatio",
    "Method",
    "ModelSpace",
    "Move",
    "Regularisation",
    "Selection",
    "Structure",
    "Switch",
    "aic",
    "aicc",
    "bic",
    "compare",
    "condition_specific",
    "free_parameter",
    "information_criterion",
    "lrt",
    "model_weights",
    "run",
]
