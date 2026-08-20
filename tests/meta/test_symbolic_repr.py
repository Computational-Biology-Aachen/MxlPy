from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from mxlpy import OdeModelBuilder
from mxlpy.meta._via_sym_repr import model_to_symbolic_repr
from mxlpy.surrogates.abstract_derivative import AbstractDerivativeSurrogate

if TYPE_CHECKING:
    import pandas as pd


def test_err_eval() -> None:
    # FIXME: implement this
    assert True


def test_model_to_symbolic_repr() -> None:
    # FIXME: implement this
    assert True


@dataclass(kw_only=True)
class _MockDerivativeSurrogate(AbstractDerivativeSurrogate):
    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],  # noqa: ARG002, for API compatibility
    ) -> dict[str, float]:
        return dict.fromkeys(self.targets, 0.0)


def _neg(x: float) -> float:
    # A named, module-level function (not a lambda): `fn_to_sympy_expr`
    # recovers a rate function's source via `inspect`, which needs the
    # function's own statement to be independently parseable — a lambda
    # split across a chained multi-line `.add_diff_eq(...)` call doesn't
    # satisfy that, a plain `def` always does.
    return -x


def _make_ode_model_with_surrogate() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_diff_eq("x", fn=_neg, args=["x"], initial_value=1.0)
        .add_surrogate("corr", _MockDerivativeSurrogate(args=[], targets=["x"]))
    )


def test_ode_builder_with_surrogate_raises_by_default() -> None:
    """`model_to_symbolic_repr` has no supported conversion for a direct-derivative surrogate yet.

    A surrogate silently dropped here would make the returned
    `SymbolicRepr` describe different dynamics than the actual model
    computes (a missing correction term), so this must raise rather than
    silently omit it — see `_ode_builder_to_symbolic_repr`'s comment.
    """
    model = _make_ode_model_with_surrogate()
    with pytest.raises(ValueError, match="corr"):
        model_to_symbolic_repr(model, custom_fns={})


def test_ode_builder_with_surrogate_only_warns_when_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _make_ode_model_with_surrogate()
    with caplog.at_level("WARNING"):
        sym = model_to_symbolic_repr(model, only_warn=True, custom_fns={})
    assert "x" in sym.variables
    assert "corr" in caplog.text
