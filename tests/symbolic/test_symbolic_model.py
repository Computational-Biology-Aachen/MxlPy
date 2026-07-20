import sympy

from mxlpy import KineticModelBuilder
from mxlpy.surrogates import qss
from mxlpy.symbolic import to_symbolic_model
from tests.models import m_2v_2p_2d_2r, m_derived_stoichiometry


def test_to_symbolic_model() -> None:
    model = m_2v_2p_2d_2r()
    sym = to_symbolic_model(model)

    assert set(sym.variables) == set(model.get_variable_names())
    assert set(sym.parameters) == set(model.get_parameter_names())
    assert len(sym.eqs) == len(sym.variables)

    # Every diff eq must be fully expanded: only variable/parameter symbols
    # may remain, no leftover derived/reaction/rate names.
    allowed = set(sym.variables.values()) | set(sym.parameters.values())
    for eq in sym.eqs:
        assert eq.free_symbols <= allowed


def test_symbolicmodel_jacobian() -> None:
    jac = to_symbolic_model(m_2v_2p_2d_2r()).jacobian()
    assert jac.shape == (2, 2)


def qss_flux(v1: float, k1: float) -> tuple[float]:
    return (k1 * v1,)


def test_to_symbolic_model_qss_surrogate_feeding_stoichiometry() -> None:
    """A QSS surrogate output used as a stoichiometry factor used to crash
    `to_symbolic_model` with a `KeyError`, since surrogate outputs were never
    added to the internal reaction-expression map.
    """
    model = (
        KineticModelBuilder()
        .add_variable("v1", 1.0)
        .add_parameter("k1", 0.1)
        .add_surrogate(
            "srg1",
            qss.Surrogate(
                model=qss_flux,
                args=["v1", "k1"],
                outputs=["v1_efflux"],
                stoichiometries={"v1_efflux": {"v1": -1.0}},
            ),
        )
    )

    sym = to_symbolic_model(model)
    assert len(sym.eqs) == 1

    v1, k1 = sympy.symbols("v1 k1")
    assert sympy.simplify(sym.eqs[0] - (-k1 * v1)) == 0
    # Fully expanded: no leftover surrogate/rate symbol.
    assert sym.eqs[0].free_symbols <= {v1, k1}

    jac = sym.jacobian()
    assert jac.shape == (1, 1)
    assert sympy.simplify(jac[0, 0] - (-k1)) == 0


def test_to_symbolic_model_dynamic_stoichiometry() -> None:
    """Dynamic (`Derived`) stoichiometry used to be built via a malformed
    `fn_to_sympy_expr` call (missing keyword args, `list * Expr`), a path
    never exercised by any test.
    """
    sym = to_symbolic_model(m_derived_stoichiometry())
    assert len(sym.eqs) == 1
    assert sympy.simplify(sym.eqs[0] - 1) == 0
