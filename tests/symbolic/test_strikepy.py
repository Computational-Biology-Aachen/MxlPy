import sympy

from mxlpy.symbolic.strikepy import StrikepyModel, strike_goldd


def sir() -> StrikepyModel:
    # variables
    s, i, r = sympy.symbols("s i r")
    # parameters
    beta, gamma, n = sympy.symbols("beta gamma n")

    return StrikepyModel(
        states=[s, i, r],
        pars=[beta, gamma, n],
        outputs=[i, r],
        known_inputs=[],
        unknown_inputs=[],
        eqs=[
            -beta / n * i * s,
            beta / n * i * s - gamma * i,
            gamma * i,
        ],
    )


def karin_2016_1a() -> StrikepyModel:
    """Hormonal circuit model with integral feedback.

    Originally published in: Karin et al, Mol Syst Biol 2016
    Corresponds to the model in Fig. 1A of: Villaverde & Banga
    arXiv:1701.02562
    """
    # 2 states
    x1 = sympy.Symbol("x1")
    x2 = sympy.Symbol("x2")
    x10 = sympy.Symbol("x10")

    # 1 known input
    uu = sympy.Symbol("uu")
    u0 = sympy.Symbol("u0")

    # 2 unknown parameters
    p1 = sympy.Symbol("p1")
    p2 = sympy.Symbol("p2")

    return StrikepyModel(
        states=[x1, x2],
        outputs=[x1],
        known_inputs=[uu],
        pars=[p1, p2],
        eqs=[
            u0 + uu - p2 * x1 - p1 * x2,  # type: ignore
            x1 - x10,  # type: ignore
        ],
    )


def bolie_1961() -> StrikepyModel:
    """

    Model from J.W. Bolie. "Coefficients of normal blood glucose regulation".
    J. Appl. Physiol., 16(5):783-788, 1961.
    """
    # 2 states
    q1 = sympy.Symbol("q1")
    q2 = sympy.Symbol("q2")

    # 1 output
    Vp = sympy.Symbol("Vp")

    # 1 known input
    delta = sympy.Symbol("delta")

    # 5 unknown parameters
    p1 = sympy.Symbol("p1")
    p2 = sympy.Symbol("p2")
    p3 = sympy.Symbol("p3")
    p4 = sympy.Symbol("p4")

    return StrikepyModel(
        states=[q1, q2],
        outputs=[
            [q1 / Vp],  # type: ignore
        ],
        known_inputs=[delta],
        pars=[p1, p2, p3, p4, Vp],
        eqs=[
            p1 * q1 - p2 * q2 + delta,  # type: ignore
            p3 * q2 + p4 * q1,  # type: ignore
        ],
    )


def los(symbols: str) -> list[sympy.Symbol]:
    return list(sympy.symbols(symbols, seq=True))


def test_bolie_1961() -> None:
    res = strike_goldd(bolie_1961())
    assert res.rank == 5
    assert res.input_obs == []
    assert res.input_unobs == []
    assert res.par_ident == []
    assert res.par_unident == los("p1 p2 p3 p4 Vp")
    assert res.state_obs == []
    assert res.state_unobs == los("q1 q2")


def test_check_identifiability() -> None:
    # FIXME: implement this
    assert True


def test_karin_2016_1a() -> None:
    res = strike_goldd(karin_2016_1a())
    assert res.rank == 4
    assert res.input_obs == []
    assert res.input_unobs == []
    assert res.par_ident == los("p1 p2")
    assert res.par_unident == []
    assert res.state_obs == los("x1 x2")
    assert res.state_unobs == []


def test_result_all_inputs_observable() -> None:
    # FIXME: implement this
    assert True


def test_result_summary() -> None:
    # FIXME: implement this
    assert True


def test_sir() -> None:
    res = strike_goldd(sir())
    assert res.rank == 5
    assert res.input_obs == []
    assert res.input_unobs == []
    assert res.par_ident == los("gamma")
    assert res.par_unident == los("beta n")
    assert res.state_obs == los("s")
    assert res.state_unobs == []


def test_strike_goldd() -> None:
    # FIXME: implement this
    assert True
