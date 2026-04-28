from mxlpy import Derived, InitialAssignment, Model, meta
from mxlpy.surrogates import qss


def _p_init(v1: float) -> float:
    return v1 + 1.0


def constant(x: float) -> float:
    return x


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def _qss_two(v1: float, p1: float) -> tuple[float, float]:
    return v1 * p1, v1 + p1


def model_nasty() -> Model:
    return (
        Model()
        .add_variable("v-1", initial_value=1.0)
        .add_variable("v-2", initial_value=2.0)
        .add_variable("v-3", initial_value=3.0)
        .add_variable("v-4", initial_value=4.0)
        .add_variable("v-5", initial_value=5.0)
        .add_variable(
            "v-ia6",
            initial_value=InitialAssignment(fn=_p_init, args=["p-ia1"]),
        )
        .add_parameter("p-1", value=1.0)
        .add_parameter("p-2", value=1.0)
        .add_parameter(
            "p-ia1",
            value=InitialAssignment(fn=_p_init, args=["p-2"]),
        )
        .add_derived(
            "d-1",
            fn=constant,
            args=["v-1"],
        )
        .add_derived(
            "d-2",
            fn=constant,
            args=["rxn-1"],
        )
        .add_reaction(
            "rxn-1",
            fn=mass_action_1s,
            args=["v-1", "d-1"],
            stoichiometry={"v-1": -1, "v-2": Derived(fn=constant, args=["p-2"])},
        )
        .add_surrogate(
            "qss1",
            qss.Surrogate(
                model=_qss_two,
                args=["v-1", "p-1"],
                outputs=["out-1", "out-2"],
                stoichiometries={"out-1": {"v-1": -1.0}},
            ),
        )
        .add_readout(
            "read-out-1",
            fn=constant,
            args=["out-2"],
        )
    )


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_nasty()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 6> model(double time, const std::array<double, 6>& variables) {",
        "    const auto [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = variables;",
        "    double p_minus_1 = 1.0;",
        "    double p_minus_2 = 1.0;",
        "    double d_minus_1 = v_minus_1;",
        "    double rxn_minus_1 = d_minus_1*v_minus_1;",
        "    double out_minus_1 = p_minus_1*v_minus_1;",
        "    double out_minus_2 = p_minus_1 + v_minus_1;",
        "    double d_minus_2 = rxn_minus_1;",
        "    double dv_minus_1dt = -1.0*out_minus_1 - 1.0*rxn_minus_1;",
        "    double dv_minus_2dt = p_minus_2*rxn_minus_1;",
        "    double dv_minus_3dt = 0;",
        "    double dv_minus_4dt = 0;",
        "    double dv_minus_5dt = 0;",
        "    double dv_minus_ia6dt = 0;",
        "    return {dv_minus_1dt, dv_minus_2dt, dv_minus_3dt, dv_minus_4dt, dv_minus_5dt, dv_minus_ia6dt};",
        "}",
        "",
        "std::array<double, 6> derived(double time, const std::array<double, 6>& variables) {",
        "    const auto [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = variables;",
        "    double p_minus_1 = 1.0;",
        "    double p_minus_2 = 1.0;",
        "    double d_minus_1 = v_minus_1;",
        "    double rxn_minus_1 = d_minus_1*v_minus_1;",
        "    double out_minus_1 = p_minus_1*v_minus_1;",
        "    double out_minus_2 = p_minus_1 + v_minus_1;",
        "    double d_minus_2 = rxn_minus_1;",
        "    double read_minus_out_minus_1 = out_minus_2;",
        "    return {d_minus_1, d_minus_2, rxn_minus_1, out_minus_1, out_minus_2, read_minus_out_minus_1};",
        "}",
        "",
        "std::pair<std::array<double, 6>, std::array<double, 2>> inits() {",
        "    double p_minus_2 = 1.0;",
        "    double v_minus_1 = 1.0;",
        "    double v_minus_2 = 2.0;",
        "    double v_minus_3 = 3.0;",
        "    double v_minus_4 = 4.0;",
        "    double v_minus_5 = 5.0;",
        "    double p_minus_ia1 = p_minus_2 + 1.0;",
        "    double v_minus_ia6 = p_minus_ia1 + 1.0;",
        "    return {{v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6}, {p_minus_2, p_minus_ia1}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_nasty()).full().split("\n") == [
        "function model(time, variables)",
        "    v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6 = variables",
        "    p_minus_1 = 1.0",
        "    p_minus_2 = 1.0",
        "    d_minus_1 = v_minus_1",
        "    rxn_minus_1 = d_minus_1 .* v_minus_1",
        "    out_minus_1 = p_minus_1 .* v_minus_1",
        "    out_minus_2 = p_minus_1 + v_minus_1",
        "    d_minus_2 = rxn_minus_1",
        "    dv_minus_1dt = -1.0 * out_minus_1 - 1.0 * rxn_minus_1",
        "    dv_minus_2dt = p_minus_2 .* rxn_minus_1",
        "    dv_minus_3dt = 0",
        "    dv_minus_4dt = 0",
        "    dv_minus_5dt = 0",
        "    dv_minus_ia6dt = 0",
        "    return [dv_minus_1dt, dv_minus_2dt, dv_minus_3dt, dv_minus_4dt, dv_minus_5dt, dv_minus_ia6dt]",
        "end",
        "",
        "function derived(time, variables)",
        "    v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6 = variables",
        "    p_minus_1 = 1.0",
        "    p_minus_2 = 1.0",
        "    d_minus_1 = v_minus_1",
        "    rxn_minus_1 = d_minus_1 .* v_minus_1",
        "    out_minus_1 = p_minus_1 .* v_minus_1",
        "    out_minus_2 = p_minus_1 + v_minus_1",
        "    d_minus_2 = rxn_minus_1",
        "    read_minus_out_minus_1 = out_minus_2",
        "    return [d_minus_1, d_minus_2, rxn_minus_1, out_minus_1, out_minus_2, read_minus_out_minus_1]",
        "end",
        "",
        "function inits()",
        "    p_minus_2 = 1.0",
        "    v_minus_1 = 1.0",
        "    v_minus_2 = 2.0",
        "    v_minus_3 = 3.0",
        "    v_minus_4 = 4.0",
        "    v_minus_5 = 5.0",
        "    p_minus_ia1 = p_minus_2 + 1.0",
        "    v_minus_ia6 = p_minus_ia1 + 1.0",
        "    return [[v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6], [p_minus_2, p_minus_ia1]]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_nasty()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = num2cell(variables){:};",
        "    p_minus_1 = 1.0;",
        "    p_minus_2 = 1.0;",
        "    d_minus_1 = v_minus_1;",
        "    rxn_minus_1 = d_minus_1.*v_minus_1;",
        "    out_minus_1 = p_minus_1.*v_minus_1;",
        "    out_minus_2 = p_minus_1 + v_minus_1;",
        "    d_minus_2 = rxn_minus_1;",
        "    dv_minus_1dt = -1.0*out_minus_1 - 1.0*rxn_minus_1;",
        "    dv_minus_2dt = p_minus_2.*rxn_minus_1;",
        "    dv_minus_3dt = 0;",
        "    dv_minus_4dt = 0;",
        "    dv_minus_5dt = 0;",
        "    dv_minus_ia6dt = 0;",
        "    dydt = [dv_minus_1dt, dv_minus_2dt, dv_minus_3dt, dv_minus_4dt, dv_minus_5dt, dv_minus_ia6dt]';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = num2cell(variables){:};",
        "    p_minus_1 = 1.0;",
        "    p_minus_2 = 1.0;",
        "    d_minus_1 = v_minus_1;",
        "    rxn_minus_1 = d_minus_1.*v_minus_1;",
        "    out_minus_1 = p_minus_1.*v_minus_1;",
        "    out_minus_2 = p_minus_1 + v_minus_1;",
        "    d_minus_2 = rxn_minus_1;",
        "    read_minus_out_minus_1 = out_minus_2;",
        "    out = [d_minus_1, d_minus_2, rxn_minus_1, out_minus_1, out_minus_2, read_minus_out_minus_1];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    p_minus_2 = 1.0;",
        "    v_minus_1 = 1.0;",
        "    v_minus_2 = 2.0;",
        "    v_minus_3 = 3.0;",
        "    v_minus_4 = 4.0;",
        "    v_minus_5 = 5.0;",
        "    p_minus_ia1 = p_minus_2 + 1.0;",
        "    v_minus_ia6 = p_minus_ia1 + 1.0;",
        "    vars = [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6]';",
        "    pars = [p_minus_2, p_minus_ia1]';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_nasty()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 6]) -> [f64; 6] {",
        "    let [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = *variables;",
        "    let p_minus_1: f64 = 1.0;",
        "    let p_minus_2: f64 = 1.0;",
        "    let d_minus_1: f64 = v_minus_1;",
        "    let rxn_minus_1: f64 = d_minus_1*v_minus_1;",
        "    let out_minus_1: f64 = p_minus_1*v_minus_1;",
        "    let out_minus_2: f64 = p_minus_1 + v_minus_1;",
        "    let d_minus_2: f64 = rxn_minus_1;",
        "    let dv_minus_1dt: f64 = -1.0*out_minus_1 - 1.0*rxn_minus_1;",
        "    let dv_minus_2dt: f64 = p_minus_2*rxn_minus_1;",
        "    let dv_minus_3dt: f64 = 0;",
        "    let dv_minus_4dt: f64 = 0;",
        "    let dv_minus_5dt: f64 = 0;",
        "    let dv_minus_ia6dt: f64 = 0;",
        "    return [dv_minus_1dt, dv_minus_2dt, dv_minus_3dt, dv_minus_4dt, dv_minus_5dt, dv_minus_ia6dt]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 6]) -> [f64; 6] {",
        "    let [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = *variables;",
        "    let p_minus_1: f64 = 1.0;",
        "    let p_minus_2: f64 = 1.0;",
        "    let d_minus_1: f64 = v_minus_1;",
        "    let rxn_minus_1: f64 = d_minus_1*v_minus_1;",
        "    let out_minus_1: f64 = p_minus_1*v_minus_1;",
        "    let out_minus_2: f64 = p_minus_1 + v_minus_1;",
        "    let d_minus_2: f64 = rxn_minus_1;",
        "    let read_minus_out_minus_1: f64 = out_minus_2;",
        "    return [d_minus_1, d_minus_2, rxn_minus_1, out_minus_1, out_minus_2, read_minus_out_minus_1]",
        "}",
        "",
        "fn inits() -> ([f64; 6], [f64; 2]) {",
        "    let p_minus_2: f64 = 1.0;",
        "    let v_minus_1: f64 = 1.0;",
        "    let v_minus_2: f64 = 2.0;",
        "    let v_minus_3: f64 = 3.0;",
        "    let v_minus_4: f64 = 4.0;",
        "    let v_minus_5: f64 = 5.0;",
        "    let p_minus_ia1: f64 = p_minus_2 + 1.0;",
        "    let v_minus_ia6: f64 = p_minus_ia1 + 1.0;",
        "    return ([v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6], [p_minus_2, p_minus_ia1])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_nasty()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = variables;",
        "    const p_minus_1: number = 1.0;",
        "    const p_minus_2: number = 1.0;",
        "    const d_minus_1: number = v_minus_1;",
        "    const rxn_minus_1: number = d_minus_1*v_minus_1;",
        "    const out_minus_1: number = p_minus_1*v_minus_1;",
        "    const out_minus_2: number = p_minus_1 + v_minus_1;",
        "    const d_minus_2: number = rxn_minus_1;",
        "    const dv_minus_1dt: number = -1.0*out_minus_1 - 1.0*rxn_minus_1;",
        "    const dv_minus_2dt: number = p_minus_2*rxn_minus_1;",
        "    const dv_minus_3dt: number = 0;",
        "    const dv_minus_4dt: number = 0;",
        "    const dv_minus_5dt: number = 0;",
        "    const dv_minus_ia6dt: number = 0;",
        "    return [dv_minus_1dt, dv_minus_2dt, dv_minus_3dt, dv_minus_4dt, dv_minus_5dt, dv_minus_ia6dt];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6] = variables;",
        "    const p_minus_1: number = 1.0;",
        "    const p_minus_2: number = 1.0;",
        "    const d_minus_1: number = v_minus_1;",
        "    const rxn_minus_1: number = d_minus_1*v_minus_1;",
        "    const out_minus_1: number = p_minus_1*v_minus_1;",
        "    const out_minus_2: number = p_minus_1 + v_minus_1;",
        "    const d_minus_2: number = rxn_minus_1;",
        "    const read_minus_out_minus_1: number = out_minus_2;",
        "    return [d_minus_1, d_minus_2, rxn_minus_1, out_minus_1, out_minus_2, read_minus_out_minus_1];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    const p_minus_2: number = 1.0;",
        "    const v_minus_1: number = 1.0;",
        "    const v_minus_2: number = 2.0;",
        "    const v_minus_3: number = 3.0;",
        "    const v_minus_4: number = 4.0;",
        "    const v_minus_5: number = 5.0;",
        "    const p_minus_ia1: number = p_minus_2 + 1.0;",
        "    const v_minus_ia6: number = p_minus_ia1 + 1.0;",
        "    return [[v_minus_1, v_minus_2, v_minus_3, v_minus_4, v_minus_5, v_minus_ia6], [p_minus_2, p_minus_ia1]];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_nasty()).full().split("\n") == [
        "\\begin{longtable}{c|c}",
        "    Parameter name & Parameter value \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{p\\_1}$ & $1.0$ \\\\",
        "     $\\mathrm{p\\_2}$ & $1.0$ \\\\",
        "     $\\mathrm{p\\_ia1}$ & $p-2 + 1.0$ \\\\",
        "    \\caption[Model parameters]{Model parameters}",
        "    \\label{table:model-pars}",
        "\\end{longtable}",
        "",
        "\\begin{longtable}{c|c}",
        "    Model name & Initial concentration \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{v\\_1}$ & $1.0$ \\\\",
        "     $\\mathrm{v\\_2}$ & $2.0$ \\\\",
        "     $\\mathrm{v\\_3}$ & $3.0$ \\\\",
        "     $\\mathrm{v\\_4}$ & $4.0$ \\\\",
        "     $\\mathrm{v\\_5}$ & $5.0$ \\\\",
        "     $\\mathrm{v\\_ia6}$ & $p-ia1 + 1.0$ \\\\",
        "    \\caption[Model variables]{Model variables}",
        "    \\label{table:model-vars}",
        "\\end{longtable}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{d\\_1} = v-1",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{d\\_2} = rxn-1",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{read\\_out\\_1} = out-2",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{out\\_2} = p-1 + v-1",
        "\\end{dmath*}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{rxn\\_1} = d-1 \\cdot v-1",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{out\\_1} = p-1 \\cdot v-1",
        "\\end{dmath*}",
        "",
        "\\begin{align*}",
        "  \\frac{d\\ \\mathrm{v\\_1}}{dt} &= - 1.0 \\cdot out-1 - 1.0 \\cdot rxn-1 \\\\",
        "  \\frac{d\\ \\mathrm{v\\_2}}{dt} &= p-2 \\cdot rxn-1 \\\\",
        "  \\frac{d\\ \\mathrm{v\\_3}}{dt} &= 0 \\\\",
        "  \\frac{d\\ \\mathrm{v\\_4}}{dt} &= 0 \\\\",
        "  \\frac{d\\ \\mathrm{v\\_5}}{dt} &= 0 \\\\",
        "  \\frac{d\\ \\mathrm{v\\_ia6}}{dt} &= 0",
        "\\end{align*}",
    ]
