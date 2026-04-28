from mxlpy import Model, meta


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def _readout_ratio(a: float, b: float) -> float:
    return a / b


def model_1rdo() -> Model:
    return (
        Model()
        .add_variable("v1", initial_value=1.0)
        .add_variable("v2", initial_value=2.0)
        .add_parameter("p1", value=1.0)
        .add_reaction(
            "r1",
            fn=mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
        )
        .add_reaction(
            "r2",
            fn=mass_action_1s,
            args=["v2", "p1"],
            stoichiometry={"v2": -1.0},
        )
        .add_readout(
            "ro1",
            fn=_readout_ratio,
            args=["r1", "r2"],
        )
    )


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_1rdo()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 2> model(double time, const std::array<double, 2>& variables) {",
        "    const auto [v1, v2] = variables;",
        "    double p1 = 1.0;",
        "    double r1 = p1*v1;",
        "    double r2 = p1*v2;",
        "    double dv1dt = -1.0*r1;",
        "    double dv2dt = -1.0*r2;",
        "    return {dv1dt, dv2dt};",
        "}",
        "",
        "std::array<double, 3> derived(double time, const std::array<double, 2>& variables) {",
        "    const auto [v1, v2] = variables;",
        "    double p1 = 1.0;",
        "    double r1 = p1*v1;",
        "    double r2 = p1*v2;",
        "    double ro1 = r1/r2;",
        "    return {r1, r2, ro1};",
        "}",
        "",
        "std::pair<std::array<double, 2>, std::array<double, 0>> inits() {",
        "    double v1 = 1.0;",
        "    double v2 = 2.0;",
        "    return {{v1, v2}, {}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_1rdo()).full().split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    r1 = p1 .* v1",
        "    r2 = p1 .* v2",
        "    dv1dt = -1.0 * r1",
        "    dv2dt = -1.0 * r2",
        "    return [dv1dt, dv2dt]",
        "end",
        "",
        "function derived(time, variables)",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    r1 = p1 .* v1",
        "    r2 = p1 .* v2",
        "    ro1 = r1 ./ r2",
        "    return [r1, r2, ro1]",
        "end",
        "",
        "function inits()",
        "    v1 = 1.0",
        "    v2 = 2.0",
        "    return [[v1, v2], []]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_1rdo()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [v1, v2] = num2cell(variables){:};",
        "    p1 = 1.0;",
        "    r1 = p1.*v1;",
        "    r2 = p1.*v2;",
        "    dv1dt = -1.0*r1;",
        "    dv2dt = -1.0*r2;",
        "    dydt = [dv1dt, dv2dt]';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [v1, v2] = num2cell(variables){:};",
        "    p1 = 1.0;",
        "    r1 = p1.*v1;",
        "    r2 = p1.*v2;",
        "    ro1 = r1./r2;",
        "    out = [r1, r2, ro1];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    v1 = 1.0;",
        "    v2 = 2.0;",
        "    vars = [v1, v2]';",
        "    pars = []';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_1rdo()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let r1: f64 = p1*v1;",
        "    let r2: f64 = p1*v2;",
        "    let dv1dt: f64 = -1.0*r1;",
        "    let dv2dt: f64 = -1.0*r2;",
        "    return [dv1dt, dv2dt]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 2]) -> [f64; 3] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let r1: f64 = p1*v1;",
        "    let r2: f64 = p1*v2;",
        "    let ro1: f64 = r1/r2;",
        "    return [r1, r2, ro1]",
        "}",
        "",
        "fn inits() -> ([f64; 2], [f64; 0]) {",
        "    let v1: f64 = 1.0;",
        "    let v2: f64 = 2.0;",
        "    return ([v1, v2], [])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_1rdo()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [v1, v2] = variables;",
        "    const p1: number = 1.0;",
        "    const r1: number = p1*v1;",
        "    const r2: number = p1*v2;",
        "    const dv1dt: number = -1.0*r1;",
        "    const dv2dt: number = -1.0*r2;",
        "    return [dv1dt, dv2dt];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [v1, v2] = variables;",
        "    const p1: number = 1.0;",
        "    const r1: number = p1*v1;",
        "    const r2: number = p1*v2;",
        "    const ro1: number = r1/r2;",
        "    return [r1, r2, ro1];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    const v1: number = 1.0;",
        "    const v2: number = 2.0;",
        "    return [[v1, v2], []];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_1rdo()).full().split("\n") == [
        "\\begin{longtable}{c|c}",
        "    Parameter name & Parameter value \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{p1}$ & $1.0$ \\\\",
        "    \\caption[Model parameters]{Model parameters}",
        "    \\label{table:model-pars}",
        "\\end{longtable}",
        "",
        "\\begin{longtable}{c|c}",
        "    Model name & Initial concentration \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{v1}$ & $1.0$ \\\\",
        "     $\\mathrm{v2}$ & $2.0$ \\\\",
        "    \\caption[Model variables]{Model variables}",
        "    \\label{table:model-vars}",
        "\\end{longtable}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{ro1} = r_{1} / r_{2}",
        "\\end{dmath*}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{r1} = p_{1} \\cdot v_{1}",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{r2} = p_{1} \\cdot v_{2}",
        "\\end{dmath*}",
        "",
        "\\begin{align*}",
        "  \\frac{d\\ \\mathrm{v1}}{dt} &= - 1.0 \\cdot r_{1} \\\\",
        "  \\frac{d\\ \\mathrm{v2}}{dt} &= - 1.0 \\cdot r_{2}",
        "\\end{align*}",
    ]
