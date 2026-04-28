from mxlpy import Model, meta


def constant(x: float) -> float:
    return x


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def model_topo_order() -> Model:
    return (
        Model()
        .add_variable("x1", initial_value=1.0)
        .add_derived(
            "d1",
            fn=constant,
            args=["x1"],
        )
        .add_derived(
            "d2",
            fn=constant,
            args=["r1"],
        )
        .add_reaction(
            "r1",
            fn=mass_action_1s,
            args=["x1", "d1"],
            stoichiometry={"x1": -1},
        )
        .add_reaction(
            "r2",
            fn=mass_action_1s,
            args=["x1", "d2"],
            stoichiometry={"x1": 1.0},
        )
    )


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_topo_order()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 1> model(double time, const std::array<double, 1>& variables) {",
        "    const auto [x1] = variables;",
        "    double d1 = x1;",
        "    double r1 = d1*x1;",
        "    double d2 = r1;",
        "    double r2 = d2*x1;",
        "    double dx1dt = -1.0*r1 + 1.0*r2;",
        "    return {dx1dt};",
        "}",
        "",
        "std::array<double, 4> derived(double time, const std::array<double, 1>& variables) {",
        "    const auto [x1] = variables;",
        "    double d1 = x1;",
        "    double r1 = d1*x1;",
        "    double d2 = r1;",
        "    double r2 = d2*x1;",
        "    return {d1, d2, r1, r2};",
        "}",
        "",
        "std::pair<std::array<double, 1>, std::array<double, 0>> inits() {",
        "    double x1 = 1.0;",
        "    return {{x1}, {}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_topo_order()).full().split("\n") == [
        "function model(time, variables)",
        "    x1 = variables",
        "    d1 = x1",
        "    r1 = d1 .* x1",
        "    d2 = r1",
        "    r2 = d2 .* x1",
        "    dx1dt = -1.0 * r1 + 1.0 * r2",
        "    return [dx1dt]",
        "end",
        "",
        "function derived(time, variables)",
        "    x1 = variables",
        "    d1 = x1",
        "    r1 = d1 .* x1",
        "    d2 = r1",
        "    r2 = d2 .* x1",
        "    return [d1, d2, r1, r2]",
        "end",
        "",
        "function inits()",
        "    x1 = 1.0",
        "    return [[x1], []]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_topo_order()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [x1] = num2cell(variables){:};",
        "    d1 = x1;",
        "    r1 = d1.*x1;",
        "    d2 = r1;",
        "    r2 = d2.*x1;",
        "    dx1dt = -1.0*r1 + 1.0*r2;",
        "    dydt = [dx1dt]';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [x1] = num2cell(variables){:};",
        "    d1 = x1;",
        "    r1 = d1.*x1;",
        "    d2 = r1;",
        "    r2 = d2.*x1;",
        "    out = [d1, d2, r1, r2];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    x1 = 1.0;",
        "    vars = [x1]';",
        "    pars = []';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_topo_order()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [x1] = *variables;",
        "    let d1: f64 = x1;",
        "    let r1: f64 = d1*x1;",
        "    let d2: f64 = r1;",
        "    let r2: f64 = d2*x1;",
        "    let dx1dt: f64 = -1.0*r1 + 1.0*r2;",
        "    return [dx1dt]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 1]) -> [f64; 4] {",
        "    let [x1] = *variables;",
        "    let d1: f64 = x1;",
        "    let r1: f64 = d1*x1;",
        "    let d2: f64 = r1;",
        "    let r2: f64 = d2*x1;",
        "    return [d1, d2, r1, r2]",
        "}",
        "",
        "fn inits() -> ([f64; 1], [f64; 0]) {",
        "    let x1: f64 = 1.0;",
        "    return ([x1], [])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_topo_order()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [x1] = variables;",
        "    const d1: number = x1;",
        "    const r1: number = d1*x1;",
        "    const d2: number = r1;",
        "    const r2: number = d2*x1;",
        "    const dx1dt: number = -1.0*r1 + 1.0*r2;",
        "    return [dx1dt];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [x1] = variables;",
        "    const d1: number = x1;",
        "    const r1: number = d1*x1;",
        "    const d2: number = r1;",
        "    const r2: number = d2*x1;",
        "    return [d1, d2, r1, r2];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    const x1: number = 1.0;",
        "    return [[x1], []];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_topo_order()).full().split("\n") == [
        "\\begin{longtable}{c|c}",
        "    Model name & Initial concentration \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{x1}$ & $1.0$ \\\\",
        "    \\caption[Model variables]{Model variables}",
        "    \\label{table:model-vars}",
        "\\end{longtable}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{d1} = x_{1}",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{d2} = r_{1}",
        "\\end{dmath*}",
        "",
        "\\begin{dmath*}",
        "  \\mathrm{r1} = d_{1} \\cdot x_{1}",
        "\\end{dmath*}",
        "\\begin{dmath*}",
        "  \\mathrm{r2} = d_{2} \\cdot x_{1}",
        "\\end{dmath*}",
        "",
        "\\begin{align*}",
        "  \\frac{d\\ \\mathrm{x1}}{dt} &= - 1.0 \\cdot r_{1} + 1.0 \\cdot r_{2}",
        "\\end{align*}",
    ]
