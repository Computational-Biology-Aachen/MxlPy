from mxlpy import InitialAssignment, Model, meta


def _ia_init(p1: float) -> float:
    return p1 * 2.0


def model_1_init_var() -> Model:
    return (
        Model()
        .add_variable(
            "v1",
            initial_value=InitialAssignment(fn=_ia_init, args=["p1"]),
        )
        .add_parameter("p1", value=1.0)
    )


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_1_init_var()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 1> model(double time, const std::array<double, 1>& variables) {",
        "    const auto [v1] = variables;",
        "    double p1 = 1.0;",
        "    double dv1dt = 0;",
        "    return {dv1dt};",
        "}",
        "",
        "std::array<double, 0> derived(double time, const std::array<double, 1>& variables) {",
        "    const auto [v1] = variables;",
        "    double p1 = 1.0;",
        "    return {};",
        "}",
        "",
        "std::pair<std::array<double, 1>, std::array<double, 1>> inits() {",
        "    double p1 = 1.0;",
        "    double v1 = 2.0*p1;",
        "    return {{v1}, {p1}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_1_init_var()).full().split("\n") == [
        "function model(time, variables)",
        "    v1 = variables",
        "    p1 = 1.0",
        "    dv1dt = 0",
        "    return [dv1dt]",
        "end",
        "",
        "function derived(time, variables)",
        "    v1 = variables",
        "    p1 = 1.0",
        "    return [()]",
        "end",
        "",
        "function inits()",
        "    p1 = 1.0",
        "    v1 = 2.0 * p1",
        "    return [[v1], [p1]]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_1_init_var()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [v1] = num2cell(variables){:};",
        "    p1 = 1.0;",
        "    dv1dt = 0;",
        "    dydt = [dv1dt]';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [v1] = num2cell(variables){:};",
        "    p1 = 1.0;",
        "    out = [];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    p1 = 1.0;",
        "    v1 = 2.0*p1;",
        "    vars = [v1]';",
        "    pars = [p1]';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_1_init_var()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let dv1dt: f64 = 0;",
        "    return [dv1dt]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 1]) -> [f64; 0] {",
        "    let [v1] = *variables;",
        "    let p1: f64 = 1.0;",
        "    return [()]",
        "}",
        "",
        "fn inits() -> ([f64; 1], [f64; 1]) {",
        "    let p1: f64 = 1.0;",
        "    let v1: f64 = 2.0*p1;",
        "    return ([v1], [p1])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_1_init_var()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [v1] = variables;",
        "    const p1: number = 1.0;",
        "    const dv1dt: number = 0;",
        "    return [dv1dt];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [v1] = variables;",
        "    const p1: number = 1.0;",
        "    return [[]];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    const p1: number = 1.0;",
        "    const v1: number = 2.0*p1;",
        "    return [[v1], [p1]];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_1_init_var()).full().split("\n") == [
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
        "     $\\mathrm{v1}$ & $2.0 \\cdot p_{1}$ \\\\",
        "    \\caption[Model variables]{Model variables}",
        "    \\label{table:model-vars}",
        "\\end{longtable}",
        "",
        "\\begin{align*}",
        "  \\frac{d\\ \\mathrm{v1}}{dt} &= 0",
        "\\end{align*}",
    ]
