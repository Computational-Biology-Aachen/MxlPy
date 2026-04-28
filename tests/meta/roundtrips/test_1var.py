from mxlpy import Model, meta


def model_1var() -> Model:
    return Model().add_variable("v1", initial_value=1.0)


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_1var()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 1> model(double time, const std::array<double, 1>& variables) {",
        "    const auto [v1] = variables;",
        "",
        "    double dv1dt = 0;",
        "    return {dv1dt};",
        "}",
        "",
        "std::array<double, 0> derived(double time, const std::array<double, 1>& variables) {",
        "    const auto [v1] = variables;",
        "",
        "    return {};",
        "}",
        "",
        "std::pair<std::array<double, 1>, std::array<double, 0>> inits() {",
        "    double v1 = 1.0;",
        "    return {{v1}, {}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_1var()).full().split("\n") == [
        "function model(time, variables)",
        "    v1 = variables",
        "",
        "    dv1dt = 0",
        "    return [dv1dt]",
        "end",
        "",
        "function derived(time, variables)",
        "    v1 = variables",
        "",
        "    return [()]",
        "end",
        "",
        "function inits()",
        "    v1 = 1.0",
        "    return [[v1], []]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_1var()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [v1] = num2cell(variables){:};",
        "",
        "    dv1dt = 0;",
        "    dydt = [dv1dt]';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [v1] = num2cell(variables){:};",
        "",
        "    out = [];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    v1 = 1.0;",
        "    vars = [v1]';",
        "    pars = []';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_1var()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "",
        "    let dv1dt: f64 = 0;",
        "    return [dv1dt]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 1]) -> [f64; 0] {",
        "    let [v1] = *variables;",
        "",
        "    return [()]",
        "}",
        "",
        "fn inits() -> ([f64; 1], [f64; 0]) {",
        "    let v1: f64 = 1.0;",
        "    return ([v1], [])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_1var()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [v1] = variables;",
        "",
        "    const dv1dt: number = 0;",
        "    return [dv1dt];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [v1] = variables;",
        "",
        "    return [[]];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    const v1: number = 1.0;",
        "    return [[v1], []];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_1var()).full().split("\n") == [
        "\\begin{longtable}{c|c}",
        "    Model name & Initial concentration \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{v1}$ & $1.0$ \\\\",
        "    \\caption[Model variables]{Model variables}",
        "    \\label{table:model-vars}",
        "\\end{longtable}",
        "",
        "\\begin{align*}",
        "  \\frac{d\\ \\mathrm{v1}}{dt} &= 0",
        "\\end{align*}",
    ]
