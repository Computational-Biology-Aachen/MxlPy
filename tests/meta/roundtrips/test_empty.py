from mxlpy import Model, meta


def model_empty() -> Model:
    return Model()


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_empty()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 0> model(double time, const std::array<double, 0>& variables) {",
        "    const auto [] = variables;",
        "",
        "",
        "    return {};",
        "}",
        "",
        "std::array<double, 0> derived(double time, const std::array<double, 0>& variables) {",
        "    const auto [] = variables;",
        "",
        "    return {};",
        "}",
        "",
        "std::pair<std::array<double, 0>, std::array<double, 0>> inits() {",
        "",
        "    return {{}, {}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_empty()).full().split("\n") == [
        "function model(time, variables)",
        "",
        "",
        "",
        "    return [()]",
        "end",
        "",
        "function derived(time, variables)",
        "",
        "",
        "    return [()]",
        "end",
        "",
        "function inits()",
        "",
        "    return [[], []]",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_empty()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    [] = num2cell(variables){:};",
        "",
        "",
        "    dydt = []';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    [] = num2cell(variables){:};",
        "",
        "    out = [];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "",
        "    vars = []';",
        "    pars = []';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_empty()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let [] = *variables;",
        "",
        "",
        "    return [()]",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let [] = *variables;",
        "",
        "    return [()]",
        "}",
        "",
        "fn inits() -> ([f64; 0], [f64; 0]) {",
        "",
        "    return ([], [])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_empty()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const [] = variables;",
        "",
        "",
        "    return [[]];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const [] = variables;",
        "",
        "    return [[]];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "",
        "    return [[], []];",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_empty()).full().split("\n") == [""]
