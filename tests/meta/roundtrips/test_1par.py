from mxlpy import Model, meta


def model_1par() -> Model:
    return Model().add_parameter("p1", value=1.0)


def test_generate_model_code_jax() -> None:
    assert meta.generate_model_code_jax(model_1par()).full().split("\n") == [
        "import jax",
        "import jax.numpy as jnp",
        "",
        "def model(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    p1 = 1.0",
        "    return jnp.array([])",
        "",
        "def derived(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    p1 = 1.0",
        "    return jnp.array([])",
        "",
        "def fluxes(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    p1 = 1.0",
        "    return jnp.array([])",
        "",
        "def nv(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    return jnp.array([])",
        "",
        "def inits(ts: jax.Array, variables: jax.Array, args: jax.Array) -> tuple[jax.Array, jax.Array]:",
        "    return (jnp.array([]), jnp.array([]))",
    ]


def test_generate_model_code_cpp() -> None:
    assert meta.generate_model_code_cpp(model_1par()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 0> model(double time, const std::array<double, 0>& variables) {",
        "    double p1 = 1.0;",
        "    return {};",
        "}",
        "",
        "std::array<double, 0> derived(double time, const std::array<double, 0>& variables) {",
        "    double p1 = 1.0;",
        "    return {};",
        "}",
        "",
        "ignored fluxes(double time, const std::array<double, 0>& variables) {",
        "    double p1 = 1.0;",
        "    return {};",
        "}",
        "",
        "ignored nv(const std::array<double, 0>& fluxes) {",
        "    return {};",
        "}",
        "",
        "std::pair<std::array<double, 0>, std::array<double, 0>> inits() {",
        "    return {{}, {}};",
        "}",
    ]


def test_generate_model_code_jl() -> None:
    assert meta.generate_model_code_jl(model_1par()).full().split("\n") == [
        "function model(time, variables)",
        "    p1 = 1.0",
        "    return []",
        "end",
        "",
        "function derived(time, variables)",
        "    p1 = 1.0",
        "    return []",
        "end",
        "",
        "function fluxes(time, variables)",
        "    p1 = 1.0",
        "    return []",
        "end",
        "",
        "function nv(fluxes)",
        "    return []",
        "end",
        "",
        "function inits()",
        "    return ([], [])",
        "end",
    ]


def test_generate_model_code_matlab() -> None:
    assert meta.generate_model_code_matlab(model_1par()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    p1 = 1.0;",
        "    dydt = []';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    p1 = 1.0;",
        "    out = [];",
        "end",
        "",
        "function out = fluxes(t, variables)",
        "    p1 = 1.0;",
        "    out = [];",
        "end",
        "",
        "function out = nv(fluxes)",
        "    out = [];",
        "end",
        "",
        "function [vars, pars] = inits()",
        "    vars = []';",
        "    pars = []';",
        "end",
    ]


def test_generate_model_code_rs() -> None:
    assert meta.generate_model_code_rs(model_1par()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let p1: f64 = 1.0;",
        "    return []",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let p1: f64 = 1.0;",
        "    return []",
        "}",
        "",
        "fn fluxes(time: f64, variables: &[f64; 0]) -> ignored {",
        "    let p1: f64 = 1.0;",
        "    return []",
        "}",
        "",
        "fn nv(fluxes: &[f64; 0]) -> ignored {",
        "    return []",
        "}",
        "",
        "fn inits() -> ([f64; 0], [f64; 0]) {",
        "    return ([], [])",
        "}",
    ]


def test_generate_model_code_ts() -> None:
    assert meta.generate_model_code_ts(model_1par()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    const p1: number = 1.0;",
        "    return [];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    const p1: number = 1.0;",
        "    return [];",
        "};",
        "",
        "function fluxes(time: number, variables: number[]): number[] {",
        "    const p1: number = 1.0;",
        "    return [];",
        "};",
        "",
        "function nv(fluxes: number[]): number[] {",
        "    return [];",
        "};",
        "",
        "function inits(): [number[], number[]] {",
        "    return ([], []);",
        "};",
    ]


def test_generate_model_code_latex() -> None:
    assert meta.generate_model_code_latex(model_1par()).full().split("\n") == [
        "\\begin{longtable}{c|c}",
        "    Parameter name & Parameter value \\\\",
        "    \\hline",
        "    \\endhead",
        "     $\\mathrm{p1}$ & $1.0$ \\\\",
        "    \\caption[Model parameters]{Model parameters}",
        "    \\label{table:model-pars}",
        "\\end{longtable}",
    ]
