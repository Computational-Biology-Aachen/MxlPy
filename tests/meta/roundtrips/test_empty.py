from mxlpy import Model, meta


def model_empty() -> Model:
    return Model()


def test_generate_model_code_jax() -> None:
    assert meta.generate_model_code_jax(model_empty()).full().split("\n") == [
        "import jax",
        "import jax.numpy as jnp",
        "",
        "def model(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    return jnp.array([])",
        "",
        "def derived(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
        "    return jnp.array([])",
        "",
        "def fluxes(ts: jax.Array, variables: jax.Array, args: jax.Array) -> jax.Array:",
        "",
        "",
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
    assert meta.generate_model_code_cpp(model_empty()).full().split("\n") == [
        "#include <array>",
        "#include <cmath>",
        "#include <utility>",
        "",
        "",
        "std::array<double, 0> model(double time, const std::array<double, 0>& variables) {",
        "    return {};",
        "}",
        "",
        "std::array<double, 0> derived(double time, const std::array<double, 0>& variables) {",
        "    return {};",
        "}",
        "",
        "ignored fluxes(double time, const std::array<double, 0>& variables) {",
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
    assert meta.generate_model_code_jl(model_empty()).full().split("\n") == [
        "function model(time, variables)",
        "    return []",
        "end",
        "",
        "function derived(time, variables)",
        "    return []",
        "end",
        "",
        "function fluxes(time, variables)",
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
    assert meta.generate_model_code_matlab(model_empty()).full().split("\n") == [
        "function dydt = model(t, variables)",
        "    dydt = []';",
        "end",
        "",
        "function out = derived(t, variables)",
        "    out = [];",
        "end",
        "",
        "function out = fluxes(t, variables)",
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
    assert meta.generate_model_code_rs(model_empty()).full().split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    return []",
        "}",
        "",
        "fn derived(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    return []",
        "}",
        "",
        "fn fluxes(time: f64, variables: &[f64; 0]) -> ignored {",
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
    assert meta.generate_model_code_ts(model_empty()).full().split("\n") == [
        "function model(time: number, variables: number[]): number[] {",
        "    return [];",
        "};",
        "",
        "function derived(time: number, variables: number[]): number[] {",
        "    return [];",
        "};",
        "",
        "function fluxes(time: number, variables: number[]): number[] {",
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
    assert meta.generate_model_code_latex(model_empty()).full().split("\n") == [""]
