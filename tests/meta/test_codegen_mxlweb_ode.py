from mxlpy import OdeModelBuilder, meta


def decay(x: float, k: float) -> float:
    return -k * x


def readout_fn(x: float) -> float:
    return x / 2.0


def constant(x: float) -> float:
    return x


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def test_generate_model_code_mxlweb_ode_empty() -> None:
    assert meta.generate_model_code_mxlweb(OdeModelBuilder()).split("\n") == [
        'import { OdeModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): OdeModelBuilder {",
        "    return new OdeModelBuilder()",
        "",
        "  }",
    ]


def test_generate_model_code_mxlweb_ode_diff_eq_uses_set_differential() -> None:
    """A diff_eq lowers to .setDifferential, not a fake self-stoichiometry reaction."""
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=decay, args=["x", "k"], initial_value=2.0)
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { OdeModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Minus, Mul, Name } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): OdeModelBuilder {",
        "    return new OdeModelBuilder()",
        '      .addParameter("k", {',
        "        value: 0.5,",
        "        texName: 'k',",
        "      })",
        '      .addVariable("x", {',
        "        value: 2.0,",
        "        texName: 'x',",
        "      })",
        '      .setDifferential("x", new Minus([new Mul([new Name("k"), new Name("x")])]))',
        "  }",
    ]


def test_generate_model_code_mxlweb_ode_readout_uses_add_readout() -> None:
    """A readout lowers to a real .addReadout call, not a downgraded .addAssignment."""
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=decay, args=["x", "k"], initial_value=2.0)
        .add_readout("half_x", fn=readout_fn, args=["x"])
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { OdeModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Minus, Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): OdeModelBuilder {",
        "    return new OdeModelBuilder()",
        '      .addParameter("k", {',
        "        value: 0.5,",
        "        texName: 'k',",
        "      })",
        '      .addVariable("x", {',
        "        value: 2.0,",
        "        texName: 'x',",
        "      })",
        '      .setDifferential("x", new Minus([new Mul([new Name("k"), new Name("x")])]))',
        '      .addReadout("half_x", {',
        '        fn: new Mul([new Num(0.5), new Name("x")]),',
        "        texName: 'half\\\\_x',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_ode_options() -> None:
    """tex_names, sliders and docstring are threaded into the Ode output too."""
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=decay, args=["x", "k"], initial_value=2.0)
    )
    assert meta.generate_model_code_mxlweb(
        model,
        tex_names={"k": "alpha", "x": "X"},
        sliders={"k": {"min": "0", "max": "10", "step": "0.1"}},
        docstring="// my ode model",
    ).split("\n") == [
        'import { OdeModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Minus, Mul, Name } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "// my ode model",
        "export function initModel(): OdeModelBuilder {",
        "    return new OdeModelBuilder()",
        '      .addParameter("k", {',
        "        value: 0.5,",
        "        texName: 'alpha',",
        "        slider: {",
        '          min: "0",',
        '          max: "10",',
        '          step: "0.1",',
        "        },",
        "      })",
        '      .addVariable("x", {',
        "        value: 2.0,",
        "        texName: 'X',",
        "      })",
        '      .setDifferential("x", new Minus([new Mul([new Name("k"), new Name("x")])]))',
        "  }",
    ]


def test_generate_model_code_mxlweb_ode_units() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5, unit="unit_param")
        .add_diff_eq("x", fn=decay, args=["x", "k"], initial_value=2.0, unit="unit_var")
        .add_readout("half_x", fn=readout_fn, args=["x"], unit="unit_readout")
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { OdeModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Minus, Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): OdeModelBuilder {",
        "    return new OdeModelBuilder()",
        '      .addParameter("k", {',
        "        value: 0.5,",
        "        texName: 'k',",
        "        unit: 'unit_param',",
        "      })",
        '      .addVariable("x", {',
        "        value: 2.0,",
        "        texName: 'x',",
        "        unit: 'unit_var',",
        "      })",
        '      .setDifferential("x", new Minus([new Mul([new Name("k"), new Name("x")])]))',
        '      .addReadout("half_x", {',
        '        fn: new Mul([new Num(0.5), new Name("x")]),',
        "        texName: 'half\\\\_x',",
        "        unit: 'unit_readout',",
        "      })",
        "  }",
    ]
