from sympy.physics.units.quantities import Quantity

from mxlpy import KineticModelBuilder, meta, units


def constant(x: float) -> float:
    return x


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def readout_fn(x: float) -> float:
    return x / 2.0


def test_generate_model_code_mxlweb_empty() -> None:
    assert meta.generate_model_code_mxlweb(KineticModelBuilder()).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        "",
        "  }",
    ]


def test_generate_model_code_mxlweb_parameter() -> None:
    model = KineticModelBuilder().add_parameter("p1", value=1.0)
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'p1',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_variable() -> None:
    model = KineticModelBuilder().add_variable("v1", initial_value=1.0)
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addVariable("v1", {',
        "        value: 1.0,",
        "        texName: 'v1',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_derived() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("x1", initial_value=1.0)
        .add_derived(
            "d1",
            fn=constant,
            args=["x1"],
        )
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Name } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addVariable("x1", {',
        "        value: 1.0,",
        "        texName: 'x1',",
        "      })",
        '      .addAssignment("d1", {',
        '        fn: new Name("x1"),',
        "        texName: 'd1',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_reaction() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("p1", value=1.0)
        .add_reaction(
            "r1",
            fn=mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
        )
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'p1',",
        "      })",
        '      .addVariable("v1", {',
        "        value: 1.0,",
        "        texName: 'v1',",
        "      })",
        '      .addReaction("r1", {',
        '        fn: new Mul([new Name("p1"), new Name("v1")]),',
        '        stoichiometry: [{ name: "v1", value: new Num(-1.0) }],',
        "        texName: 'r1',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_topo_order() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("x1", initial_value=1.0)
        .add_derived("d1", fn=constant, args=["x1"])
        .add_derived("d2", fn=constant, args=["r1"])
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
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addVariable("x1", {',
        "        value: 1.0,",
        "        texName: 'x1',",
        "      })",
        '      .addAssignment("d1", {',
        '        fn: new Name("x1"),',
        "        texName: 'd1',",
        "      })",
        '      .addAssignment("d2", {',
        '        fn: new Name("r1"),',
        "        texName: 'd2',",
        "      })",
        '      .addReaction("r1", {',
        '        fn: new Mul([new Name("d1"), new Name("x1")]),',
        '        stoichiometry: [{ name: "x1", value: new Num(-1.0) }],',
        "        texName: 'r1',",
        "      })",
        '      .addReaction("r2", {',
        '        fn: new Mul([new Name("d2"), new Name("x1")]),',
        '        stoichiometry: [{ name: "x1", value: new Num(1.0) }],',
        "        texName: 'r2',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_options() -> None:
    """tex_names, sliders and docstring are threaded into the output."""
    model = (
        KineticModelBuilder()
        .add_parameter("p1", value=1.0)
        .add_variable("v1", initial_value=2.0)
    )
    assert meta.generate_model_code_mxlweb(
        model,
        tex_names={"p1": "alpha", "v1": "V"},
        sliders={"p1": {"min": "0", "max": "10", "step": "0.1"}},
        docstring="// my model",
    ).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "// my model",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'alpha',",
        "        slider: {",
        '          min: "0",',
        '          max: "10",',
        '          step: "0.1",',
        "        },",
        "      })",
        '      .addVariable("v1", {',
        "        value: 2.0,",
        "        texName: 'V',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_units() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("p1", value=1.0, unit=units.per_second)
        .add_variable("x1", initial_value=2.0, unit=units.mmol)
        .add_derived("d1", fn=constant, args=["x1"], unit=units.mmol)
        .add_reaction(
            "r1",
            fn=mass_action_1s,
            args=["x1", "p1"],
            stoichiometry={"x1": -1.0},
            unit=units.mmol_s,
        )
        .add_readout(
            "half_x1",
            fn=readout_fn,
            args=["x1"],
            unit=Quantity("OD600", abbrev="OD600"),
        )
    )
    mmol = '{"factors": [{"kind": "mole", "prefix": "milli", "exponent": 1}]}'
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { KineticModelBuilder, Unit } from "@computational-biology-aachen/mxlweb-core";',
        'import { Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        '      .addCustomUnit("OD600", {})',
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'p1',",
        '        unit: Unit.fromJson({"factors": [{"kind": "second", "exponent": -1}]}),',
        "      })",
        '      .addVariable("x1", {',
        "        value: 2.0,",
        "        texName: 'x1',",
        f"        unit: Unit.fromJson({mmol}),",
        "      })",
        '      .addAssignment("d1", {',
        '        fn: new Name("x1"),',
        "        texName: 'd1',",
        f"        unit: Unit.fromJson({mmol}),",
        "      })",
        '      .addReaction("r1", {',
        '        fn: new Mul([new Name("p1"), new Name("x1")]),',
        '        stoichiometry: [{ name: "x1", value: new Num(-1.0) }],',
        "        texName: 'r1',",
        '        unit: Unit.fromJson({"factors": [{"kind": "mole", "prefix": "milli", "exponent": 1}, {"kind": "second", "exponent": -1}]}),',
        "      })",
        '      .addReadout("half_x1", {',
        '        fn: new Mul([new Num(0.5), new Name("x1")]),',
        "        texName: 'half\\\\_x1',",
        '        unit: Unit.fromJson({"factors": [{"kind": "OD600", "exponent": 1}]}),',
        "      })",
        "  }",
    ]
