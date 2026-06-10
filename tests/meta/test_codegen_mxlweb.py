from mxlpy import Model, meta


def constant(x: float) -> float:
    return x


def mass_action_1s(s1: float, k: float) -> float:
    return k * s1


def test_generate_model_code_mxlweb_empty() -> None:
    assert meta.generate_model_code_mxlweb(Model()).split("\n") == [
        'import { KineticModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): KineticModelBuilder {",
        "    return new KineticModelBuilder()",
        "",
        "  }",
    ]


def test_generate_model_code_mxlweb_parameter() -> None:
    model = Model().add_parameter("p1", value=1.0)
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
    model = Model().add_variable("v1", initial_value=1.0)
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
        Model()
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
        Model()
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
        Model()
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
    model = Model().add_parameter("p1", value=1.0).add_variable("v1", initial_value=2.0)
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
