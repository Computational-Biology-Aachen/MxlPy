from mxlpy import SteadyStateModelBuilder, meta


def double(p: float) -> float:
    return 2.0 * p


def test_generate_model_code_mxlweb_steady_state_empty() -> None:
    assert meta.generate_model_code_mxlweb(SteadyStateModelBuilder()).split("\n") == [
        'import { SteadyStateModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): SteadyStateModelBuilder {",
        "    return new SteadyStateModelBuilder()",
        "",
        "  }",
    ]


def test_generate_model_code_mxlweb_steady_state_parameter() -> None:
    model = SteadyStateModelBuilder().add_parameter("p1", value=1.0)
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { SteadyStateModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): SteadyStateModelBuilder {",
        "    return new SteadyStateModelBuilder()",
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'p1',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_steady_state_derived() -> None:
    """No .addVariable/.addReaction/.setDifferential/.addReadout — only parameters+assignments exist here."""
    model = (
        SteadyStateModelBuilder()
        .add_parameter("p", 2.0)
        .add_derived("d", fn=double, args=["p"])
    )
    assert meta.generate_model_code_mxlweb(model).split("\n") == [
        'import { SteadyStateModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import { Mul, Name, Num } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "export function initModel(): SteadyStateModelBuilder {",
        "    return new SteadyStateModelBuilder()",
        '      .addParameter("p", {',
        "        value: 2.0,",
        "        texName: 'p',",
        "      })",
        '      .addAssignment("d", {',
        '        fn: new Mul([new Num(2.0), new Name("p")]),',
        "        texName: 'd',",
        "      })",
        "  }",
    ]


def test_generate_model_code_mxlweb_steady_state_options() -> None:
    """tex_names, sliders and docstring are threaded into the SteadyState output too."""
    model = SteadyStateModelBuilder().add_parameter("p1", value=1.0)
    assert meta.generate_model_code_mxlweb(
        model,
        tex_names={"p1": "alpha"},
        sliders={"p1": {"min": "0", "max": "10", "step": "0.1"}},
        docstring="// my ss model",
    ).split("\n") == [
        'import { SteadyStateModelBuilder } from "@computational-biology-aachen/mxlweb-core";',
        'import {  } from "@computational-biology-aachen/mxlweb-core/mathml";',
        "",
        "// my ss model",
        "export function initModel(): SteadyStateModelBuilder {",
        "    return new SteadyStateModelBuilder()",
        '      .addParameter("p1", {',
        "        value: 1.0,",
        "        texName: 'alpha',",
        "        slider: {",
        '          min: "0",',
        '          max: "10",',
        '          step: "0.1",',
        "        },",
        "      })",
        "  }",
    ]
