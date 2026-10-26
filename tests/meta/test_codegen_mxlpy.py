# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from mxlpy import KineticModelBuilder, Slider
from mxlpy.meta import generate_mxlpy_code


def test_generate_mxlpy_code() -> None:
    # FIXME: implement this
    assert True


def test_generate_mxlpy_code_from_symbolic_repr() -> None:
    # FIXME: implement this
    assert True


def test_generate_mxlpy_code_names() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0, display_name="Rate", tex_name="k_1")
        .add_variable(
            "x",
            2.0,
            display_name="Substrate",
            slider=Slider(min="0", max="5", step="1"),
        )
        .add_variable("y", 3.0)
    )
    namespace: dict[str, object] = {}
    exec(generate_mxlpy_code(model), namespace)  # noqa: S102
    restored = namespace["create_model"]()

    parameter = restored.get_raw_parameters()["k"]
    assert (parameter.display_name, parameter.tex_name) == ("Rate", "k_1")
    assert restored.get_raw_variables()["x"].display_name == "Substrate"
    assert restored.get_raw_variables()["x"].slider == Slider(
        min="0", max="5", step="1"
    )
    assert restored.get_raw_variables()["y"].display_name is None
