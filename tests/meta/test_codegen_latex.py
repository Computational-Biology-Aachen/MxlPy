# Copyright 2025 Marvin van Aalst, Anna Matuszyńska


def test_default_init() -> None:
    # FIXME: implement this
    assert True


def test_generate_latex_code() -> None:
    # FIXME: implement this
    assert True


def test_get_model_tex_diff() -> None:
    from example_models import get_linear_chain_1v  # noqa: PLC0415
    from mxlpy.meta import generate_latex_diff  # noqa: PLC0415

    diff = generate_latex_diff(
        get_linear_chain_1v(),
        get_linear_chain_1v().update_parameter("k1", 2.0),
        only_changes=True,
    )
    # Only the changed parameter, with its old and new value
    ((_, old, new),) = diff.parameters
    assert (float(old), float(new)) == (1.0, 2.0)
    assert diff.reactions == []
    assert (
        len(generate_latex_diff(get_linear_chain_1v(), get_linear_chain_1v()).reactions)
        == 2
    )


def test_texexport_export_all() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_derived() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_diff_eqs() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_document() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_parameters() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_reactions() -> None:
    # FIXME: implement this
    assert True


def test_texexport_export_variables() -> None:
    # FIXME: implement this
    assert True


def test_texexport_rename_with_glossary() -> None:
    # FIXME: implement this
    assert True


def test_to_tex_export() -> None:
    # FIXME: implement this
    assert True
