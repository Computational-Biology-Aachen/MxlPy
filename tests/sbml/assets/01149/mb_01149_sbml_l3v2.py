from mxlpy import Model


def create_model() -> Model:
    return Model().add_variable(
        name="p8",
        initial_value=8.0,
    )
