from mxlpy import Model


def create_model() -> Model:
    return Model().add_parameter("p", value=3.0)
