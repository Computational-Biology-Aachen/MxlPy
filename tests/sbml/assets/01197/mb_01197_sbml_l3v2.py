from mxlpy import Model


def create_model() -> Model:
    return Model().add_parameter("C", value=4.0)
