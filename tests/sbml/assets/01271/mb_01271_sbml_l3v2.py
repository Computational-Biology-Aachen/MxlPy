from mxlpy import Model


def create_model() -> Model:
    return Model().add_parameter("p1", value=2.0)
