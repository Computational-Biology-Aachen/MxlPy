from mxlpy import Model


def create_model() -> Model:
    return Model().add_parameter("J1", value=5.0)
