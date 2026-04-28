from mxlpy import Model


def create_model() -> Model:
    return Model().add_parameter("P", value=3.7)
