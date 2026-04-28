from mxlpy import Model


def create_model() -> Model:
    return Model().add_variable("S1", initial_value=2.0).add_parameter("C", value=1.0)
