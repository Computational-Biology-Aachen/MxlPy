from mxlpy import Model


def create_model() -> Model:
    return Model().add_variable("p", initial_value=3.0)
