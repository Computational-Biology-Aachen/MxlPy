from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_variable("p", initial_value=3.0)
