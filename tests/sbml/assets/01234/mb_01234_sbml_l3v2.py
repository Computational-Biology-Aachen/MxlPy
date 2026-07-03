from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter("p", value=3.0)
