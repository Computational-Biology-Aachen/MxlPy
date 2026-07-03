from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter("p1", value=2.0)
