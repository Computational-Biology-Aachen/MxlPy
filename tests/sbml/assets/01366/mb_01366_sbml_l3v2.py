from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter("J1", value=5.0)
