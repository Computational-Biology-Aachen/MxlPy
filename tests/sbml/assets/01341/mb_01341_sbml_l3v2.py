from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("S1", initial_value=2.0)
        .add_parameter("C", value=1.0)
    )
