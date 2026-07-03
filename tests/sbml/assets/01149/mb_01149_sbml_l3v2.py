from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_variable(
        name="p8",
        initial_value=8.0,
    )
