import math

from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter("P", value=math.inf)
        .add_parameter("Q", value=float("-inf"))
        .add_parameter("R", value=math.nan)
    )
