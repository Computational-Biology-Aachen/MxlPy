import math

from mxlpy import Model


def create_model() -> Model:
    return (
        Model()
        .add_parameter("P", value=math.inf)
        .add_parameter("Q", value=float("-inf"))
        .add_parameter("R", value=math.nan)
    )
