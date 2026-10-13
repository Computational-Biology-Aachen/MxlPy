# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from mxlpy import KineticModelBuilder


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter("C", value=4.0)
