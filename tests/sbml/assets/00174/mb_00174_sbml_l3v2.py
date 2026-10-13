# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from mxlpy import KineticModelBuilder


def S1() -> float:
    return 7


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_derived(
        "S1",
        fn=S1,
        args=[],
    )
