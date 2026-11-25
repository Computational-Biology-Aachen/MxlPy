# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numerical parameter identifiability and uncertainty.

- `fisher_information`: asymptotic covariance, Wald intervals and locally
  non-identifiable parameter combinations from one sensitivity evaluation
- `profile_likelihood`: likelihood-ratio confidence intervals and
  identifiability classification from refits along each parameter
- `identifiability_test`: fast check for flat directions around the optimum
  by pulling the parameters away from it
- `related_parameters`: identifiable parameter combinations from the
  parameter paths along profiles
- `prediction_profile`, `prediction_band`: likelihood-based confidence
  intervals and bands of model predictions (`Prediction`)
- `validation_profile`: whether a new data point is consistent with the
  calibrated model
- `profile_likelihood_2d`: joint confidence regions of two parameters, or a
  parameter and a prediction
"""

from __future__ import annotations

from mxlpy.fit._problem import Prediction
from mxlpy.identify._fisher import FisherInformation, fisher_information
from mxlpy.identify._prediction import (
    PredictionBand,
    PredictionProfile,
    ValidationProfile,
    prediction_band,
    prediction_profile,
    validation_profile,
)
from mxlpy.identify._profile import (
    Identifiability,
    Profile,
    ProfileLikelihood,
    SideStatus,
    profile_likelihood,
)
from mxlpy.identify._profile2d import ProfileLikelihood2D, profile_likelihood_2d
from mxlpy.identify._pulling import (
    IdentifiabilityTest,
    PullingRound,
    identifiability_test,
)
from mxlpy.identify._related import (
    ParameterRelation,
    RelatedParameters,
    related_parameters,
)

__all__ = [
    "FisherInformation",
    "Identifiability",
    "IdentifiabilityTest",
    "ParameterRelation",
    "Prediction",
    "PredictionBand",
    "PredictionProfile",
    "Profile",
    "ProfileLikelihood",
    "ProfileLikelihood2D",
    "PullingRound",
    "RelatedParameters",
    "SideStatus",
    "ValidationProfile",
    "fisher_information",
    "identifiability_test",
    "prediction_band",
    "prediction_profile",
    "profile_likelihood",
    "profile_likelihood_2d",
    "related_parameters",
    "validation_profile",
]
