# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras", exc_type=ImportError)

from mxlpy.npe._keras import SteadyStateTrainer


def test_default_optimizer_not_shared() -> None:
    """Each model gets its own default optimizer, see tests/surrogates/test_keras.py."""
    features = pd.DataFrame({"x": np.linspace(0, 1, 8), "y": np.linspace(1, 2, 8)})
    targets = pd.DataFrame({"k": features["x"] + features["y"]})
    first = SteadyStateTrainer(features, targets).train(epochs=2)
    second = SteadyStateTrainer(features, targets).train(epochs=2)
    assert first.model.optimizer is not second.model.optimizer
