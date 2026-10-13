# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mxlpy.parameterise import get_km_and_kcat_from_brenda


@pytest.fixture
def mock_brenda() -> Generator[MagicMock, Any, None]:
    """Mock to avoid actual database access during testing."""
    with patch("mxlpy.parameterise.Brenda") as mock_brenda_class:
        mock_brenda_instance = MagicMock()
        mock_brenda_class.return_value = mock_brenda_instance

        mock_kms = pd.DataFrame(
            {
                "substrate": ["glucose", "fructose"],
                "km": [1.0, 2.0],
                "unit": ["mM", "mM"],
            }
        )

        mock_kcats = pd.DataFrame(
            {
                "substrate": ["glucose", "fructose"],
                "kcat": [10.0, 20.0],
                "unit": ["1/s", "1/s"],
            }
        )

        mock_brenda_instance.get_kms_and_kcats.return_value = (mock_kms, mock_kcats)
        yield mock_brenda_instance


def test_get_km_and_kcat_from_brenda(mock_brenda: MagicMock) -> None:
    test_path = Path("/test/path/to/brenda.txt")
    ec = "1.1.1.1"

    kms, kcats = get_km_and_kcat_from_brenda(ec, test_path)

    mock_brenda.get_kms_and_kcats.assert_called_once_with(
        ec=ec, filter_mutant=True, filter_missing_sequences=True
    )

    assert isinstance(kms, pd.DataFrame)
    assert isinstance(kcats, pd.DataFrame)
    assert len(kms) == 2
    assert len(kcats) == 2
