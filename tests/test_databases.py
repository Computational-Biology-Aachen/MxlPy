from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from mxlpy.databases import (
    load_biomodel,
    load_jws_model,
    search_biomodels,
    search_jws_by_reaction,
    search_jws_by_species,
)
from mxlpy.model import Model
from mxlpy.types import Result

SBML_FIXTURE = (
    Path("tests") / "sbml" / "assets" / "00001" / "00001-sbml-l3v2.xml"
).read_bytes()


def _mock_response(content: bytes) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    mock.raise_for_status.return_value = None
    return mock


def _mock_json_response(data: object) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = data
    return mock


def test_load_biomodel_int() -> None:
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_response(SBML_FIXTURE)
    ) as mock_get:
        result = load_biomodel(12)
    assert isinstance(result, Result)
    assert isinstance(result.unwrap_or_err(), Model)
    url = mock_get.call_args[0][0]
    assert "BIOMD0000000012" in url


def test_load_biomodel_str() -> None:
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_response(SBML_FIXTURE)
    ) as mock_get:
        result = load_biomodel("BIOMD0000000012")
    assert isinstance(result, Result)
    assert isinstance(result.unwrap_or_err(), Model)
    url = mock_get.call_args[0][0]
    assert "BIOMD0000000012" in url


def test_load_biomodel_http_error() -> None:
    with patch("mxlpy.databases.requests.get", side_effect=requests.HTTPError("404")):
        result = load_biomodel(12)
    assert isinstance(result, Result)
    assert isinstance(result.value, Exception)


def test_search_biomodels() -> None:
    payload = {"models": [{"id": "BIOMD0000000012", "name": "Edelstein"}]}
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_json_response(payload)
    ):
        hits = search_biomodels("edelstein", n=1)
    assert hits == [{"id": "BIOMD0000000012", "name": "Edelstein"}]


def test_search_biomodels_empty() -> None:
    with patch("mxlpy.databases.requests.get", return_value=_mock_json_response({})):
        hits = search_biomodels("nothing")
    assert hits == []


def test_load_jws_model() -> None:
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_response(SBML_FIXTURE)
    ):
        result = load_jws_model("teusink")
    assert isinstance(result, Result)
    assert isinstance(result.unwrap_or_err(), Model)


def test_load_jws_model_http_error() -> None:
    with patch("mxlpy.databases.requests.get", side_effect=requests.HTTPError("404")):
        result = load_jws_model("nonexistent")
    assert isinstance(result, Result)
    assert isinstance(result.value, Exception)


def test_search_jws_by_species() -> None:
    payload = [{"slug": "teusink", "name": "Teusink2000"}]
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_json_response(payload)
    ):
        hits = search_jws_by_species("atp")
    assert hits == payload


def test_search_jws_by_reaction() -> None:
    payload = [{"slug": "teusink", "name": "Teusink2000"}]
    with patch(
        "mxlpy.databases.requests.get", return_value=_mock_json_response(payload)
    ):
        hits = search_jws_by_reaction("pfk")
    assert hits == payload


def test_load_biomodel_raises_on_unwrap_when_error() -> None:
    with patch(
        "mxlpy.databases.requests.get", side_effect=requests.ConnectionError("down")
    ):
        result = load_biomodel(1)
    with pytest.raises(requests.ConnectionError):
        result.unwrap_or_err()
