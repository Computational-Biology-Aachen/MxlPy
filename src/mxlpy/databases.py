"""Model database integration for BioModels and JWS Online."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from mxlpy import sbml
from mxlpy.types import Result

if TYPE_CHECKING:
    from mxlpy.model import Model

__all__ = [
    "load_biomodel",
    "load_jws_model",
    "search_biomodels",
    "search_jws_by_reaction",
    "search_jws_by_species",
]

_BIOMODELS_BASE = "https://www.ebi.ac.uk/biomodels"
_JWS_BASE = "https://jjj.bio.vu.nl/rest"


def _biomodel_id(accession: int | str) -> str:
    if isinstance(accession, int):
        return f"BIOMD{accession:010d}"
    return accession


def _sbml_bytes_to_model(content: bytes) -> Model:
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return sbml.read(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_biomodel(accession: int | str) -> Result[Model]:
    """Load a model from BioModels by accession ID or integer index.

    Parameters
    ----------
    accession
        BioModels accession string (e.g. ``"BIOMD0000000012"``) or integer
        index (e.g. ``12``).

    Returns
    -------
    Result[Model]
        Loaded model, or an exception if the fetch or parse failed.

    Examples
    --------
    >>> model = load_biomodel(12).unwrap_or_err()
    >>> model = load_biomodel("BIOMD0000000012").unwrap_or_err()

    """
    try:
        model_id = _biomodel_id(accession)
        response = requests.get(
            f"{_BIOMODELS_BASE}/{model_id}/download",
            params={"filename": f"{model_id}_url.xml"},
            timeout=30,
        )
        response.raise_for_status()
        return Result(_sbml_bytes_to_model(response.content))
    except Exception as e:  # noqa: BLE001
        return Result(e)


def search_biomodels(query: str, n: int = 10) -> list[dict[str, Any]]:
    """Search the BioModels database.

    Parameters
    ----------
    query
        Free-text search query (e.g. ``"glycolysis"``).
    n
        Maximum number of results to return.

    Returns
    -------
    list[dict[str, Any]]
        List of model metadata dicts with keys including ``id``, ``name``.

    Examples
    --------
    >>> hits = search_biomodels("glycolysis", n=5)

    """
    response = requests.get(
        f"{_BIOMODELS_BASE}/search",
        params={"query": query, "numResults": n, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("models", [])


def load_jws_model(slug: str) -> Result[Model]:
    """Load a model from JWS Online by slug identifier.

    Parameters
    ----------
    slug
        JWS Online model slug (e.g. ``"teusink"``).

    Returns
    -------
    Result[Model]
        Loaded model, or an exception if the fetch or parse failed.

    Examples
    --------
    >>> model = load_jws_model("teusink").unwrap_or_err()

    """
    try:
        response = requests.get(
            f"{_JWS_BASE}/models/{slug}/sbml",
            timeout=30,
        )
        response.raise_for_status()
        return Result(_sbml_bytes_to_model(response.content))
    except Exception as e:  # noqa: BLE001
        return Result(e)


def search_jws_by_species(species: str) -> list[dict[str, Any]]:
    """Search JWS Online models containing a given species.

    Parameters
    ----------
    species
        Species name or identifier (e.g. ``"atp"``).

    Returns
    -------
    list[dict[str, Any]]
        List of model metadata dicts.

    Examples
    --------
    >>> hits = search_jws_by_species("atp")

    """
    response = requests.get(
        f"{_JWS_BASE}/models/",
        params={"species": species},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def search_jws_by_reaction(reaction: str) -> list[dict[str, Any]]:
    """Search JWS Online models containing a given reaction.

    Parameters
    ----------
    reaction
        Reaction name or identifier (e.g. ``"pfk"``).

    Returns
    -------
    list[dict[str, Any]]
        List of model metadata dicts.

    Examples
    --------
    >>> hits = search_jws_by_reaction("pfk")

    """
    response = requests.get(
        f"{_JWS_BASE}/models/",
        params={"reaction": reaction},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
