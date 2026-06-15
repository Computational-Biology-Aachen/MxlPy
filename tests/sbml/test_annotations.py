from __future__ import annotations

import libsbml
import pytest

from mxlpy import Annotation, Model, Parameter, Variable, fns
from mxlpy.paths import default_tmp_dir
from mxlpy.sbml import read, write

TMP_DIR = default_tmp_dir(None, remove_old_cache=False)

UNIPROT = "https://identifiers.org/uniprot:P00533"
CHEBI = "https://identifiers.org/chebi:16796"
ECCODE = "https://identifiers.org/ec-code:1.1.1.1"
GO = "https://identifiers.org/go:0005737"
SBO = "https://identifiers.org/sbo:0000176"
BIOMODELS = "https://identifiers.org/biomodels.db:BIOMD0000000048"


def test_normalize_single_annotation() -> None:
    m = Model().add_variable("x", 1.0, annotations=Annotation(UNIPROT))
    assert m.get_raw_variables()["x"].annotations == [Annotation(UNIPROT)]


def test_normalize_list_annotation() -> None:
    anns = [Annotation(UNIPROT), Annotation(CHEBI, predicate="hasPart")]
    m = Model().add_variable("x", 1.0, annotations=anns)
    assert m.get_raw_variables()["x"].annotations == anns


def test_default_annotations_empty() -> None:
    m = Model().add_variable("x", 1.0)
    assert m.get_raw_variables()["x"].annotations == []


def test_annotation_default_predicate() -> None:
    assert Annotation(UNIPROT).predicate == "is"


def test_add_variable_annotations() -> None:
    m = Model().add_variable("x", 1.0, annotations=Annotation(UNIPROT))
    assert m.get_raw_variables()["x"].annotations == [Annotation(UNIPROT)]


def test_add_parameter_annotations() -> None:
    m = Model().add_parameter("k1", 1.0, annotations=Annotation(ECCODE))
    assert m.get_raw_parameters()["k1"].annotations == [Annotation(ECCODE)]


def test_add_derived_annotations() -> None:
    m = (
        Model()
        .add_parameter("k1", 1.0)
        .add_derived("d1", fns.constant, args=["k1"], annotations=Annotation(GO))
    )
    assert m.get_raw_derived()["d1"].annotations == [Annotation(GO)]


def test_add_reaction_annotations() -> None:
    m = (
        Model()
        .add_parameter("k1", 1.0)
        .add_variable("x", 1.0)
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={"x": -1},
            annotations=Annotation(SBO),
        )
    )
    assert m.get_raw_reactions()["v1"].annotations == [Annotation(SBO)]


def test_update_variable_annotations() -> None:
    m = Model().add_variable("x", 1.0, annotations=Annotation(UNIPROT))
    m.update_variable("x", 2.0, annotations=Annotation(CHEBI))
    assert m.get_raw_variables()["x"].annotations == [Annotation(CHEBI)]


def test_update_variable_keeps_annotations_when_omitted() -> None:
    m = Model().add_variable("x", 1.0, annotations=Annotation(UNIPROT))
    m.update_variable("x", 2.0)
    assert m.get_raw_variables()["x"].annotations == [Annotation(UNIPROT)]


def test_update_parameter_annotations() -> None:
    m = Model().add_parameter("k1", 1.0)
    m.update_parameter("k1", 2.0, annotations=Annotation(ECCODE))
    assert m.get_raw_parameters()["k1"].annotations == [Annotation(ECCODE)]


def test_update_derived_annotations() -> None:
    m = Model().add_parameter("k1", 1.0).add_derived("d1", fns.constant, args=["k1"])
    m.update_derived("d1", annotations=Annotation(GO))
    assert m.get_raw_derived()["d1"].annotations == [Annotation(GO)]


def test_update_reaction_annotations() -> None:
    m = (
        Model()
        .add_parameter("k1", 1.0)
        .add_variable("x", 1.0)
        .add_reaction("v1", fns.constant, args=["k1"], stoichiometry={"x": -1})
    )
    m.update_reaction("v1", annotations=Annotation(SBO))
    assert m.get_raw_reactions()["v1"].annotations == [Annotation(SBO)]


def test_add_variables_propagates_annotations() -> None:
    m = Model().add_variables({"x": Variable(1.0, annotations=[Annotation(UNIPROT)])})
    assert m.get_raw_variables()["x"].annotations == [Annotation(UNIPROT)]


def test_add_parameters_propagates_annotations() -> None:
    m = Model().add_parameters({"k1": Parameter(1.0, annotations=[Annotation(ECCODE)])})
    assert m.get_raw_parameters()["k1"].annotations == [Annotation(ECCODE)]


def test_annotate_model_single() -> None:
    m = Model().annotate_model(Annotation(BIOMODELS, predicate="isDerivedFrom"))
    assert m.get_annotations() == [Annotation(BIOMODELS, predicate="isDerivedFrom")]


def test_annotate_model_extends() -> None:
    m = (
        Model()
        .annotate_model(Annotation(BIOMODELS, predicate="isDerivedFrom"))
        .annotate_model(Annotation(GO, predicate="is"))
    )
    assert m.get_annotations() == [
        Annotation(BIOMODELS, predicate="isDerivedFrom"),
        Annotation(GO, predicate="is"),
    ]


def test_roundtrip_variable_annotation() -> None:
    m1 = Model().add_variable("x", 1.0, annotations=Annotation(UNIPROT))
    m2 = read(write(m1, TMP_DIR / "ann_variable.xml"))
    assert m2.get_raw_variables()["x"].annotations == [Annotation(UNIPROT)]


def test_roundtrip_variable_multiple_annotations() -> None:
    anns = [Annotation(UNIPROT), Annotation(CHEBI, predicate="hasPart")]
    m1 = Model().add_variable("x", 1.0, annotations=anns)
    m2 = read(write(m1, TMP_DIR / "ann_variable_multi.xml"))
    assert m2.get_raw_variables()["x"].annotations == anns


def test_roundtrip_parameter_annotation() -> None:
    m1 = Model().add_parameter(
        "k1", 1.0, annotations=Annotation(ECCODE, predicate="isVersionOf")
    )
    m2 = read(write(m1, TMP_DIR / "ann_parameter.xml"))
    assert m2.get_raw_parameters()["k1"].annotations == [
        Annotation(ECCODE, predicate="isVersionOf")
    ]


def test_roundtrip_derived_annotation() -> None:
    m1 = (
        Model()
        .add_parameter("k1", 1.0)
        .add_derived(
            "d1",
            fns.constant,
            args=["k1"],
            annotations=Annotation(GO, predicate="occursIn"),
        )
    )
    m2 = read(write(m1, TMP_DIR / "ann_derived.xml"))
    assert m2.get_raw_derived()["d1"].annotations == [
        Annotation(GO, predicate="occursIn")
    ]


def test_roundtrip_reaction_annotation() -> None:
    m1 = (
        Model()
        .add_parameter("k1", 1.0)
        .add_variable("x", 1.0)
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={"x": -1},
            annotations=Annotation(SBO),
        )
    )
    m2 = read(write(m1, TMP_DIR / "ann_reaction.xml"))
    assert m2.get_raw_reactions()["v1"].annotations == [Annotation(SBO)]


def test_roundtrip_model_annotation() -> None:
    m1 = Model().annotate_model(Annotation(BIOMODELS, predicate="isDerivedFrom"))
    m2 = read(write(m1, TMP_DIR / "ann_model.xml"))
    assert m2.get_annotations() == [Annotation(BIOMODELS, predicate="isDerivedFrom")]


def test_export_rejects_bqmodel_predicate_on_component() -> None:
    m = Model().add_variable(
        "x", 1.0, annotations=Annotation(UNIPROT, predicate="isDerivedFrom")
    )
    with pytest.raises(ValueError, match="bqbiol"):
        write(m, TMP_DIR / "ann_bad_component.xml")


def test_export_rejects_bqbiol_predicate_on_model() -> None:
    m = Model().annotate_model(Annotation(UNIPROT, predicate="hasTaxon"))
    with pytest.raises(ValueError, match="bqmodel"):
        write(m, TMP_DIR / "ann_bad_model.xml")


def test_export_without_annotations_has_no_metaid() -> None:
    m = Model().add_variable("x", 1.0)
    path = write(m, TMP_DIR / "ann_none.xml")
    doc = libsbml.readSBMLFromFile(str(path))
    species = doc.getModel().getSpecies(0)
    assert not species.isSetMetaId()
