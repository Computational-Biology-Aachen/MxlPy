# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Backend-native surrogate save/load (`surrogate.save` / `mxlpy.surrogates.load`)."""

from __future__ import annotations

import json
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy import polynomial

from mxlpy import surrogates
from mxlpy.surrogates import equinox as es
from mxlpy.surrogates import keras as ks
from mxlpy.surrogates import poly, qss
from mxlpy.surrogates import torch as ts
from mxlpy.surrogates._io import META_FILE
from mxlpy.surrogates.abstract import MockSurrogate
from mxlpy.types import Derived, SerializationError
from mxlpy.units import mmol_s

if TYPE_CHECKING:
    from pathlib import Path

requires_torch = pytest.mark.skipif(find_spec("torch") is None, reason="needs torch")
requires_keras = pytest.mark.skipif(find_spec("keras") is None, reason="needs keras")
requires_equinox = pytest.mark.skipif(
    find_spec("equinox") is None, reason="needs equinox"
)


def _double(k: float) -> float:
    return 2.0 * k


def _qss_fn(x: float, y: float) -> tuple[float, float]:
    return x + y, x * y


KINETIC_META: dict[str, Any] = {
    "args": ["x", "y"],
    "outputs": ["v1"],
    "stoichiometries": {"v1": {"x": -1.0, "z": Derived(fn=_double, args=["k"])}},
    "units": {"v1": mmol_s},
}
ODE_META: dict[str, Any] = {
    "args": ["x", "y"],
    "outputs": ["dx"],
    "targets": {"dx": ["x"]},
}
INPUTS = {"x": 0.3, "y": -1.2}


def _assert_kinetic_meta(loaded: Any) -> None:
    assert loaded.args == KINETIC_META["args"]
    assert loaded.outputs == KINETIC_META["outputs"]
    assert loaded.units == KINETIC_META["units"]
    stoich = loaded.stoichiometries["v1"]
    assert stoich["x"] == -1.0
    assert isinstance(stoich["z"], Derived)
    assert stoich["z"].args == ["k"]
    assert stoich["z"].fn(3.0) == pytest.approx(6.0)


def _assert_same_prediction(a: Any, b: Any) -> None:
    pa, pb = a.predict(dict(INPUTS)), b.predict(dict(INPUTS))
    assert pa.keys() == pb.keys()
    for k in pa:
        assert float(pb[k]) == pytest.approx(float(pa[k]), rel=1e-6)


###############################################################################
# torch
###############################################################################


def _torch_net(seed: int) -> Any:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    )


@requires_torch
def test_torch_kinetic_roundtrip(tmp_path: Path) -> None:
    srg = ts.Surrogate(model=_torch_net(0), **KINETIC_META)
    srg.save(tmp_path / "srg")
    assert (tmp_path / "srg" / "model.pt").is_file()

    loaded = surrogates.load(tmp_path / "srg", model=_torch_net(1))
    assert isinstance(loaded, ts.Surrogate)
    _assert_kinetic_meta(loaded)
    _assert_same_prediction(srg, loaded)


@requires_torch
def test_torch_ode_roundtrip(tmp_path: Path) -> None:
    srg = ts.OdeSurrogate(model=_torch_net(0), **ODE_META)
    srg.save(tmp_path)
    loaded = ts.load(tmp_path, model=_torch_net(1))
    assert isinstance(loaded, ts.OdeSurrogate)
    assert loaded.targets == ODE_META["targets"]
    _assert_same_prediction(srg, loaded)


@requires_torch
def test_torch_custom_module_roundtrip(tmp_path: Path) -> None:
    """Non-dense modules (not representable in .mxl.json) persist natively."""
    torch = pytest.importorskip("torch", exc_type=ImportError)

    def net(seed: int) -> Any:
        torch.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.LayerNorm(3), torch.nn.Linear(3, 1)
        )

    srg = ts.Surrogate(model=net(0), **KINETIC_META)
    with pytest.raises(SerializationError):
        srg.to_mxl_json()
    srg.save(tmp_path)
    loaded = surrogates.load(tmp_path, model=net(1))
    _assert_same_prediction(srg, loaded)


@requires_torch
def test_torch_requires_template(tmp_path: Path) -> None:
    ts.Surrogate(model=_torch_net(0), **KINETIC_META).save(tmp_path)
    with pytest.raises(ValueError, match="model="):
        surrogates.load(tmp_path)


@requires_torch
def test_torch_wrong_template(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    ts.Surrogate(model=_torch_net(0), **KINETIC_META).save(tmp_path)
    with pytest.raises(RuntimeError):
        ts.load(tmp_path, model=torch.nn.Sequential(torch.nn.Linear(2, 1)))


###############################################################################
# keras
###############################################################################


def _keras_net() -> Any:
    keras = pytest.importorskip("keras", exc_type=ImportError)
    keras.utils.set_random_seed(0)
    return keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(4, activation="tanh"),
            keras.layers.Dense(1),
        ]
    )


@requires_keras
def test_keras_kinetic_roundtrip(tmp_path: Path) -> None:
    srg = ks.Surrogate(model=_keras_net(), **KINETIC_META)
    srg.save(tmp_path)
    assert (tmp_path / "model.keras").is_file()

    loaded = surrogates.load(tmp_path)
    assert isinstance(loaded, ks.Surrogate)
    _assert_kinetic_meta(loaded)
    _assert_same_prediction(srg, loaded)


@requires_keras
def test_keras_ode_roundtrip(tmp_path: Path) -> None:
    srg = ks.OdeSurrogate(model=_keras_net(), **ODE_META)
    srg.save(tmp_path)
    loaded = ks.load(tmp_path)
    assert isinstance(loaded, ks.OdeSurrogate)
    assert loaded.targets == ODE_META["targets"]
    _assert_same_prediction(srg, loaded)


@requires_keras
def test_keras_rejects_template(tmp_path: Path) -> None:
    ks.Surrogate(model=_keras_net(), **KINETIC_META).save(tmp_path)
    with pytest.raises(ValueError, match="model="):
        surrogates.load(tmp_path, model=_keras_net())


###############################################################################
# equinox
###############################################################################


def _eqx_net(seed: int) -> Any:
    eqx = pytest.importorskip("equinox", exc_type=ImportError)
    jax = pytest.importorskip("jax", exc_type=ImportError)
    return eqx.nn.MLP(
        in_size=2, out_size=1, width_size=4, depth=1, key=jax.random.PRNGKey(seed)
    )


@requires_equinox
def test_equinox_kinetic_roundtrip(tmp_path: Path) -> None:
    srg = es.Surrogate(model=_eqx_net(0), **KINETIC_META)
    srg.save(tmp_path)
    assert (tmp_path / "model.eqx").is_file()

    loaded = surrogates.load(tmp_path, model=_eqx_net(1))
    assert isinstance(loaded, es.Surrogate)
    _assert_kinetic_meta(loaded)
    _assert_same_prediction(srg, loaded)


@requires_equinox
def test_equinox_ode_roundtrip(tmp_path: Path) -> None:
    srg = es.OdeSurrogate(model=_eqx_net(0), **ODE_META)
    srg.save(tmp_path)
    loaded = es.load(tmp_path, model=_eqx_net(1))
    assert isinstance(loaded, es.OdeSurrogate)
    assert loaded.targets == ODE_META["targets"]
    _assert_same_prediction(srg, loaded)


@requires_equinox
def test_equinox_requires_template(tmp_path: Path) -> None:
    es.Surrogate(model=_eqx_net(0), **KINETIC_META).save(tmp_path)
    with pytest.raises(ValueError, match="model="):
        surrogates.load(tmp_path)


###############################################################################
# poly / qss
###############################################################################


@pytest.mark.parametrize(
    "series",
    [
        polynomial.Polynomial,
        polynomial.Chebyshev,
        polynomial.Legendre,
        polynomial.Laguerre,
        polynomial.Hermite,
        polynomial.HermiteE,
    ],
)
def test_poly_roundtrip(tmp_path: Path, series: Any) -> None:
    x = np.linspace(0, 2, 20)
    model = series.fit(x, np.sin(x), 3)
    srg = poly.Surrogate(model=model, **{**KINETIC_META, "args": ["x"]})
    srg.save(tmp_path)

    loaded = surrogates.load(tmp_path)
    assert isinstance(loaded, poly.Surrogate)
    assert type(loaded.model) is series
    np.testing.assert_allclose(np.asarray(loaded.model.coef, dtype=float), model.coef)
    np.testing.assert_allclose(loaded.model.domain, model.domain)
    np.testing.assert_allclose(loaded.model.window, model.window)
    np.testing.assert_allclose(loaded.predict_raw(x), srg.predict_raw(x))


def test_poly_unknown_series(tmp_path: Path) -> None:
    srg = poly.Surrogate(model=polynomial.Polynomial([1.0, 2.0]), **KINETIC_META)
    srg.save(tmp_path)
    state = json.loads((tmp_path / "model.json").read_text())
    (tmp_path / "model.json").write_text(json.dumps(state | {"kind": "Spline"}))
    with pytest.raises(SerializationError, match="Spline"):
        poly.load(tmp_path)


def test_qss_roundtrip(tmp_path: Path) -> None:
    srg = qss.Surrogate(model=_qss_fn, args=["x", "y"], outputs=["a", "b"])
    srg.save(tmp_path)
    loaded = surrogates.load(tmp_path)
    assert isinstance(loaded, qss.Surrogate)
    assert loaded.predict(dict(INPUTS)) == srg.predict(dict(INPUTS))


def test_qss_rejects_template(tmp_path: Path) -> None:
    qss.Surrogate(model=_qss_fn, args=["x", "y"], outputs=["a", "b"]).save(tmp_path)
    with pytest.raises(ValueError, match="model="):
        surrogates.load(tmp_path, model=_qss_fn)


###############################################################################
# shared behaviour
###############################################################################


def test_overwrite_existing(tmp_path: Path) -> None:
    poly.Surrogate(model=polynomial.Polynomial([1.0]), args=["x"], outputs=["a"]).save(
        tmp_path
    )
    qss.Surrogate(model=_qss_fn, args=["x", "y"], outputs=["a", "b"]).save(tmp_path)
    loaded = surrogates.load(tmp_path)
    assert isinstance(loaded, qss.Surrogate)


def test_unsupported_format_version(tmp_path: Path) -> None:
    poly.Surrogate(model=polynomial.Polynomial([1.0]), args=["x"], outputs=["a"]).save(
        tmp_path
    )
    meta = json.loads((tmp_path / META_FILE).read_text())
    (tmp_path / META_FILE).write_text(json.dumps(meta | {"format_version": 99}))
    with pytest.raises(SerializationError, match="format_version"):
        surrogates.load(tmp_path)


def test_backend_mismatch(tmp_path: Path) -> None:
    qss.Surrogate(model=_qss_fn, args=["x", "y"], outputs=["a", "b"]).save(tmp_path)
    with pytest.raises(SerializationError, match="'qss'"):
        poly.load(tmp_path)


def test_bad_unit(tmp_path: Path) -> None:
    srg = qss.Surrogate(
        model=_qss_fn, args=["x", "y"], outputs=["a", "b"], units={"a": mmol_s + 1}
    )
    with pytest.raises(SerializationError, match="unit of 'a'"):
        srg.save(tmp_path)


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=META_FILE):
        surrogates.load(tmp_path / "nope")


def test_save_not_implemented(tmp_path: Path) -> None:
    srg = MockSurrogate(fn=_qss_fn, args=["x", "y"], outputs=["a", "b"])
    with pytest.raises(NotImplementedError, match="MockSurrogate"):
        srg.save(tmp_path)
