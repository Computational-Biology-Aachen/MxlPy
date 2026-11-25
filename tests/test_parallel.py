# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from pathlib import Path

import numpy as np
import pandas as pd

from mxlpy import KineticModelBuilder, fns
from mxlpy.parallel import Cache, content_hash, parallelise


def test_parallelise() -> None:
    # FIXME: implement this
    assert True


def test_parallelise_keyless() -> None:
    # FIXME: implement this
    assert True


def test_content_hash_is_stable_and_order_independent() -> None:

    a = {"x": 1.0, "y": [1, 2], "df": pd.DataFrame({"a": [1.0, 2.0]})}
    b = {"df": pd.DataFrame({"a": [1.0, 2.0]}), "y": [1, 2], "x": 1.0}
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash({**a, "x": 1.0000001})
    assert content_hash(np.arange(3.0)) != content_hash(np.arange(3))
    assert content_hash(1) != content_hash(1.0)
    assert content_hash((1, 2)) != content_hash((2, 1))


def test_content_hash_of_models_follows_the_model() -> None:

    def build(fn):  # noqa: ANN001, ANN202
        return (
            KineticModelBuilder()
            .add_parameters({"k": 1.0})
            .add_variables({"x": 1.0})
            .add_reaction("v", fn, args=["x", "k"], stoichiometry={"x": -1})
        )

    m = build(fns.mass_action_1s)
    assert content_hash(m) == content_hash(build(fns.mass_action_1s))
    assert content_hash(m) != content_hash(build(fns.add))
    assert content_hash(m) != content_hash(
        build(fns.mass_action_1s).update_parameter("k", 2.0)
    )


def test_cache_content_hash_recomputes_changed_inputs(tmp_path: Path) -> None:
    calls: list[float] = []

    def fn(x: float) -> float:
        calls.append(x)
        return 2 * x

    cache = Cache(tmp_path, content_hash=True, salt="model-v1")
    assert parallelise(fn, [("a", 1.0)], cache=cache, parallel=False) == [("a", 2.0)]
    assert parallelise(fn, [("a", 1.0)], cache=cache, parallel=False) == [("a", 2.0)]
    assert calls == [1.0]
    # same key, other input or other salt: recomputed instead of loaded
    assert parallelise(fn, [("a", 3.0)], cache=cache, parallel=False) == [("a", 6.0)]
    other = Cache(tmp_path, content_hash=True, salt="model-v2")
    assert parallelise(fn, [("a", 1.0)], cache=other, parallel=False) == [("a", 2.0)]
    assert calls == [1.0, 3.0, 1.0]
    # without content_hash the key alone decides, as before
    plain = Cache(tmp_path / "plain")
    parallelise(fn, [("a", 1.0)], cache=plain, parallel=False)
    assert parallelise(fn, [("a", 5.0)], cache=plain, parallel=False) == [("a", 2.0)]


def test_content_hash_of_object_columns_is_stable_across_processes() -> None:
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = (
        "import pandas as pd; from mxlpy.parallel import content_hash; "
        "print(content_hash(pd.DataFrame({'a': [1.0, 2.0], 'b': [True, 'x']})))"
    )
    hashes = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(hashes) == 1
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [True, "x"]})
    assert content_hash(df) == hashes.pop()
    assert content_hash(df) != content_hash(df.astype({"a": "float32"}))
