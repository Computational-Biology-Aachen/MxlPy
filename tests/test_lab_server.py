# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import json
import shutil
import time
from pathlib import Path

import pytest

from mxlpy import lab
from mxlpy.lab import views
from mxlpy.lab._core import STEPS
from tests.test_lab import FAST, _data, _decay

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from mxlpy.cli import main
from mxlpy.lab.server import LabServer, create_app

TOKEN = "secret-token"  # noqa: S105
# The module fixture runs a whole lab (~30 s alone, over 60 s on a loaded
# CI runner), and its setup counts towards the first test's timeout
pytestmark = pytest.mark.timeout(300)


def _wait_for(condition, timeout: float = 60.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            msg = "condition not reached"
            raise TimeoutError(msg)
        time.sleep(0.05)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run through all steps, with a failing fit (diagnosis runs)."""
    run_dir = tmp_path_factory.mktemp("lab") / "run"
    data = {"a": _data(_decay(), 1), "b": _data(_decay(), 2, x0=3.0)}
    lab.FittingLab(_decay(), data, run_dir, **FAST).run(auto=True)
    return run_dir


def _copy(src: Path, tmp_path: Path) -> lab.FittingLab:
    dst = tmp_path / "run"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("run.lock"))
    return lab.FittingLab.resume(dst)


def _client(run: lab.FittingLab, *, token: str | None = TOKEN) -> TestClient:
    app = create_app(run, token=token, hosts={"testserver", "127.0.0.1"})
    app.state.lab.tail()
    client = TestClient(app)
    if token is not None:
        client.headers["authorization"] = f"Bearer {token}"
    return client


###############################################################################
# Options, progress
###############################################################################


def test_options_of_all_steps(finished: Path, tmp_path: Path) -> None:
    run = _copy(finished, tmp_path)
    with pytest.raises(TypeError, match="true or false"):
        run.decide("identifiability", refit="yes")
    assert run["identifiability"].options["fix"].kind == "subset"
    assert run["diagnose"].options["model"].choices == ("original",)
    assert run["setup"].options["integrator"].kind == "choice"
    assert json.loads((run.store.step_dir("fit") / "result.json").read_text())[
        "options"
    ]["next"]["choices"] == ["identifiability", "diagnose"]
    run.reopen("fit")
    assert run.decision("fit") is None
    overview = views.overview(run)
    assert overview["open_gate"] == "fit"
    assert not overview["finished"]


def test_fit_groups() -> None:
    assert views._group("start/3") == "start"
    assert views._group("a/start/0") == "a/start"
    assert views._group("refit-4711-2") == "refit"
    assert views._group("hidden1") == "hidden1"
    assert views._group(None) == "fit"


def test_progress_aggregates_events(finished: Path) -> None:
    store = lab.FittingLab.resume(finished).store
    progress = views.Progress()
    progress.feed(store.read_events()[0])
    fit = progress.steps["fit"].to_dict()
    # joint starts and the separate fits of the two data sets
    assert set(fit["waterfalls"]) == {"start", "a/start", "b/start"}
    assert len(fit["waterfalls"]["start"]) == FAST["n_starts"]
    trace = fit["traces"]["start"]
    assert [b for _, b in trace] == sorted((b for _, b in trace), reverse=True)
    assert fit["finished"] is not None
    assert progress.state()["finished"]
    assert progress.state()["running"] is None


###############################################################################
# Views
###############################################################################


def test_step_views(finished: Path) -> None:
    run = lab.FittingLab.resume(finished)
    ids = {s: [c["id"] for c in views.step_view(run, s)["charts"]] for s in STEPS}
    assert ids["setup"] == ["data/a", "data/b"]
    assert ids["parameters"] == ["importance"]
    assert ids["fit"] == ["fit/a", "fit/b", "waterfall"]
    assert ids["diagnose"][0] == "hidden"
    assert any(i.startswith("correction/") for i in ids["diagnose"])
    assert ids["identifiability"] == ["profile/k1", "profile/k2"]
    assert len(ids["uncertainty"]) == 4
    fit = views.step_view(run, "fit")
    # data as points, predictions as lines, colour per observable
    lines = fit["charts"][0]["series"]
    assert [(s["type"], s["color"]) for s in lines] == [
        ("points", 0),
        ("line", 0),
        ("points", 1),
        ("line", 1),
    ]
    assert fit["result"]["options"]["next"]["kind"] == "choice"
    assert {t["id"] for t in fit["tables"]} >= {"estimates", "gof", "residuals"}
    profile = views.step_view(run, "identifiability")["charts"][0]
    assert profile["hlines"][0]["label"] == "threshold"
    band = views.step_view(run, "uncertainty")["charts"][0]["series"][0]
    assert band["type"] == "band"
    assert len(band["lower"]) == len(band["x"]) == FAST["prediction_times"]
    # everything is plain JSON
    json.dumps([views.step_view(run, s) for s in STEPS], allow_nan=False)
    overview = views.overview(run)
    assert [s["status"] for s in overview["steps"]] == [
        "done",
        "done",
        "failed",
        "failed",
        "passed",
        "done",
    ]
    assert all(s["gate"] == "auto" for s in overview["steps"])
    assert overview["finished"]
    assert overview["open_gate"] is None


def test_views_of_an_empty_run(tmp_path: Path) -> None:
    run = lab.FittingLab(_decay(), _data(_decay(), 3), tmp_path / "run", **FAST)
    assert views.step_view(run, "fit")["status"] == "pending"
    assert views.overview(run)["steps"][0]["status"] == "pending"


def test_live_fit(finished: Path) -> None:
    run = lab.FittingLab.resume(finished)
    current = run._state().current("fit")
    charts = views.live_fit(current, dict(current.estimates))
    assert charts is not None
    assert [c["id"] for c in charts] == ["live/a", "live/b"]
    assert views.live_fit(current, {"k1": 1.0}) is None


def test_live_fit_of_a_running_step(finished: Path, tmp_path: Path) -> None:
    run = _copy(finished, tmp_path)
    state = create_app(run).state.lab
    state.tail()
    estimates = dict(run["fit"].summary["estimates"])
    run.store.append_event({"source": "lab", "kind": "step_started", "step": "fit"})
    run.store.append_event(
        {
            "step": "fit",
            "start": "start/0",
            "kind": "evaluation",
            "n_evaluations": 1,
            "best": 5.0,
            "improved": True,
            "parameters": estimates,
        }
    )
    state.tail()
    live = state.live("fit")
    assert live["group"] == "start"
    assert live["best"] == 5.0
    assert [c["id"] for c in live["charts"]] == ["live/a", "live/b"]
    assert state.live("fit") is live  # cached until the best changes
    assert state.progress.state()["running"]["step"] == "fit"
    run.store.append_event({"source": "lab", "kind": "step_finished", "step": "fit"})
    state.tail()
    assert state.live("fit") is None


###############################################################################
# Server
###############################################################################


def test_api_needs_token_and_local_host(finished: Path) -> None:
    run = lab.FittingLab.resume(finished)
    app = create_app(run, token=TOKEN, hosts={"testserver"})
    client = TestClient(app)
    assert client.get("/api/overview").status_code == 401
    bad = client.get("/api/overview", headers={"authorization": "Bearer wrong"})
    assert bad.status_code == 401
    # the token in the address becomes a cookie
    res = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/"
    cookie = res.headers["set-cookie"].lower()
    assert "samesite=strict" in cookie
    # one cookie per port: servers of several runs don't log each other out
    assert cookie.startswith("mxlpy_lab_token_80=")
    assert client.get("/api/overview").status_code == 200
    # DNS rebinding: a foreign Host or Origin is refused
    assert client.get("/api/overview", headers={"host": "evil.com"}).status_code == 403
    cross = client.post("/api/pause", headers={"origin": "http://evil.com"})
    assert cross.status_code == 403
    assert not run.paused


def test_api(finished: Path, tmp_path: Path) -> None:
    run = _copy(finished, tmp_path)
    client = _client(run)
    overview = client.get("/api/overview").json()
    assert [s["step"] for s in overview["steps"]] == list(STEPS)
    assert overview["progress"]["finished"]
    assert not overview["driving"]

    step = client.get("/api/steps/2").json()
    assert step["step"] == "fit"
    assert step["charts"]
    assert client.get("/api/steps/nope").status_code == 404

    res = client.post("/api/steps/fit/decision", json={"values": {"next": "nowhere"}})
    assert res.status_code == 400
    assert "one of" in res.json()["error"]
    res = client.post(
        "/api/steps/fit/decision", json={"values": {"next": "identifiability"}}
    )
    assert res.status_code == 200
    assert run.decisions()["fit"]["source"] == "user"
    assert client.delete("/api/steps/fit/decision").status_code == 200
    assert "fit" not in run.decisions()

    # this app only watches: another process runs the steps
    assert client.post("/api/run").status_code == 409
    assert not client.get("/api/overview").json()["can_run"]

    assert client.post("/api/pause").status_code == 200
    assert run.paused
    assert client.get("/api/overview").json()["paused"]
    client.post("/api/unpause")
    assert not run.paused

    progress = client.get("/api/progress/fit").json()
    assert progress["waterfalls"]["start"]
    # live fits only while a step runs
    assert client.get("/api/live/fit").json() is None

    report = client.get("/api/report")
    assert report.status_code == 200
    assert "Fitting lab" in report.text


def test_static_files(finished: Path, tmp_path: Path) -> None:
    run = lab.FittingLab.resume(finished)
    static = tmp_path / "static"
    app = create_app(run, token=None, static=static)
    client = TestClient(app)
    assert client.get("/").status_code == 503  # not built
    (static / "_app").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>lab</title>")
    (static / "_app" / "a.js").write_text("export {}")
    assert client.get("/").text.startswith("<!doctype html>")
    js = client.get("/_app/a.js")
    assert js.headers["content-type"].startswith("text/javascript")
    # client-side routes get the app, unknown API routes don't
    assert client.get("/steps/fit").text.startswith("<!doctype html>")
    assert client.get("/api/nope").status_code == 404
    assert client.get("/..%2F..%2Fetc%2Fpasswd").text.startswith("<!doctype html>")


def test_server_drives_the_run(tmp_path: Path) -> None:
    run = lab.FittingLab(_decay(), _data(_decay(), 3), tmp_path / "run", **FAST)
    server = LabServer(run, port=0, token=TOKEN)
    server.start()
    try:
        assert server.url.endswith(f"/?token={TOKEN}")
        _wait_for(lambda: server.state.progress.gate == "setup")
        state = server.state
        # a decision (from the UI or Python) continues the run
        run.decide("setup")
        _wait_for(lambda: state.progress.gate == "parameters")
        assert "parameters" in run.results
        assert run.store.is_driven()
        overview = views.overview(run)
        assert overview["steps"][0]["gate"] == "user"
    finally:
        server.stop()
    assert not run.store.is_driven()


def test_cli(finished: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lab", "status", str(finished)]) == 0
    assert "identifiability" in capsys.readouterr().out
    assert main(["lab", "status", str(finished / "missing")]) == 1
