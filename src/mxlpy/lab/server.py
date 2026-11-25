# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Local web UI of the fitting lab (`mxlpy[lab]`).

A Starlette app serving the prebuilt frontend (`static/`), a JSON API over
the run and a Server-Sent Events stream of its progress. The run directory
stays the source of truth: decisions and pauses go through the same files
as `FittingLab.decide` and `FittingLab.pause`, so Python, the CLI and the
browser can steer one run.

API (all under `/api`, JSON):

    GET    overview                       steps, gates, pause, run state
    GET    steps/{step}                   result, options, decision, charts
    POST   steps/{step}/decision          {"values": {...}} decide the gate
    DELETE steps/{step}/decision          reopen the gate
    GET    progress/{step}                loss traces and waterfalls
    GET    live/{step}                    data with the current best fit
    POST   pause | unpause                pause before the next step
    POST   run                            (re)start driving: {"auto": bool}
    GET    report                         the HTML report (`FittingLab.report`)
    GET    events                         Server-Sent Events: progress events

Requests need the access token (`?token=` once, then a cookie, or an
`Authorization: Bearer` header) and a local `Host`, which protects the run
against other web pages in the same browser.
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import json
import logging
import mimetypes
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlsplit

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from mxlpy.lab._lab import _key
from mxlpy.lab.store import RunLockedError
from mxlpy.lab.views import Progress, live_fit, overview, step_view

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping

    from mxlpy.lab._core import Current
    from mxlpy.lab._lab import FittingLab

__all__ = ["STATIC", "LabServer", "create_app"]

_LOGGER = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TAIL = 0.25
_LOG_SIZE = 2000
_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _media_type(path: Path) -> str:
    return _TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "text/plain"


type _Scope = MutableMapping[str, Any]
type _Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
type _Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
type _ASGIApp = Callable[[_Scope, _Receive, _Send], Awaitable[None]]


class _Guard:
    """ASGI middleware: local `Host` (DNS rebinding) and the access token."""

    def __init__(
        self, app: _ASGIApp, *, token: str | None, hosts: set[str] | None
    ) -> None:
        self.app = app
        self.token = token
        self.hosts = hosts

    @staticmethod
    def cookie(request: Request) -> str:
        """Cookie of the token, per port (cookies are shared across ports)."""
        return f"mxlpy_lab_token_{request.url.port or 80}"

    def _host_ok(self, request: Request) -> bool:
        if self.hosts is None:
            return True
        host = request.headers.get("host", "")
        name = urlsplit(f"//{host}").hostname or ""
        if name not in self.hosts:
            return False
        origin = request.headers.get("origin")
        return origin is None or (urlsplit(origin).hostname or "") in self.hosts

    def _token_ok(self, request: Request) -> bool:
        if self.token is None:
            return True
        given = request.cookies.get(self.cookie(request)) or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            given = auth[7:]
        return hmac.compare_digest(given.encode(), self.token.encode())

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if not self._host_ok(request):
            await PlainTextResponse("Forbidden host", 403)(scope, receive, send)
            return
        query = request.query_params.get("token")
        if (
            self.token is not None
            and query is not None
            and hmac.compare_digest(query.encode(), self.token.encode())
        ):
            # Remember the token and drop it from the address bar
            response = Response(status_code=303, headers={"location": request.url.path})
            response.set_cookie(
                self.cookie(request), self.token, httponly=True, samesite="strict"
            )
            await response(scope, receive, send)
            return
        if scope["path"].startswith("/api/") and not self._token_ok(request):
            await JSONResponse(
                {"error": "Open the address with the token printed by the server"},
                401,
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _State:
    """What the routes share: the lab, its progress and the driver."""

    def __init__(self, lab: FittingLab, *, watch: bool, auto: bool) -> None:
        self.lab = lab
        self.watch = watch
        self.auto = auto
        self.progress = Progress()
        self.log: collections.deque[tuple[int, dict[str, Any]]] = collections.deque(
            maxlen=_LOG_SIZE
        )
        self.seq = 0
        self.offset = 0
        self.lock = threading.Lock()
        self._tail_lock = threading.Lock()
        self._live_lock = threading.Lock()
        self.report_lock = threading.Lock()
        self.stop = threading.Event()
        self.driver: threading.Thread | None = None
        self.driver_error: str | None = None
        self._live: tuple[Any, Any] = (None, None)
        self._current: tuple[Any, Current | None] = (None, None)

    # Events ##################################################################

    def tail(self) -> None:
        """Read new events: progress, log for the SSE clients, fresh results."""
        with self._tail_lock:
            events, self.offset = self.lab.store.read_events(self.offset)
            if not events:
                return
            with self.lock:
                self.progress.feed(events)
                for ev in events:
                    self.seq += 1
                    light = {k: v for k, v in ev.items() if k != "parameters"}
                    self.log.append((self.seq, light))
        if any(e.get("source") == "lab" for e in events):
            # Results another process wrote (cheap: checks modification times)
            self.lab._refresh()  # noqa: SLF001

    def events_after(self, seq: int) -> list[tuple[int, dict[str, Any]]]:
        with self.lock:
            return [(s, e) for s, e in self.log if s > seq]

    def _tail_loop(self) -> None:
        while not self.stop.wait(_TAIL):
            try:
                self.tail()
            except Exception:
                _LOGGER.debug("Reading events failed", exc_info=True)

    # Driver ##################################################################

    def _drive(self) -> None:
        """Run the steps; after the end or an error, again once a gate changes."""
        store = self.lab.store
        while not self.stop.is_set():
            snapshot = store.decisions()
            try:
                self.lab.run(auto=self.auto, wait=True, stop=self.stop)
            except RunLockedError as e:
                self.driver_error = str(e)
                return
            except Exception as e:
                self.driver_error = f"{type(e).__name__}: {e}"
                _LOGGER.warning("The lab stopped with an error", exc_info=True)
            # Reopened or changed decisions make later steps outdated
            while snapshot == store.decisions() and not self.stop.wait(0.5):
                pass
            self.driver_error = None

    def start_driver(self, *, auto: bool | None = None) -> bool:
        """Run the steps in the background, False if already running."""
        if self.driver is not None and self.driver.is_alive():
            return False
        if auto is not None:
            self.auto = auto
        self.driver_error = None
        self.driver = threading.Thread(target=self._drive, daemon=True)
        self.driver.start()
        return True

    def start(self) -> None:
        self.tail()
        threading.Thread(target=self._tail_loop, daemon=True).start()
        if not self.watch:
            self.start_driver()

    # Live fit ################################################################

    def live(self, step: str) -> Any:
        """Data and the model at the best values so far of a running step."""
        progress = self.progress.steps.get(step)
        if progress is None or progress.finished is not None:
            return None
        current = self._current_of(step, progress.started)
        if current is None:
            return None
        with self.lock:
            candidates = [
                (progress.best.get(g, float("inf")), g, dict(v))
                for g, v in progress.best_parameters.items()
                if set(v) == set(current.estimates)
            ]
        if not candidates:
            return None
        best, group, values = min(candidates, key=lambda c: c[0])
        key = (step, progress.started, group, best)
        # One simulation per improvement, however many tabs ask
        with self._live_lock:
            if self._live[0] != key:
                charts = live_fit(current, values)
                self._live = (key, {"group": group, "best": best, "charts": charts})
            return self._live[1]

    def _current_of(self, step: str, started: float) -> Current | None:
        """What the running step started from (rebuilt when it restarts)."""
        key = (step, started)
        with self._live_lock:
            if self._current[0] != key:
                try:
                    current = self.lab._state().current(step)  # type: ignore[arg-type]  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    current = None
                self._current = (key, current)
            return self._current[1]


def _step_of(request: Request) -> str:
    step: str = request.path_params["step"]
    return _key(int(step) if step.isdigit() else step)


def create_app(
    lab: FittingLab,
    *,
    token: str | None = None,
    hosts: set[str] | None = None,
    watch: bool = True,
    auto: bool = False,
    static: Path = STATIC,
) -> Starlette:
    """The web app of a lab (without starting it: `app.state.lab.start()`).

    Parameters
    ----------
    lab
        The lab to show
    token
        Access token of the API, None for none
    hosts
        Allowed `Host` names, None for any
    watch
        Only watch and decide; otherwise run the steps once started
    auto
        Accept every proposal while running the steps
    static
        Directory of the built frontend

    """
    state = _State(lab, watch=watch, auto=auto)

    def _error(e: Exception, status: int = 400) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status)

    async def get_overview(_: Request) -> JSONResponse:
        def build() -> dict[str, Any]:
            out = overview(lab)
            with state.lock:
                out["progress"] = state.progress.state()
            alive = state.driver is not None and state.driver.is_alive()
            out["driving"] = alive
            out["can_run"] = not state.watch
            out["driver_error"] = state.driver_error
            return out

        return JSONResponse(await run_in_threadpool(build))

    async def get_step(request: Request) -> JSONResponse:
        try:
            step = _step_of(request)
        except ValueError as e:
            return _error(e, 404)
        return JSONResponse(await run_in_threadpool(step_view, lab, step))

    async def post_decision(request: Request) -> JSONResponse:
        try:
            step = _step_of(request)
            body = await request.json()
            values = body.get("values", {}) if isinstance(body, dict) else None
            if not isinstance(values, dict):
                msg = "Send {'values': {...}}"
                raise TypeError(msg)  # noqa: TRY301
            await run_in_threadpool(lab.decide, step, **values)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            return _error(e)
        return JSONResponse({"ok": True})

    async def delete_decision(request: Request) -> JSONResponse:
        try:
            step = _step_of(request)
        except ValueError as e:
            return _error(e, 404)
        await run_in_threadpool(lab.reopen, step)
        return JSONResponse({"ok": True})

    async def get_progress(request: Request) -> JSONResponse:
        try:
            step = _step_of(request)
        except ValueError as e:
            return _error(e, 404)
        with state.lock:
            p = state.progress.steps.get(step)
            out = None if p is None else p.to_dict()
        return JSONResponse(out)

    async def get_live(request: Request) -> JSONResponse:
        try:
            step = _step_of(request)
        except ValueError as e:
            return _error(e, 404)
        return JSONResponse(await run_in_threadpool(state.live, step))

    async def post_pause(_: Request) -> JSONResponse:
        await run_in_threadpool(lab.pause)
        return JSONResponse({"ok": True})

    async def post_unpause(_: Request) -> JSONResponse:
        await run_in_threadpool(lab.unpause)
        return JSONResponse({"ok": True})

    async def post_run(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        auto = body.get("auto") if isinstance(body, dict) else None
        if state.watch:
            return _error(RunLockedError("This server only watches the run"), 409)
        if lab.store.is_driven() and not (state.driver and state.driver.is_alive()):
            return _error(RunLockedError("Another process runs the steps"), 409)
        started = state.start_driver(auto=None if auto is None else bool(auto))
        return JSONResponse({"ok": True, "started": started})

    async def get_report(_: Request) -> Response:
        try:

            def report() -> Path:
                # pyplot isn't thread-safe: one report at a time
                with state.report_lock:
                    return lab.report()

            path = await run_in_threadpool(report)
        except Exception as e:  # noqa: BLE001
            return _error(e, 500)
        return FileResponse(
            path,
            media_type="text/html",
            headers={"content-disposition": 'attachment; filename="report.html"'},
        )

    async def get_events(request: Request) -> StreamingResponse:
        try:
            seq = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            seq = 0
        if seq == 0:
            seq = state.seq

        async def stream() -> AsyncIterator[str]:
            nonlocal seq
            yield "retry: 2000\n\n"
            last = time.monotonic()
            while not state.stop.is_set():
                if await request.is_disconnected():
                    return
                for s, ev in state.events_after(seq):
                    seq = s
                    yield f"id: {s}\ndata: {json.dumps(ev)}\n\n"
                    last = time.monotonic()
                if time.monotonic() - last > 15:  # noqa: PLR2004
                    yield ": keep-alive\n\n"
                    last = time.monotonic()
                await asyncio.sleep(_TAIL)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    def get_static(request: Request) -> Response:
        root = static.resolve()
        index = root / "index.html"
        if not index.exists():
            return HTMLResponse(
                "<!doctype html><title>Fitting lab</title><p>The web UI isn't "
                "built: run <code>npm ci && npm run build</code> in "
                "<code>web/lab</code>. The API is under <code>/api</code>.</p>",
                503,
            )
        path = (root / request.path_params.get("path", "")).resolve()
        if path.is_relative_to(root) and path.is_file():
            return FileResponse(path, media_type=_media_type(path))
        if request.path_params.get("path", "").startswith("api/"):
            return JSONResponse({"error": "Not found"}, 404)
        return FileResponse(index, media_type="text/html")

    step_route = "/api/steps/{step}"
    routes = [
        Route("/api/overview", get_overview),
        Route(step_route, get_step),
        Route(f"{step_route}/decision", post_decision, methods=["POST"]),
        Route(f"{step_route}/decision", delete_decision, methods=["DELETE"]),
        Route("/api/progress/{step}", get_progress),
        Route("/api/live/{step}", get_live),
        Route("/api/pause", post_pause, methods=["POST"]),
        Route("/api/unpause", post_unpause, methods=["POST"]),
        Route("/api/run", post_run, methods=["POST"]),
        Route("/api/report", get_report),
        Route("/api/events", get_events),
        Route("/{path:path}", get_static),
    ]
    app = Starlette(routes=routes)
    app.state.lab = state
    app.add_middleware(_Guard, token=token, hosts=hosts)  # type: ignore[arg-type]
    return app


class LabServer:
    """A running web UI of a lab, see `FittingLab.serve`.

    Attributes
    ----------
    url
        Address of the UI, with the access token

    """

    def __init__(
        self,
        lab: FittingLab,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: bool | str = True,
        watch: bool = False,
        auto: bool = False,
    ) -> None:
        self.lab = lab
        self.host = host
        self.port = port
        if token is True:
            self.token: str | None = secrets.token_urlsafe(24)
        elif token is False:
            self.token = None
        else:
            self.token = str(token)
        hosts = None if host in ("0.0.0.0", "::") else _LOCAL_HOSTS | {host}  # noqa: S104
        self.app = create_app(
            lab, token=self.token, hosts=hosts, watch=watch, auto=auto
        )
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> _State:
        """Shared state of the app."""
        return self.app.state.lab

    @property
    def url(self) -> str:
        """Address of the UI (with the token)."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host  # noqa: S104
        if ":" in host:
            host = f"[{host}]"
        query = "" if self.token is None else f"/?token={self.token}"
        return f"http://{host}:{self.port}{query}"

    def start(self, *, open_browser: bool = False) -> Self:
        """Serve in a background thread (and run the steps unless watching)."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="off",
            timeout_graceful_shutdown=1,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                msg = f"The lab server couldn't start on {self.host}:{self.port}"
                raise OSError(msg)
            time.sleep(0.02)
        if self.port == 0:
            sockets = self._server.servers[0].sockets
            self.port = int(sockets[0].getsockname()[1])
        self.state.start()
        print(f"Fitting lab: {self.url}")  # noqa: T201
        if open_browser:
            webbrowser.open(self.url)
        return self

    def wait(self) -> None:
        """Block until the server stops (Ctrl-C stops it)."""
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop serving and driving (a running step finishes first)."""
        self.state.stop.set()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(5)
        driver = self.state.driver
        if driver is not None and driver.is_alive():
            print("Waiting for the running step to finish...")  # noqa: T201
            driver.join()

    def __repr__(self) -> str:
        """Address."""
        return f"LabServer({self.url!r})"
