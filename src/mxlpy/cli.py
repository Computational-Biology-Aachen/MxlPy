# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Command line interface: `mxlpy lab serve|status|report <run_dir>`."""

from __future__ import annotations

import argparse
import os
import sys

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mxlpy", description="MxlPy tools")
    commands = parser.add_subparsers(dest="command", required=True)
    lab = commands.add_parser("lab", help="Fitting lab runs")
    actions = lab.add_subparsers(dest="action", required=True)

    serve = actions.add_parser(
        "serve",
        help="Watch and steer a run in the browser (needs mxlpy[lab])",
        description=(
            "Serve the web UI of a run. By default the server also runs the "
            "steps, waiting at every gate until it is decided in the browser."
        ),
    )
    serve.add_argument("run_dir", help="Directory of the run")
    serve.add_argument("--host", default="127.0.0.1", help="Interface to listen on")
    serve.add_argument("--port", type=int, default=8765, help="Port (0: any free)")
    serve.add_argument(
        "--no-token", action="store_true", help="Don't require the access token"
    )
    serve.add_argument(
        "--no-browser", action="store_true", help="Don't open the browser"
    )
    serve.add_argument(
        "--watch",
        action="store_true",
        help="Only watch and decide; another process runs the steps",
    )
    serve.add_argument(
        "--auto", action="store_true", help="Accept every proposal while running"
    )

    status = actions.add_parser("status", help="Steps and gates of a run")
    status.add_argument("run_dir")

    report = actions.add_parser("report", help="Write report.html of a run")
    report.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    args = _parser().parse_args(argv)
    # The server renders report figures in worker threads: no GUI backend
    os.environ.setdefault("MPLBACKEND", "Agg")
    from mxlpy.lab import FittingLab  # noqa: PLC0415 (fast --help)

    try:
        run = FittingLab.resume(args.run_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)  # noqa: T201
        return 1
    if args.action == "status":
        print(run)  # noqa: T201
    elif args.action == "report":
        print(run.report())  # noqa: T201
    else:
        try:
            server = run.serve(
                host=args.host,
                port=args.port,
                token=not args.no_token,
                open_browser=not args.no_browser,
                watch=args.watch,
                auto=args.auto,
            )
        except ImportError as e:
            print(  # noqa: T201
                f"{e}\nThe web UI needs the lab extra: pip install 'mxlpy[lab]'",
                file=sys.stderr,
            )
            return 1
        server.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
