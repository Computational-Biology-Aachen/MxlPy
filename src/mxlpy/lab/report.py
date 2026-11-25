# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Self-contained HTML and JSON report of a lab run."""

from __future__ import annotations

import base64
import html as html_lib
import io
import logging
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mxlpy import plot
from mxlpy.lab._core import STEPS

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from mxlpy.lab._core import StepResult
    from mxlpy.lab._lab import FittingLab

__all__ = ["write_report"]

_LOGGER = logging.getLogger(__name__)

_TITLES = {
    "setup": "0 · Setup",
    "parameters": "1 · Parameter selection",
    "fit": "2 · Fit",
    "diagnose": "3 · Missing dynamics",
    "identifiability": "4 · Identifiability",
    "uncertainty": "5 · Uncertainty",
}

_STYLE = """
:root { --fg: #1d2327; --muted: #5f6b73; --bg: #ffffff; --line: #d9dee2;
        --ok: #1e7a46; --bad: #b42318; --accent: #006165; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e9eb; --muted: #9aa5ad; --bg: #15191c; --line: #2f363b;
          --ok: #5fcf8d; --bad: #f08a80; --accent: #4fb3b6; } }
body { font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
       max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { color: var(--accent); } h2 { border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.status { font-weight: 600; } .passed { color: var(--ok); } .failed { color: var(--bad); }
.muted { color: var(--muted); }
table { border-collapse: collapse; margin: 8px 0; font-size: 13px; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 3px 8px; text-align: right; }
th { text-align: left; } img { max-width: 100%; height: auto; }
pre { white-space: pre-wrap; }
"""


def _figure(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img alt="figure" src="data:image/png;base64,{data}">'


def _table(df: pd.DataFrame | pd.Series) -> str:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.to_html(float_format=lambda v: f"{v:.4g}", border=0)


def _fit_figure(lab: FittingLab, result: StepResult) -> str | None:
    ev = result.artifacts.get("evaluation")
    if ev is None:
        return None
    conditions = lab.results["setup"].artifacts["current"].conditions
    n = len(conditions)
    fig, axs = plot.grid_layout(n, n_cols=min(n, 3), col_width=4, row_height=3)
    for ax, (name, cond) in zip(
        np.ravel(np.asarray(axs)), conditions.items(), strict=False
    ):
        pred = ev.predictions.get(name)
        data = cast(pd.DataFrame, cond.data)
        for i, col in enumerate(data.columns):
            color = f"C{i}"
            ax.plot(data.index, data[col], "o", color=color, ms=3, alpha=0.6)
            if pred is not None:
                ax.plot(pred.index, pred[col], "-", color=color, label=col)
        ax.set_title(name)
        ax.legend(fontsize=8)
    return _figure(fig)


def _figures(lab: FittingLab, step: str, result: StepResult) -> list[str]:
    out: list[str] = []
    try:
        if step == "parameters":
            fig, ax = plot._default_fig_ax(ax=None, grid=True, figsize=(6, 3))  # noqa: SLF001
            importance = pd.Series(
                {
                    k: float(v["importance"])
                    for k, v in result.summary["parameters"].items()
                }
            )
            importance.plot.bar(ax=ax)
            ax.set_ylabel("RMS scaled sensitivity")
            out.append(_figure(fig))
        elif step == "fit" and (img := _fit_figure(lab, result)) is not None:
            out.append(img)
        elif step == "diagnose":
            chosen = result.summary.get("chosen_hidden")
            if chosen is not None:
                fig, _ = plot.correction(result.artifacts["augmented"][chosen])
                out.append(_figure(fig))
        elif step == "identifiability" and "profiles" in result.artifacts:
            fig, _ = plot.profile_likelihood(result.artifacts["profiles"])
            out.append(_figure(fig))
        elif step == "uncertainty":
            for name, bands in result.artifacts.get("bands", {}).items():
                for col, band in bands.items():
                    fig, ax = plot.prediction_band(band, label=col)
                    data = cast(pd.DataFrame, lab.conditions[name].data)
                    if col in data:
                        ax.plot(data.index, data[col], "o", ms=3, color="k")
                    ax.set_title(f"{name}: {col}")
                    out.append(_figure(fig))
    except Exception as e:  # noqa: BLE001
        _LOGGER.warning("Report figures of %s failed: %s", step, e)
    return out


def _summary_tables(step: str, result: StepResult) -> list[str]:
    s = result.summary
    out: list[str] = []
    if step == "setup":
        out.append(_table(pd.Series(s.get("targets", {}), name="target")))
    elif step == "parameters":
        out.append(_table(pd.DataFrame(s.get("parameters", {})).T))
        out.append(_table(pd.Series(s.get("limits", {}), name="limit")))
    elif step == "fit":
        out.append(_table(pd.Series(s.get("estimates", {}), name="estimate")))
        joint = s.get("joint", {})
        if joint.get("residuals"):
            out.append(_table(pd.DataFrame(joint["residuals"])))
        if joint.get("nrmse"):
            out.append(_table(pd.DataFrame(joint["nrmse"])))
    elif step == "diagnose":
        out.append(_table(pd.DataFrame(s.get("hidden", {})).T))
        if s.get("variants"):
            out.append(_table(pd.DataFrame(s["variants"])))
    elif step == "identifiability":
        if "profiles" in s:
            out.append(_table(cast(pd.DataFrame, s["profiles"]["ci"])))
        elif "fisher" in s:
            out.append(_table(cast(pd.DataFrame, s["fisher"]["wald_ci"])))
    return out


def _section(lab: FittingLab, step: str) -> str:
    result = lab.results.get(step)
    title = html_lib.escape(_TITLES[step])
    if result is None:
        return f"<h2>{title}</h2><p class='muted'>Not run.</p>"
    if result.skipped:
        return f"<h2>{title}</h2><p class='muted'>Not needed.</p>"
    status = {True: "passed", False: "failed", None: "done"}[result.passed]
    decision = lab.decisions().get(step)
    parts = [
        f"<h2>{title}</h2>",
        (
            f"<p><span class='status {status}'>{status}</span> "
            f"<span class='muted'>· {result.duration:.1f} s</span></p>"
        ),
        "<ul>"
        + "".join(f"<li>{html_lib.escape(m)}</li>" for m in result.messages)
        + "</ul>",
    ]
    if decision is not None:
        parts.append(
            f"<p class='muted'>Decision ({html_lib.escape(str(decision.get('source')))}): "
            f"{html_lib.escape(str(decision.get('values')))}</p>"
        )
    parts.extend(_summary_tables(step, result))
    parts.extend(_figures(lab, step, result))
    return "\n".join(parts)


def write_report(lab: FittingLab, *, html: bool = True) -> Path:
    """Write `report.json` and (with `html`) `report.html` into the run directory."""
    store = lab.store
    store.write_json(
        store.root / "report.json",
        {
            "config": lab.config.to_dict(),
            "decisions": lab.decisions(),
            "steps": {k: v.to_dict() for k, v in lab.results.items()},
            "runtime": lab.runtime.to_dict(),
            "active_time": lab.active_time,
        },
    )
    path = store.root / "report.html"
    if not html:
        return path
    with mpl.rc_context({"figure.max_open_warning": 0}):
        body = "\n".join(_section(lab, step) for step in STEPS)
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Fitting lab report</title><style>{_STYLE}</style></head><body>"
        f"<h1>Fitting lab</h1><p class='muted'>{html_lib.escape(str(store.root))}</p>"
        f"{body}</body></html>"
    )
    path.write_text(page, encoding="utf-8")
    return path
