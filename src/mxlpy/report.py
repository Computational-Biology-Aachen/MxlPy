"""Generate a report comparing two models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import sympy

from mxlpy.meta.sympy_tools import fn_to_sympy, list_of_symbols
from mxlpy.model import Model

__all__ = ["AnalysisFn", "MarkdownReport", "markdown"]

type AnalysisFn = Callable[[Model, Model, Path], tuple[str, Path]]


def _latex_view(expr: sympy.Expr | None) -> str:
    if expr is None:
        return "<span style='color:red'>PARSE ERROR<span>"
    return f"${sympy.latex(expr)}$"


def _new_removed_changed[T](
    d1: dict[str, T], d2: dict[str, T]
) -> tuple[dict[str, T], list[str], dict[str, tuple[T, T]]]:
    s1 = set(d1)
    s2 = set(d2)

    removed = sorted(s1 - s2)
    new = {k: d2[k] for k in s2 - s1}
    changed = {k: (v1, v2) for k in s1 - set(removed) if (v1 := d1[k]) != (v2 := d2[k])}
    return new, removed, changed


def _table_row(items: list[str]) -> str:
    return f"| {' | '.join(items)} |"


def _table_header(items: list[str]) -> str:
    return f"{_table_row(items)}\n{_table_row(['---'] * len(items))}"


@dataclass
class MarkdownReport:
    """Report of model comparison."""

    data: str

    def __str__(self) -> str:
        """Markdown string representation."""
        return self.data

    def __repr__(self) -> str:
        """Markdown string representation."""
        return self.data

    def _repr_markdown_(self) -> str:
        return self.data

    def write(self, path: Path) -> None:
        """Write report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w+") as fp:
            fp.write(self.data)


def markdown(
    m1: Model,
    m2: Model,
    *,
    analyses: list[AnalysisFn] | None = None,
    rel_change: float = 1e-2,
    img_path: Path = Path(),
    m1_name: str = "model 1",
    m2_name: str = "model 2",
    include_rhs: bool = True,
) -> MarkdownReport:
    """Generate a markdown report comparing two models.

    Args:
        m1: The first model to compare
        m2: The second model to compare
        analyses: A list of functions that analyze both models and return a report section with image
        rel_change: The relative change threshold for numerical differences
        img_path: The path to save images
        m1_name: Name of the first model
        m2_name: Name of the second model
        include_rhs: Whether to include numerical differences in the right hand side

    Returns:
        str: Markdown formatted report comparing the two models

    Examples:
        >>> from mxlpy import Model
        >>> m1 = Model().add_parameter("k1", 0.1).add_variable("S", 1.0)
        >>> m2 = Model().add_parameter("k1", 0.2).add_variable("S", 1.0)
        >>> report = markdown(m1, m2)
        >>> "Parameters" in report and "k1" in report
        True

        >>> # With custom analysis function
        >>> def custom_analysis(m1, m2, path):
        ...     return "## Custom analysis", path / "image.png"
        >>> report = markdown(m1, m2, analyses=[custom_analysis])
        >>> "Custom analysis" in report
        True

    """
    content: list[str] = [
        f"# Report:  {datetime.now(UTC).strftime('%Y-%m-%d')}\n",
    ]

    # Unused
    if unused := m2.get_unused_parameters():
        content.append("## <span style='color: red'>Unused parameters</span>\n")
        names = "\n".join(f"<li>{i}</li>\n" for i in sorted(unused))
        content.append(f"<ul>\n{names}\n</ul>\n")

    # Model stats
    content.extend(
        [
            f"| Model component | {m1_name} | {m2_name} |",
            "| --- | --- | --- |",
            f"| variables | {len(m1.get_raw_parameters())} | {len(m2.get_raw_parameters())}|",
            f"| parameters | {len(m1.get_parameter_values())} | {len(m2.get_parameter_values())}|",
            f"| derived parameters | {len(m1.get_derived_parameters())} | {len(m2.get_derived_parameters())}|",
            f"| derived variables | {len(m1.get_derived_variables())} | {len(m2.get_derived_variables())}|",
            f"| reactions | {len(m1.get_raw_reactions())} | {len(m2.get_raw_reactions())}|",
            f"| surrogates | {len(m1._surrogates)} | {len(m2._surrogates)}|",  # noqa: SLF001
        ]
    )

    # Variables
    new_variables, removed_variables, changed_variables = _new_removed_changed(
        m1.get_initial_conditions(), m2.get_initial_conditions()
    )
    variables = []
    variables.extend(
        f"| <span style='color:green'>{k}<span> | - | {v} |"
        for k, v in new_variables.items()
    )
    variables.extend(
        f"| <span style='color: orange'>{k}</span> | {v1} | {v2} |"
        for k, (v1, v2) in changed_variables.items()
    )
    variables.extend(
        f"| <span style='color:red'>{k}</span> | - | - |" for k in removed_variables
    )
    if len(variables) >= 1:
        content.extend(
            (
                "\n## Variables\n",
                f"| Name | {m1_name} | {m2_name} |",
                "| ---- | --------- | --------- |",
            )
        )
        content.append("\n".join(variables))

    # Parameters
    new_parameters, removed_parameters, changed_parameters = _new_removed_changed(
        m1.get_parameter_values(), m2.get_parameter_values()
    )
    pars = []
    pars.extend(
        f"| <span style='color:green'>{k}<span> | - | {v} |"
        for k, v in new_parameters.items()
    )
    pars.extend(
        f"| <span style='color: orange'>{k}</span> | {v1} | {v2} |"
        for k, (v1, v2) in changed_parameters.items()
    )
    pars.extend(
        f"| <span style='color:red'>{k}</span> | - | - |" for k in removed_parameters
    )
    if len(pars) >= 1:
        content.extend(
            (
                "\n## Parameters\n",
                f"| Name | {m1_name} | {m2_name} |",
                "| ---- | --------- | --------- |",
            )
        )
        content.append("\n".join(pars))

    # Derived
    new_derived, removed_derived, changed_derived = _new_removed_changed(
        m1.get_raw_derived(),
        m2.get_raw_derived(),
    )
    derived = []
    for k, v in new_derived.items():
        expr = _latex_view(
            fn_to_sympy(
                v.fn,
                origin=k,
                model_args=list_of_symbols(v.args),
            )
        )
        derived.append(f"| <span style='color:green'>{k}<span> | - | {expr} |")

    for k, (v1, v2) in changed_derived.items():
        expr1 = _latex_view(
            fn_to_sympy(
                v1.fn,
                origin=k,
                model_args=list_of_symbols(v1.args),
            )
        )
        expr2 = _latex_view(
            fn_to_sympy(
                v2.fn,
                origin=k,
                model_args=list_of_symbols(v2.args),
            )
        )
        derived.append(
            f"| <span style='color: orange'>{k}</span> | {expr1} | {expr2} |"
        )
    derived.extend(
        f"| <span style='color:red'>{k}</span> | - | - |" for k in removed_derived
    )
    if len(derived) >= 1:
        content.extend(
            (
                "\n## Derived\n",
                f"| Name | {m1_name} | {m2_name} |",
                "| ---- | --------- | --------- |",
            )
        )
        content.append("\n".join(derived))

    # Reactions
    new_reactions, removed_reactions, changed_reactions = _new_removed_changed(
        m1.get_raw_reactions(), m2.get_raw_reactions()
    )
    reactions = []
    for k, v in new_reactions.items():
        expr = _latex_view(
            fn_to_sympy(
                v.fn,
                origin=k,
                model_args=list_of_symbols(v.args),
            )
        )
        reactions.append(f"| <span style='color:green'>{k}<span> | - | {expr} |")

    for k, (v1, v2) in changed_reactions.items():
        expr1 = _latex_view(
            fn_to_sympy(
                v1.fn,
                origin=k,
                model_args=list_of_symbols(v1.args),
            )
        )
        expr2 = _latex_view(
            fn_to_sympy(
                v2.fn,
                origin=k,
                model_args=list_of_symbols(v2.args),
            )
        )
        reactions.append(
            f"| <span style='color: orange'>{k}</span> | {expr1} | {expr2} |"
        )
    reactions.extend(
        f"| <span style='color:red'>{k}</span> | - | - |" for k in removed_reactions
    )

    if len(reactions) >= 1:
        content.extend(
            (
                "\n## Reactions\n",
                f"| Name | {m1_name} | {m2_name} |",
                "| ---- | --------- | --------- |",
            )
        )
        content.append("\n".join(reactions))

    # Now check for any numerical differences
    dependent = []
    d1 = m1.get_args()
    d2 = m2.get_args()
    rel_diff = ((d1 - d2) / d1).dropna()
    for k, v in rel_diff.loc[rel_diff.abs() >= rel_change].items():
        k = cast(str, k)
        dependent.append(
            f"| <span style='color:orange'>{k}</span> | {d1[k]:.2f} | {d2[k]:.2f} |  {v:.1%} |"
        )
    if len(dependent) >= 1:
        content.extend(
            (
                "## Numerical differences of dependent values\n\n",
                f"| Name | {m1_name} | {m2_name} | Relative Change | ",
                "| ---- | --------- | --------- | --------------- | ",
            )
        )
        content.append("\n".join(dependent))

    if include_rhs:
        rhs = []
        r1 = m1.get_right_hand_side()
        r2 = m2.get_right_hand_side()
        rel_diff = ((r1 - r2) / r1).dropna()
        for k, v in rel_diff.loc[rel_diff.abs() >= rel_change].items():
            k = cast(str, k)
            rhs.append(
                f"| <span style='color:orange'>{k}</span> | {r1[k]:.2f} | {r2[k]:.2f} |  {v:.1%} |"
            )
        if len(rhs) >= 1:
            content.extend(
                (
                    "\n## Numerical differences of right hand side values\n",
                    f"| Name | {m1_name} | {m2_name} | Relative Change | ",
                    "| ---- | --------- | --------- | --------------- | ",
                )
            )
            content.append("\n".join(rhs))

    # Comparison functions
    if analyses is not None:
        for f in analyses:
            name, img_path = f(m1, m2, img_path)
            content.append(name)
            # content.append(f"![{name}]({img_path})")
            content.append(f"<img src='{img_path}' alt='{name}' width='500'/>")

    return MarkdownReport(data="\n".join(content))
