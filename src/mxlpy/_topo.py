"""Topological searches for dependencies."""

from dataclasses import dataclass
from queue import Empty, SimpleQueue

from wadler_lindig import pformat


class CircularDependencyError(Exception):
    """Raised when dependencies cannot be sorted topologically.

    This typically indicates circular dependencies in model components.
    """

    def __init__(
        self,
        missing: dict[str, set[str]],
    ) -> None:
        """Initialise exception."""
        missing_by_module = "\n".join(f"\t{k}: {v}" for k, v in missing.items())
        msg = (
            f"Exceeded max iterations on sorting dependencies.\n"
            "Check if there are circular references. "
            "Missing dependencies:\n"
            f"{missing_by_module}"
        )
        super().__init__(msg)


class MissingDependenciesError(Exception):
    """Raised when dependencies cannot be sorted topologically.

    This typically indicates circular dependencies in model components.
    """

    def __init__(self, not_solvable: dict[str, list[str]]) -> None:
        """Initialise exception."""
        missing_by_module = "\n".join(f"\t{k}: {v}" for k, v in not_solvable.items())
        msg = (
            f"Dependencies cannot be solved. Missing dependencies:\n{missing_by_module}"
        )
        super().__init__(msg)


@dataclass
class Dependency:
    """Container class for building dependency tree.

    This is mostly needed due to surrogates, as they have multiple outputs.
    So one surrogate requires `n` args and provides `m` outputs.
    """

    name: str
    required: set[str]
    provided: set[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


def _check_if_is_sortable(
    available: set[str],
    elements: list[Dependency],
) -> None:
    all_available = available.copy()
    for dependency in elements:
        all_available.update(dependency.provided)

    # Check if it can be sorted in the first place
    not_solvable = {}
    for dependency in elements:
        if not dependency.required.issubset(all_available):
            not_solvable[dependency.name] = sorted(
                dependency.required.difference(all_available)
            )

    if not_solvable:
        raise MissingDependenciesError(not_solvable=not_solvable)


def sort_dependencies(
    available: set[str],
    elements: list[Dependency],
) -> list[str]:
    """Sort model elements topologically based on their dependencies.

    Parameters
    ----------
    available
        Set of available component names
    elements
        List of (name, dependencies, supplier) tuples to sort

    Returns
    -------
        List of element names in dependency order

    Raises
    ------
    SortError
        If circular dependencies are detected

    """
    _check_if_is_sortable(available, elements)

    order = []
    # FIXME: what is the worst case here?
    max_iterations = len(elements) ** 2
    queue: SimpleQueue[Dependency] = SimpleQueue()
    for dependency in elements:
        queue.put(dependency)

    last_name = None
    i = 0
    while True:
        try:
            dependency = queue.get_nowait()
        except Empty:
            break
        if dependency.required.issubset(available):
            available.update(dependency.provided)
            order.append(dependency.name)

        else:
            if last_name == dependency.name:
                order.append(last_name)
                break
            queue.put(dependency)
            last_name = dependency.name
        i += 1

        # Failure case
        if i > max_iterations:
            unsorted = []
            while True:
                try:
                    unsorted.append(queue.get_nowait().name)
                except Empty:
                    break

            mod_to_args: dict[str, set[str]] = {
                dependency.name: dependency.required for dependency in elements
            }
            missing = {k: mod_to_args[k].difference(available) for k in unsorted}
            raise CircularDependencyError(missing=missing)
    return order


def sort_all(
    available: set[str],
    elements: list[Dependency],
) -> list[str]:
    """Sort all model elements topologically.

    Parameters
    ----------
    available
        Set of available component names
    elements
        List of (name, dependencies, supplier) tuples to sort

    Returns
    -------
        List of element names in dependency order

    Raises
    ------
    SortError
        If circular dependencies are detected

    """
    return sorted(available - {"time"}) + sort_dependencies(available, elements)


def get_all_dependencies_of(
    requested: set[str],
    leaves: set[str],
    dependees: dict[str, set[str]],
):

    needed: set[str] = set()
    queue: SimpleQueue[str] = SimpleQueue()
    for name in requested:
        queue.put(name)

    while True:
        try:
            name = queue.get_nowait()

            if name in needed:
                continue
            needed.add(name)

            if name in leaves:
                continue

            for dep in dependees[name].difference(needed):
                queue.put(dep)
        except Empty:
            break
    return needed
