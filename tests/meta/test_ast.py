from __future__ import annotations

import math

import numpy  # noqa: ICN001
import numpy as np
from numpy import sin as numpy_sin

from mxlpy.meta.source_tools import fn_to_source


def half(x: float) -> float:
    return np.divide(x, 0.5)


def simple_no_deps(x: float) -> float:
    """Function with no external dependencies."""
    return x * 2.0


def helper_function(x: float) -> float:
    """Helper function for testing dependency resolution."""
    return x + 1.0


def function_with_deps(x: float) -> float:
    """Function that calls another user-defined function."""
    result = helper_function(x)
    return result * 2.0


def function_with_numpy(x: float) -> float:
    """Function that uses numpy."""
    return float(numpy.divide(x, 0.5))


def function_with_numpy_alias(x: float) -> float:
    """Function that uses numpy aliased as np."""
    return float(np.divide(x, 0.5))


def function_with_builtin(x: float) -> float:
    """Function that calls builtin abs and float."""
    return float(abs(x))


def function_with_numpy_nested(x: float) -> float:
    """Function using numpy.linalg nested module access."""
    return float(numpy.linalg.norm([x]))


def function_with_numpy_alias_nested(x: float) -> float:
    """Function using np.linalg nested module access."""
    return float(np.linalg.norm([x]))


def function_with_stdlib(x: float) -> float:
    """Function that uses a stdlib (math) module."""
    return math.sqrt(x)


def function_calls_numpy_sin(x: float) -> float:
    """Function calling a numpy C-extension directly (no source available)."""
    return float(numpy_sin(x))


def circular_a(x: float) -> float:
    """Mutually recursive with circular_b."""
    return circular_b(x) + 1.0


def circular_b(x: float) -> float:
    """Mutually recursive with circular_a."""
    return circular_a(x) + 1.0


def level_a(x: float) -> float:
    """Leaf of a three-level dependency chain."""
    return x + 1.0


def level_b(x: float) -> float:
    """Mid-level function calling level_a."""
    return level_a(x) * 2.0


def level_c(x: float) -> float:
    """Top-level function calling level_b (transitive dep on level_a)."""
    return level_b(x) - 1.0


###############################################################################
# Tests
###############################################################################


def test_simple_function() -> None:
    """Test that simple function source is correctly extracted."""
    result = fn_to_source(simple_no_deps, "simple_no_deps", [])

    assert result.main_source.strip().split("\n") == [
        "def simple_no_deps(x: float) -> float:",
        '    """Function with no external dependencies."""',
        "    return x * 2.0",
    ]
    assert len(result.imports) == 0
    assert len(result.dependencies) == 0


def test_function_with_dependency() -> None:
    """Test that called user functions are discovered."""
    result = fn_to_source(
        function_with_deps,
        "function_with_deps",
        [],
    )

    assert result.main_source.strip().split("\n") == [
        "def function_with_deps(x: float) -> float:",
        '    """Function that calls another user-defined function."""',
        "    result = helper_function(x)",
        "    return result * 2.0",
    ]
    assert "helper_function" in result.dependencies
    assert result.dependencies["helper_function"].strip().split("\n") == [
        "def helper_function(x: float) -> float:",
        '    """Helper function for testing dependency resolution."""',
        "    return x + 1.0",
    ]


def test_numpy_full_import() -> None:
    """numpy.divide (full module name) adds 'import numpy' to imports."""
    result = fn_to_source(
        function_with_numpy,
        "function_with_numpy",
        [],
    )

    assert "import numpy" in result.imports
    assert len(result.dependencies) == 0


def test_numpy_alias_import() -> None:
    """np.divide (aliased import) adds 'import numpy as np' to imports."""
    result = fn_to_source(
        function_with_numpy_alias,
        "function_with_numpy_alias",
        [],
    )

    assert "import numpy as np" in result.imports
    assert len(result.dependencies) == 0


def test_circular_dependency_no_infinite_loop() -> None:
    """Mutually recursive functions don't cause infinite recursion."""
    result = fn_to_source(circular_a, "circular_a", [])

    # circular_b is a direct dep
    assert "circular_b" in result.dependencies
    # circular_a is the root — visited guard must block re-adding it
    assert "circular_a" not in result.dependencies


def test_multi_level_dependencies() -> None:
    """Transitive deps (level_c → level_b → level_a) are all resolved."""
    result = fn_to_source(level_c, "level_c", [])

    assert "level_b" in result.dependencies
    assert "level_a" in result.dependencies


def test_stdlib_not_in_deps_or_imports() -> None:
    """stdlib calls (math.sqrt) are not added to dependencies or imports."""
    result = fn_to_source(
        function_with_stdlib,
        "function_with_stdlib",
        [],
    )

    assert len(result.dependencies) == 0
    assert not any("math" in imp for imp in result.imports)


def test_unavailable_source_skipped_gracefully() -> None:
    """Calling a C-extension (numpy_sin) completes without crashing."""
    result = fn_to_source(
        function_calls_numpy_sin,
        "function_calls_numpy_sin",
        [],
    )

    # C-extension source is unavailable — must be silently skipped
    assert "numpy_sin" not in result.dependencies
    assert "sin" not in result.dependencies


def test_builtin_not_in_deps_or_imports() -> None:
    """Builtin calls (abs, float) are not added to dependencies or imports."""
    result = fn_to_source(
        function_with_builtin,
        "function_with_builtin",
        [],
    )

    assert len(result.dependencies) == 0
    assert len(result.imports) == 0


def test_numpy_nested_module_import() -> None:
    """numpy.linalg.norm (nested access) still adds 'import numpy' to imports."""
    result = fn_to_source(
        function_with_numpy_nested,
        "function_with_numpy_nested",
        [],
    )

    assert "import numpy" in result.imports
    assert len(result.dependencies) == 0


def test_numpy_alias_nested_module_import() -> None:
    """np.linalg.norm (nested access via alias) still adds 'import numpy as np' to imports."""
    result = fn_to_source(
        function_with_numpy_alias_nested,
        "function_with_numpy_alias_nested",
        [],
    )

    assert "import numpy as np" in result.imports
    assert len(result.dependencies) == 0
