"""Small helpers shared by debug-viewer task adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def to_jsonable(value: Any, env_index: int | None = None, max_items: int = 8) -> Any:
    """Convert tensors and small objects into JSON-safe values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if env_index is not None and getattr(value, "ndim", 0) > 0 and value.shape[0] > env_index:
            value = value[env_index]
        if getattr(value, "numel", lambda: 1)() == 1:
            return float(value.item())
        flat = value.flatten()
        return [float(item) for item in flat[:max_items].tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item, env_index, max_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, env_index, max_items) for item in value[:max_items]]
    return str(value)


def to_scalar(value: Any, env_index: int) -> float | bool | str | None:
    """Convert manager term values into a scalar when possible."""
    json_value = to_jsonable(value, env_index)
    if isinstance(json_value, list) and len(json_value) == 1:
        return json_value[0]
    if isinstance(json_value, (float, int, bool, str)) or json_value is None:
        return json_value
    return str(json_value)


def number(value: Any) -> float | None:
    """Return a float for scalar-like values, otherwise None."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return number(value[0])
    return None


def distance(a: list[float] | None, b: list[float] | None) -> float | None:
    """Return 3D Euclidean distance between two xyz positions."""
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def xy_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    """Return XY distance between two xyz positions."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def stage(
    name: str,
    label: str,
    active: bool | None,
    detail: str,
    value: Any = None,
    margin: float | None = None,
    margin_unit: str = "",
) -> dict[str, Any]:
    """Build a task-diagnostic stage row."""
    return {
        "name": name,
        "label": label,
        "active": active,
        "detail": detail,
        "value": value,
        "margin": margin,
        "marginUnit": margin_unit,
    }


def fmt_number(value: Any) -> str:
    """Format a scalar value for short diagnostics text."""
    scalar = number(value)
    if scalar is None:
        return "-"
    return f"{scalar:.3f}"


def record_error(errors: list[dict[str, str]], source: str, exc: Exception) -> None:
    """Record adapter diagnostic errors without breaking the viewer."""
    errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
