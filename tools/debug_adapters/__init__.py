"""Task-specific diagnostic adapters for the Isaac Lab debug viewer."""

from __future__ import annotations

from types import ModuleType


def resolve_task_adapter(task: str | None) -> ModuleType | None:
    """Return the adapter module for a task, or None for generic-only tasks."""
    from . import object_in_bowl

    adapters = (object_in_bowl,)
    for adapter in adapters:
        if adapter.matches(task):
            return adapter
    return None
