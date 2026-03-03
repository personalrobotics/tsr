"""Relational TSR library: register and query TSR templates by entity and task."""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Tuple, Union
import numpy as np

from .template import TSRTemplate
from .tsr import TSR
from .schema import TaskType, EntityClass

# (subject, reference, task) → list of templates
_Key = Tuple[EntityClass, EntityClass, TaskType]

# Generator: takes T_ref_world, returns templates
Generator = Callable[[np.ndarray], List[TSRTemplate]]


class TSRLibraryRelational:
    """Registry for TSR templates keyed by (subject, reference, task).

    Supports two registration modes:
    - register(): register a generator function that produces templates
    - register_template(): register individual templates directly

    Both modes are queried through query(), which always returns
    instantiated TSR objects.
    """

    def __init__(self) -> None:
        self._generators: Dict[_Key, Generator] = {}
        self._templates: Dict[_Key, List[TSRTemplate]] = {}

    def register(
        self, subject: EntityClass, reference: EntityClass,
        task: TaskType, generator: Generator
    ) -> None:
        """Register a generator function for a (subject, reference, task) key."""
        self._generators[(subject, reference, task)] = generator

    def register_template(
        self, subject: EntityClass, reference: EntityClass,
        task: TaskType, template: TSRTemplate, description: str = ""
    ) -> None:
        """Register a single template for a (subject, reference, task) key."""
        key = (subject, reference, task)
        self._templates.setdefault(key, []).append(template)

    def query(
        self, subject: EntityClass, reference: EntityClass,
        task: TaskType, T_ref_world: np.ndarray
    ) -> List[TSR]:
        """Get instantiated TSRs for a (subject, reference, task) key.

        Checks generators first, then registered templates.

        Raises:
            KeyError: If no generator or templates are registered for the key.
        """
        key = (subject, reference, task)

        if key in self._generators:
            templates = self._generators[key](T_ref_world)
            return [t.instantiate(T_ref_world) for t in templates]

        if key in self._templates:
            return [t.instantiate(T_ref_world) for t in self._templates[key]]

        raise KeyError(f"No generator or templates registered for {key}")

    def query_templates(
        self, subject: EntityClass, reference: EntityClass, task: TaskType
    ) -> List[TSRTemplate]:
        """Get registered templates (not instantiated) for a key.

        Raises:
            KeyError: If no templates are registered for the key.
        """
        key = (subject, reference, task)
        if key not in self._templates:
            raise KeyError(f"No templates registered for {key}")
        return list(self._templates[key])

    def list_tasks_for_reference(
        self, reference: EntityClass,
        subject_filter: Optional[EntityClass] = None,
        task_prefix: Optional[str] = None
    ) -> List[TaskType]:
        """List all tasks available for a reference entity."""
        all_keys = set(self._generators.keys()) | set(self._templates.keys())
        tasks = []
        for subj, ref, task in all_keys:
            if ref != reference:
                continue
            if subject_filter is not None and subj != subject_filter:
                continue
            if task_prefix is not None and not str(task).startswith(task_prefix):
                continue
            tasks.append(task)
        return tasks
