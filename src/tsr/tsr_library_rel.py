from __future__ import annotations

from typing import Dict, List, Optional, Callable, Tuple, Union
import numpy as np

try:
    from tsr.core.tsr_template import TSRTemplate  # type: ignore[attr-defined]
    from tsr.schema import TaskType, EntityClass  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    TSRTemplate = object  # type: ignore[assignment]
    TaskType = object  # type: ignore[assignment]
    EntityClass = object  # type: ignore[assignment]

# Type alias for generator functions
Generator = Callable[[np.ndarray], List[TSRTemplate]]

# Type alias for relational keys
RelKey = tuple[EntityClass, EntityClass, TaskType]

# Type alias for template entries with descriptions
TemplateEntry = dict[str, Union[TSRTemplate, str]]


class TSRLibraryRelational:
    """Relational TSR library for task-based TSR generation and querying.
    
    This class provides a registry for TSR generators that can be queried
    based on subject entity, reference entity, and task type. It enables
    task-based TSR generation where different TSR templates are available
    for different combinations of entities and tasks.
    
    The library uses a relational key structure: (subject, reference, task)
    where:
    - subject: The entity performing the action (e.g., gripper)
    - reference: The entity being acted upon (e.g., object, surface)
    - task: The type of task being performed (e.g., grasp, place)
    
    The library supports two registration modes:
    1. Generator-based: Register functions that generate templates dynamically
    2. Template-based: Register individual templates with descriptions
    """

    def __init__(self) -> None:
        """Initialize an empty relational TSR library."""
        self._reg: Dict[RelKey, Generator] = {}
        self._templates: Dict[RelKey, List[TemplateEntry]] = {}

    def register(
        self, 
        subject: EntityClass, 
        reference: EntityClass, 
        task: TaskType, 
        generator: Generator
    ) -> None:
        """Register a TSR generator for a specific entity/task combination.
        
        Args:
            subject: The entity performing the action (e.g., gripper)
            reference: The entity being acted upon (e.g., object, surface)
            task: The type of task being performed
            generator: Function that takes T_ref_world and returns list of TSRTemplate objects
        """
        self._reg[(subject, reference, task)] = generator

    def register_template(
        self,
        subject: EntityClass,
        reference: EntityClass,
        task: TaskType,
        template: TSRTemplate,
        description: str = ""
    ) -> None:
        """Register a TSR template with semantic context and description.
        
        Args:
            subject: The entity performing the action (e.g., gripper)
            reference: The entity being acted upon (e.g., object, surface)
            task: The type of task being performed
            template: The TSR template to register
            description: Optional description of the template
        """
        key = (subject, reference, task)
        if key not in self._templates:
            self._templates[key] = []
        
        self._templates[key].append({
            'template': template,
            'description': description
        })

    def query(
        self, 
        subject: EntityClass, 
        reference: EntityClass, 
        task: TaskType, 
        T_ref_world: np.ndarray
    ) -> List["CoreTSR"]:
        """Query TSRs for a specific entity/task combination.
        
        This method looks up the registered generator for the given
        subject/reference/task combination and calls it with the provided
        reference pose to generate concrete TSRs.
        
        Args:
            subject: The entity performing the action
            reference: The entity being acted upon
            task: The type of task being performed
            T_ref_world: 4×4 pose of the reference entity in world frame
            
        Returns:
            List of instantiated TSR objects
            
        Raises:
            KeyError: If no generator is registered for the given combination
        """
        try:
            from tsr.core.tsr import TSR as CoreTSR  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            CoreTSR = object  # type: ignore[assignment]
            
        key = (subject, reference, task)
        if key not in self._reg:
            raise KeyError(f"No generator registered for {key}")
        
        generator = self._reg[key]
        templates = generator(T_ref_world)
        return [tmpl.instantiate(T_ref_world) for tmpl in templates]

    def query_templates(
        self,
        subject: EntityClass,
        reference: EntityClass,
        task: TaskType,
        include_descriptions: bool = False
    ) -> Union[List[TSRTemplate], List[Tuple[TSRTemplate, str]]]:
        """Query templates for a specific entity/task combination.
        
        This method returns the registered templates for the given
        subject/reference/task combination.
        
        Args:
            subject: The entity performing the action
            reference: The entity being acted upon
            task: The type of task being performed
            include_descriptions: If True, return (template, description) tuples
            
        Returns:
            List of TSRTemplate objects or (template, description) tuples
            
        Raises:
            KeyError: If no templates are registered for the given combination
        """
        key = (subject, reference, task)
        if key not in self._templates:
            raise KeyError(f"No templates registered for {key}")
        
        entries = self._templates[key]
        if include_descriptions:
            return [(entry['template'], entry['description']) for entry in entries]
        else:
            return [entry['template'] for entry in entries]

    def list_tasks_for_reference(
        self, 
        reference: EntityClass, 
        subject_filter: Optional[EntityClass] = None,
        task_prefix: Optional[str] = None
    ) -> List[TaskType]:
        """List all tasks available for a reference entity.
        
        This method discovers what tasks can be performed on a given
        reference entity by examining the registered generators.
        
        Args:
            reference: The reference entity to list tasks for
            subject_filter: Optional filter to only show tasks for specific subject
            task_prefix: Optional filter to only show tasks starting with this prefix
            
        Returns:
            List of TaskType objects that can be performed on the reference entity
        """
        tasks = []
        for (subj, ref, task) in self._reg.keys():
            if ref != reference:
                continue
            if subject_filter is not None and subj != subject_filter:
                continue
            if task_prefix is not None and not str(task).startswith(task_prefix):
                continue
            tasks.append(task)
        return tasks

    def list_available_templates(
        self,
        subject: Optional[EntityClass] = None,
        reference: Optional[EntityClass] = None,
        task_category: Optional[str] = None
    ) -> List[Tuple[EntityClass, EntityClass, TaskType, str]]:
        """List available templates with descriptions, optionally filtered.
        
        This method provides a comprehensive view of all registered templates
        with their descriptions, useful for browsing and discovery.
        
        Args:
            subject: Optional filter by subject entity
            reference: Optional filter by reference entity
            task_category: Optional filter by task category (e.g., "grasp", "place")
            
        Returns:
            List of (subject, reference, task, description) tuples
        """
        results = []
        for (subj, ref, task), entries in self._templates.items():
            if (subject is None or subj == subject) and \
               (reference is None or ref == reference) and \
               (task_category is None or task.category.value == task_category):
                for entry in entries:
                    results.append((subj, ref, task, entry['description']))
        return results

    def get_template_info(
        self,
        subject: EntityClass,
        reference: EntityClass,
        task: TaskType
    ) -> List[Tuple[str, str]]:
        """Get template names and descriptions for a specific combination.
        
        Args:
            subject: The entity performing the action
            reference: The entity being acted upon
            task: The type of task being performed
            
        Returns:
            List of (name, description) tuples for available templates
            
        Raises:
            KeyError: If no templates are registered for the given combination
        """
        key = (subject, reference, task)
        if key not in self._templates:
            raise KeyError(f"No templates registered for {key}")
        
        return [(entry['template'].name, entry['description']) 
                for entry in self._templates[key]]
