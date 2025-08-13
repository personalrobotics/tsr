from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskCategory(str, Enum):
    """Controlled vocabulary for high-level manipulation tasks.
    
    This enum provides a standardized set of task categories that can be used
    across different robotics applications. Each category represents a fundamental
    manipulation action that can be performed on objects.
    
    Examples:
        >>> TaskCategory.GRASP
        TaskCategory.GRASP
        >>> str(TaskCategory.PLACE)
        'place'
        >>> TaskCategory.GRASP == "grasp"
        True
    """
    GRASP = "grasp"      # Pick up an object
    PLACE = "place"      # Put down an object
    DISCARD = "discard"  # Throw away an object
    INSERT = "insert"    # Insert object into receptacle
    INSPECT = "inspect"  # Examine object closely
    PUSH = "push"        # Push/move object
    ACTUATE = "actuate"  # Operate controls/mechanisms


@dataclass(frozen=True)
class TaskType:
    """Structured task type: controlled category + freeform variant.
    
    A TaskType combines a standardized TaskCategory with a specific variant
    that describes how the task should be performed. This provides both
    consistency (through the category) and flexibility (through the variant).
    
    Attributes:
        category: The standardized task category (e.g., GRASP, PLACE)
        variant: Freeform description of how to perform the task (e.g., "side", "on")
    
    Examples:
        >>> task = TaskType(TaskCategory.GRASP, "side")
        >>> str(task)
        'grasp/side'
        >>> TaskType.from_str("place/on")
        TaskType(category=TaskCategory.PLACE, variant='on')
    """
    category: TaskCategory
    variant: str  # e.g., "side", "on", "opening", "valve/turn_ccw"

    def __str__(self) -> str:
        """Return string representation as 'category/variant'."""
        return f"{self.category.value}/{self.variant}"

    @staticmethod
    def from_str(s: str) -> "TaskType":
        """Create TaskType from string representation.
        
        Args:
            s: String in format "category/variant"
            
        Returns:
            TaskType instance
            
        Raises:
            ValueError: If string format is invalid
            
        Examples:
            >>> TaskType.from_str("grasp/side")
            TaskType(category=TaskCategory.GRASP, variant='side')
            >>> TaskType.from_str("place/on")
            TaskType(category=TaskCategory.PLACE, variant='on')
        """
        parts = s.split("/", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid TaskType string: {s!r}")
        cat, var = parts
        return TaskType(TaskCategory(cat), var)


class EntityClass(str, Enum):
    """Unified scene entities (objects, fixtures, tools/grippers).
    
    This enum provides a standardized vocabulary for different types of
    entities that can appear in robotic manipulation scenarios. Entities
    are categorized into grippers/tools and objects/fixtures.
    
    Examples:
        >>> EntityClass.GENERIC_GRIPPER
        EntityClass.GENERIC_GRIPPER
        >>> str(EntityClass.MUG)
        'mug'
        >>> EntityClass.ROBOTIQ_2F140 == "robotiq_2f140"
        True
    """
    # Grippers/tools
    GENERIC_GRIPPER = "generic_gripper"  # Generic end-effector
    ROBOTIQ_2F140 = "robotiq_2f140"      # Robotiq 2F-140 parallel gripper
    SUCTION = "suction"                  # Suction cup end-effector
    
    # Objects/fixtures
    MUG = "mug"                          # Drinking vessel
    BIN = "bin"                          # Container for objects
    PLATE = "plate"                      # Flat serving dish
    BOX = "box"                          # Rectangular container
    TABLE = "table"                      # Flat surface for placement
    SHELF = "shelf"                      # Horizontal storage surface
    VALVE = "valve"                      # Mechanical control device
