#!/usr/bin/env python
"""
Tests for TSR schema components.

Tests the TaskCategory, TaskType, and EntityClass enums and their functionality.
"""

import unittest
from tsr.schema import TaskCategory, TaskType, EntityClass


class TestTaskCategory(unittest.TestCase):
    """Test TaskCategory enum functionality."""
    
    def test_task_category_values(self):
        """Test that all expected task categories exist."""
        expected_categories = {
            'grasp', 'place', 'discard', 'insert', 
            'inspect', 'push', 'actuate'
        }
        
        actual_categories = {cat.value for cat in TaskCategory}
        self.assertEqual(actual_categories, expected_categories)
    
    def test_task_category_comparison(self):
        """Test task category comparison operations."""
        self.assertEqual(TaskCategory.GRASP, TaskCategory.GRASP)
        self.assertNotEqual(TaskCategory.GRASP, TaskCategory.PLACE)
        
        # Test string comparison
        self.assertEqual(TaskCategory.GRASP, "grasp")
        self.assertEqual("grasp", TaskCategory.GRASP)
    
    def test_task_category_string_representation(self):
        """Test string representation of task categories."""
        self.assertEqual(str(TaskCategory.GRASP), "TaskCategory.GRASP")
        self.assertEqual(repr(TaskCategory.PLACE), "<TaskCategory.PLACE: 'place'>")


class TestTaskType(unittest.TestCase):
    """Test TaskType dataclass functionality."""
    
    def test_task_type_creation(self):
        """Test creating TaskType instances."""
        task = TaskType(TaskCategory.GRASP, "side")
        self.assertEqual(task.category, TaskCategory.GRASP)
        self.assertEqual(task.variant, "side")
    
    def test_task_type_string_representation(self):
        """Test string representation of TaskType."""
        task = TaskType(TaskCategory.GRASP, "side")
        self.assertEqual(str(task), "grasp/side")
        
        task2 = TaskType(TaskCategory.PLACE, "on")
        self.assertEqual(str(task2), "place/on")
    
    def test_task_type_from_str(self):
        """Test creating TaskType from string."""
        task = TaskType.from_str("grasp/side")
        self.assertEqual(task.category, TaskCategory.GRASP)
        self.assertEqual(task.variant, "side")
        
        task2 = TaskType.from_str("place/on")
        self.assertEqual(task2.category, TaskCategory.PLACE)
        self.assertEqual(task2.variant, "on")
    
    def test_task_type_from_str_invalid(self):
        """Test TaskType.from_str with invalid strings."""
        invalid_strings = [
            "grasp",           # Missing variant
            "",               # Empty string
            "/side",          # Missing category
        ]
        
        for invalid_str in invalid_strings:
            with self.assertRaises(Exception):  # Any exception is fine
                TaskType.from_str(invalid_str)
    
    def test_task_type_equality(self):
        """Test TaskType equality."""
        task1 = TaskType(TaskCategory.GRASP, "side")
        task2 = TaskType(TaskCategory.GRASP, "side")
        task3 = TaskType(TaskCategory.GRASP, "top")
        
        self.assertEqual(task1, task2)
        self.assertNotEqual(task1, task3)
        self.assertNotEqual(task1, TaskCategory.GRASP)


class TestEntityClass(unittest.TestCase):
    """Test EntityClass enum functionality."""
    
    def test_entity_class_values(self):
        """Test that all expected entity classes exist."""
        expected_entities = {
            # Grippers/tools
            'generic_gripper', 'robotiq_2f140', 'suction',
            # Objects/fixtures
            'mug', 'bin', 'plate', 'box', 'table', 'shelf', 'valve'
        }
        
        actual_entities = {entity.value for entity in EntityClass}
        self.assertEqual(actual_entities, expected_entities)
    
    def test_entity_class_comparison(self):
        """Test entity class comparison operations."""
        self.assertEqual(EntityClass.GENERIC_GRIPPER, EntityClass.GENERIC_GRIPPER)
        self.assertNotEqual(EntityClass.GENERIC_GRIPPER, EntityClass.MUG)
        
        # Test string comparison
        self.assertEqual(EntityClass.GENERIC_GRIPPER, "generic_gripper")
        self.assertEqual("generic_gripper", EntityClass.GENERIC_GRIPPER)
    
    def test_entity_class_string_representation(self):
        """Test string representation of entity classes."""
        self.assertEqual(str(EntityClass.GENERIC_GRIPPER), "EntityClass.GENERIC_GRIPPER")
        self.assertEqual(repr(EntityClass.MUG), "<EntityClass.MUG: 'mug'>")
    
    def test_entity_class_categorization(self):
        """Test that we can categorize entities."""
        grippers = {
            EntityClass.GENERIC_GRIPPER,
            EntityClass.ROBOTIQ_2F140,
            EntityClass.SUCTION
        }
        
        objects = {
            EntityClass.MUG,
            EntityClass.BIN,
            EntityClass.PLATE,
            EntityClass.BOX,
            EntityClass.TABLE,
            EntityClass.SHELF,
            EntityClass.VALVE
        }
        
        # Verify all entities are categorized
        all_entities = set(EntityClass)
        self.assertEqual(all_entities, grippers | objects)


class TestSchemaIntegration(unittest.TestCase):
    """Test integration between schema components."""
    
    def test_task_type_with_entity_classes(self):
        """Test creating task types for different entity combinations."""
        # Grasp tasks
        grasp_side = TaskType(TaskCategory.GRASP, "side")
        grasp_top = TaskType(TaskCategory.GRASP, "top")
        
        # Place tasks
        place_on = TaskType(TaskCategory.PLACE, "on")
        place_in = TaskType(TaskCategory.PLACE, "in")
        
        # Verify they work with entity classes
        self.assertEqual(str(grasp_side), "grasp/side")
        self.assertEqual(str(place_on), "place/on")
    
    def test_schema_consistency(self):
        """Test that schema components work together consistently."""
        # Create a realistic task scenario
        gripper = EntityClass.ROBOTIQ_2F140
        object_entity = EntityClass.MUG
        task = TaskType(TaskCategory.GRASP, "side")
        
        # Verify all components work together
        self.assertEqual(gripper, "robotiq_2f140")
        self.assertEqual(object_entity, "mug")
        self.assertEqual(task.category, TaskCategory.GRASP)
        self.assertEqual(task.variant, "side")
    
    def test_schema_validation(self):
        """Test that schema components validate correctly."""
        # Valid combinations
        valid_combinations = [
            (EntityClass.GENERIC_GRIPPER, EntityClass.MUG, TaskType(TaskCategory.GRASP, "side")),
            (EntityClass.ROBOTIQ_2F140, EntityClass.BOX, TaskType(TaskCategory.PLACE, "on")),
            (EntityClass.SUCTION, EntityClass.PLATE, TaskType(TaskCategory.INSPECT, "top")),
        ]
        
        for subject, reference, task in valid_combinations:
            # These should not raise any exceptions
            self.assertIsInstance(subject, EntityClass)
            self.assertIsInstance(reference, EntityClass)
            self.assertIsInstance(task, TaskType)
            self.assertIsInstance(task.category, TaskCategory)


if __name__ == '__main__':
    unittest.main()
