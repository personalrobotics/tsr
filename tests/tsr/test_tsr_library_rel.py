"""Tests for TSRLibraryRelational class."""

import unittest
import numpy as np

from tsr.library import TSRLibraryRelational
from tsr.template import TSRTemplate
from tsr.schema import EntityClass, TaskCategory, TaskType


class TestTSRLibraryRelational(unittest.TestCase):
    """Test TSRLibraryRelational functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.library = TSRLibraryRelational()
        
        # Create test templates
        self.template1 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject="generic_gripper",
            reference="mug",
            task="grasp",
            variant="side",
            name="Side Grasp",
            description="Grasp mug from the side"
        )
        
        self.template2 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject="generic_gripper",
            reference="mug",
            task="grasp",
            variant="top",
            name="Top Grasp",
            description="Grasp mug from the top"
        )

    def test_register_template(self):
        """Test registering templates with descriptions."""
        # Register templates
        self.library.register_template(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            template=self.template1,
            description="Grasp mug from the side with 5cm approach"
        )
        
        self.library.register_template(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "top"),
            template=self.template2,
            description="Grasp mug from the top with vertical approach"
        )
        
        # Verify templates are registered
        templates = self.library.query_templates(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side")
        )
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0].name, "Side Grasp")

    def test_query_templates(self):
        """Test querying templates."""
        # Register templates
        self.library.register_template(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            self.template1,
            "Side grasp description"
        )

        templates = self.library.query_templates(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side")
        )
        self.assertEqual(len(templates), 1)
        self.assertIsInstance(templates[0], TSRTemplate)

    def test_query_templates_not_found(self):
        """Test querying non-existent templates."""
        with self.assertRaises(KeyError):
            self.library.query_templates(
                EntityClass.GENERIC_GRIPPER,
                EntityClass.MUG,
                TaskType(TaskCategory.GRASP, "nonexistent")
            )

    def test_multiple_templates_same_key(self):
        """Test registering multiple templates for the same key."""
        # Register multiple templates for the same combination
        self.library.register_template(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            self.template1,
            "First side grasp"
        )
        
        self.library.register_template(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            self.template2,
            "Second side grasp"
        )
        
        # Query should return both templates
        templates = self.library.query_templates(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side")
        )
        self.assertEqual(len(templates), 2)


class TestTSRLibraryRelationalGeneratorMode(unittest.TestCase):
    """Test TSRLibraryRelational in generator mode (existing functionality)."""

    def setUp(self):
        """Set up test fixtures."""
        self.library = TSRLibraryRelational()
        
        # Create a test template
        self.template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject="generic_gripper",
            reference="mug",
            task="grasp",
            variant="side"
        )

    def test_register_generator(self):
        """Test registering a generator function."""
        def generator(T_ref_world):
            return [self.template]
        
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=generator
        )

    def test_query_generator(self):
        """Test querying with a generator."""
        def generator(T_ref_world):
            return [self.template]
        
        self.library.register(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            generator
        )
        
        T_ref_world = np.eye(4)
        tsrs = self.library.query(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            T_ref_world
        )
        
        self.assertEqual(len(tsrs), 1)
        self.assertIsInstance(tsrs[0], object)  # CoreTSR

    def test_list_tasks_for_reference(self):
        """Test listing tasks for a reference entity."""
        def generator(T_ref_world):
            return [self.template]
        
        self.library.register(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            generator
        )
        
        self.library.register(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.PLACE, "on"),
            generator
        )
        
        tasks = self.library.list_tasks_for_reference(EntityClass.MUG)
        self.assertEqual(len(tasks), 2)
        
        task_strings = [str(task) for task in tasks]
        self.assertIn("grasp/side", task_strings)
        self.assertIn("place/on", task_strings)

    def test_list_tasks_with_filters(self):
        """Test listing tasks with filters."""
        def generator(T_ref_world):
            return [self.template]
        
        self.library.register(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            generator
        )
        
        self.library.register(
            EntityClass.ROBOTIQ_2F140,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "top"),
            generator
        )
        
        # Filter by subject
        tasks = self.library.list_tasks_for_reference(
            EntityClass.MUG,
            subject_filter=EntityClass.GENERIC_GRIPPER
        )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(str(tasks[0]), "grasp/side")
        
        # Filter by task prefix
        tasks = self.library.list_tasks_for_reference(
            EntityClass.MUG,
            task_prefix="grasp"
        )
        self.assertEqual(len(tasks), 2)


class TestTSRLibraryRelationalMixedMode(unittest.TestCase):
    """Test TSRLibraryRelational with both generator and template modes."""

    def setUp(self):
        """Set up test fixtures."""
        self.library = TSRLibraryRelational()
        
        self.template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject="generic_gripper",
            reference="mug",
            task="grasp",
            variant="side"
        )

    def test_generator_and_template_independence(self):
        """Test that generator and template registrations are independent."""
        # Register generator
        def generator(T_ref_world):
            return [self.template]
        
        self.library.register(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            generator
        )
        
        # Register template
        self.library.register_template(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            self.template,
            "Template description"
        )
        
        # Both should work independently
        T_ref_world = np.eye(4)
        
        # Query generator
        tsrs = self.library.query(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side"),
            T_ref_world
        )
        self.assertEqual(len(tsrs), 1)
        
        # Query templates
        templates = self.library.query_templates(
            EntityClass.GENERIC_GRIPPER,
            EntityClass.MUG,
            TaskType(TaskCategory.GRASP, "side")
        )
        self.assertEqual(len(templates), 1)


if __name__ == '__main__':
    unittest.main()
