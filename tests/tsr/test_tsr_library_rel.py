#!/usr/bin/env python
"""
Tests for TSRLibraryRelational functionality.

Tests the relational TSR library for registering and querying TSR generators.
"""

import unittest
import numpy as np
from typing import List
from tsr.tsr_library_rel import TSRLibraryRelational
from tsr.schema import TaskCategory, TaskType, EntityClass
from tsr.core.tsr_template import TSRTemplate


class TestTSRLibraryRelational(unittest.TestCase):
    """Test TSRLibraryRelational functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.library = TSRLibraryRelational()
        
        # Create test TSR templates
        self.template1 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [0, 0],      # x: fixed
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [-np.pi, np.pi]  # yaw: full rotation
            ])
        )
        
        self.template2 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],  # x: small range
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [0, 0]       # yaw: fixed
            ])
        )
        
        # Create test generator functions
        def grasp_generator(T_ref_world: np.ndarray) -> List[TSRTemplate]:
            """Generate grasp templates."""
            return [self.template1, self.template2]
        
        def place_generator(T_ref_world: np.ndarray) -> List[TSRTemplate]:
            """Generate place templates."""
            return [self.template1]
        
        self.grasp_generator = grasp_generator
        self.place_generator = place_generator
    
    def test_library_creation(self):
        """Test TSRLibraryRelational creation."""
        self.assertIsInstance(self.library, TSRLibraryRelational)
    
    def test_register_and_query(self):
        """Test registering and querying TSR generators."""
        # Register a generator
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=self.grasp_generator
        )
        
        # Query the generator
        T_ref_world = np.eye(4)
        tsrs = self.library.query(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            T_ref_world=T_ref_world
        )
        
        # Should return list of TSRs
        self.assertIsInstance(tsrs, list)
        self.assertEqual(len(tsrs), 2)  # Two templates from grasp_generator
        
        # Each should be a TSR
        for tsr in tsrs:
            from tsr.core.tsr import TSR
            self.assertIsInstance(tsr, TSR)
    
    def test_query_unregistered(self):
        """Test querying unregistered generator."""
        with self.assertRaises(KeyError):
            self.library.query(
                subject=EntityClass.GENERIC_GRIPPER,
                reference=EntityClass.MUG,
                task=TaskType(TaskCategory.GRASP, "side"),
                T_ref_world=np.eye(4)
            )
    
    def test_multiple_registrations(self):
        """Test registering multiple generators."""
        # Register multiple generators
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=self.grasp_generator
        )
        
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.PLACE, "on"),
            generator=self.place_generator
        )
        
        # Query both
        T_ref_world = np.eye(4)
        
        grasp_tsrs = self.library.query(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            T_ref_world=T_ref_world
        )
        
        place_tsrs = self.library.query(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.PLACE, "on"),
            T_ref_world=T_ref_world
        )
        
        # Should return different numbers of TSRs
        self.assertEqual(len(grasp_tsrs), 2)
        self.assertEqual(len(place_tsrs), 1)
    
    def test_list_tasks_for_reference(self):
        """Test listing tasks for a reference entity."""
        # Register generators for different tasks
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=self.grasp_generator
        )
        
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.PLACE, "on"),
            generator=self.place_generator
        )
        
        self.library.register(
            subject=EntityClass.ROBOTIQ_2F140,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "top"),
            generator=self.grasp_generator
        )
        
        # List tasks for MUG reference
        tasks = self.library.list_tasks_for_reference(EntityClass.MUG)
        
        # Should return all tasks for MUG
        expected_tasks = {
            TaskType(TaskCategory.GRASP, "side"),
            TaskType(TaskCategory.PLACE, "on"),
            TaskType(TaskCategory.GRASP, "top")
        }
        self.assertEqual(set(tasks), expected_tasks)
    
    def test_list_tasks_with_subject_filter(self):
        """Test listing tasks with subject filter."""
        # Register generators for different subjects
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=self.grasp_generator
        )
        
        self.library.register(
            subject=EntityClass.ROBOTIQ_2F140,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "top"),
            generator=self.grasp_generator
        )
        
        # List tasks for MUG with GENERIC_GRIPPER filter
        tasks = self.library.list_tasks_for_reference(
            EntityClass.MUG,
            subject_filter=EntityClass.GENERIC_GRIPPER
        )
        
        # Should only return tasks for GENERIC_GRIPPER
        expected_tasks = {TaskType(TaskCategory.GRASP, "side")}
        self.assertEqual(set(tasks), expected_tasks)
    
    def test_list_tasks_with_prefix_filter(self):
        """Test listing tasks with prefix filter."""
        # Register generators for different task categories
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=self.grasp_generator
        )
        
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.PLACE, "on"),
            generator=self.place_generator
        )
        
        # List tasks with "grasp" prefix
        tasks = self.library.list_tasks_for_reference(
            EntityClass.MUG,
            task_prefix="grasp"
        )
        
        # Should only return grasp tasks
        expected_tasks = {TaskType(TaskCategory.GRASP, "side")}
        self.assertEqual(set(tasks), expected_tasks)
    
    def test_generator_with_reference_pose(self):
        """Test that generators receive the reference pose correctly."""
        received_pose = None
        
        def test_generator(T_ref_world: np.ndarray) -> List[TSRTemplate]:
            nonlocal received_pose
            received_pose = T_ref_world.copy()
            return [self.template1]
        
        self.library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=test_generator
        )
        
        # Query with specific pose
        test_pose = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        self.library.query(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            T_ref_world=test_pose
        )
        
        # Generator should have received the pose
        self.assertIsNotNone(received_pose)
        np.testing.assert_array_almost_equal(received_pose, test_pose)


class TestTSRLibraryRelationalExamples(unittest.TestCase):
    """Test TSRLibraryRelational with realistic examples."""
    
    def test_grasp_and_place_scenario(self):
        """Test a complete grasp and place scenario."""
        library = TSRLibraryRelational()
        
        # Create realistic templates
        def mug_grasp_generator(T_ref_world: np.ndarray) -> List[TSRTemplate]:
            """Generate grasp templates for mug."""
            # Side grasp template
            side_template = TSRTemplate(
                T_ref_tsr=np.eye(4),
                Tw_e=np.array([
                    [0, 0, 1, -0.05],  # Approach from -z
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.05],
                    [0, 0, 0, 1]
                ]),
                Bw=np.array([
                    [0, 0],           # x: fixed
                    [0, 0],           # y: fixed
                    [-0.01, 0.01],    # z: small tolerance
                    [0, 0],           # roll: fixed
                    [0, 0],           # pitch: fixed
                    [-np.pi, np.pi]   # yaw: full rotation
                ])
            )
            return [side_template]
        
        def mug_place_generator(T_ref_world: np.ndarray) -> List[TSRTemplate]:
            """Generate place templates for mug."""
            # Place on table template
            place_template = TSRTemplate(
                T_ref_tsr=np.eye(4),
                Tw_e=np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.02],  # Slightly above surface
                    [0, 0, 0, 1]
                ]),
                Bw=np.array([
                    [-0.1, 0.1],      # x: allow sliding
                    [-0.1, 0.1],      # y: allow sliding
                    [0, 0],           # z: fixed height
                    [0, 0],           # roll: keep level
                    [0, 0],           # pitch: keep level
                    [-np.pi/4, np.pi/4]  # yaw: some rotation
                ])
            )
            return [place_template]
        
        # Register generators
        library.register(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            generator=mug_grasp_generator
        )
        
        library.register(
            subject=EntityClass.MUG,
            reference=EntityClass.TABLE,
            task=TaskType(TaskCategory.PLACE, "on"),
            generator=mug_place_generator
        )
        
        # Test grasp query
        mug_pose = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        grasp_tsrs = library.query(
            subject=EntityClass.GENERIC_GRIPPER,
            reference=EntityClass.MUG,
            task=TaskType(TaskCategory.GRASP, "side"),
            T_ref_world=mug_pose
        )
        
        self.assertEqual(len(grasp_tsrs), 1)
        
        # Test place query
        table_pose = np.eye(4)
        
        place_tsrs = library.query(
            subject=EntityClass.MUG,
            reference=EntityClass.TABLE,
            task=TaskType(TaskCategory.PLACE, "on"),
            T_ref_world=table_pose
        )
        
        self.assertEqual(len(place_tsrs), 1)
        
        # Test task discovery
        mug_tasks = library.list_tasks_for_reference(EntityClass.MUG)
        self.assertEqual(len(mug_tasks), 1)
        self.assertIn(TaskType(TaskCategory.GRASP, "side"), mug_tasks)
        
        table_tasks = library.list_tasks_for_reference(EntityClass.TABLE)
        self.assertEqual(len(table_tasks), 1)
        self.assertIn(TaskType(TaskCategory.PLACE, "on"), table_tasks)


if __name__ == '__main__':
    unittest.main()

