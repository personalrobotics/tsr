#!/usr/bin/env python
"""
Tests for TSRTemplate functionality.

Tests the TSRTemplate class for scene-agnostic TSR definitions.
"""

import unittest
import numpy as np
import yaml
import dataclasses
from numpy import pi
from tsr.core.tsr_template import TSRTemplate
from tsr.core.tsr import TSR
from tsr.schema import EntityClass, TaskCategory, TaskType


class TestTSRTemplate(unittest.TestCase):
    """Test TSRTemplate creation and instantiation."""

    def setUp(self):
        """Set up test fixtures."""
        self.T_ref_tsr = np.eye(4)
        self.Tw_e = np.array([
            [0, 0, 1, -0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.05],
            [0, 0, 0, 1]
        ])
        self.Bw = np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position
            [-0.01, 0.01],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-np.pi, np.pi]   # yaw: full rotation
        ])
        
        self.template = TSRTemplate(
            T_ref_tsr=self.T_ref_tsr,
            Tw_e=self.Tw_e,
            Bw=self.Bw,
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side",
            name="Cylinder Side Grasp",
            description="Grasp a cylindrical object from the side with 5cm approach distance"
        )

    def test_tsr_template_creation(self):
        """Test TSRTemplate creation with semantic context."""
        self.assertEqual(self.template.subject_entity, EntityClass.GENERIC_GRIPPER)
        self.assertEqual(self.template.reference_entity, EntityClass.MUG)
        self.assertEqual(self.template.task_category, TaskCategory.GRASP)
        self.assertEqual(self.template.variant, "side")
        self.assertEqual(self.template.name, "Cylinder Side Grasp")
        self.assertEqual(self.template.description, "Grasp a cylindrical object from the side with 5cm approach distance")
        
        np.testing.assert_array_equal(self.template.T_ref_tsr, self.T_ref_tsr)
        np.testing.assert_array_equal(self.template.Tw_e, self.Tw_e)
        np.testing.assert_array_equal(self.template.Bw, self.Bw)

    def test_tsr_template_instantiation(self):
        """Test TSRTemplate instantiation."""
        T_ref_world = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        tsr = self.template.instantiate(T_ref_world)
        
        # Check that the instantiated TSR has the correct T0_w
        expected_T0_w = T_ref_world @ self.T_ref_tsr
        np.testing.assert_array_equal(tsr.T0_w, expected_T0_w)
        
        # Check that Tw_e and Bw are preserved
        np.testing.assert_array_equal(tsr.Tw_e, self.Tw_e)
        np.testing.assert_array_equal(tsr.Bw, self.Bw)

    def test_tsr_template_default_values(self):
        """Test TSRTemplate creation with default values."""
        template = TSRTemplate(
            T_ref_tsr=self.T_ref_tsr,
            Tw_e=self.Tw_e,
            Bw=self.Bw,
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side"
        )
        
        self.assertEqual(template.name, "")
        self.assertEqual(template.description, "")

    def test_tsr_template_immutability(self):
        """Test that TSRTemplate is immutable."""
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.template.name = "New Name"


class TestTSRTemplateSerialization(unittest.TestCase):
    """Test TSRTemplate serialization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0],
                [0, 0],
                [-0.01, 0.01],
                [0, 0],
                [0, 0],
                [-np.pi, np.pi]
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side",
            name="Test Template",
            description="Test description"
        )

    def test_to_dict(self):
        """Test TSRTemplate.to_dict() method."""
        result = self.template.to_dict()
        
        self.assertEqual(result['name'], "Test Template")
        self.assertEqual(result['description'], "Test description")
        self.assertEqual(result['subject_entity'], "generic_gripper")
        self.assertEqual(result['reference_entity'], "mug")
        self.assertEqual(result['task_category'], "grasp")
        self.assertEqual(result['variant'], "side")
        
        # Check that arrays are converted to lists
        self.assertIsInstance(result['T_ref_tsr'], list)
        self.assertIsInstance(result['Tw_e'], list)
        self.assertIsInstance(result['Bw'], list)
        
        # Check array contents
        np.testing.assert_array_equal(np.array(result['T_ref_tsr']), self.template.T_ref_tsr)
        np.testing.assert_array_equal(np.array(result['Tw_e']), self.template.Tw_e)
        np.testing.assert_array_equal(np.array(result['Bw']), self.template.Bw)

    def test_from_dict(self):
        """Test TSRTemplate.from_dict() method."""
        data = {
            'name': 'Test Template',
            'description': 'Test description',
            'subject_entity': 'generic_gripper',
            'reference_entity': 'mug',
            'task_category': 'grasp',
            'variant': 'side',
            'T_ref_tsr': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            'Tw_e': [[0, 0, 1, -0.05], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]],
            'Bw': [[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-3.14159, 3.14159]]
        }
        
        reconstructed = TSRTemplate.from_dict(data)
        
        self.assertEqual(reconstructed.name, "Test Template")
        self.assertEqual(reconstructed.description, "Test description")
        self.assertEqual(reconstructed.subject_entity, EntityClass.GENERIC_GRIPPER)
        self.assertEqual(reconstructed.reference_entity, EntityClass.MUG)
        self.assertEqual(reconstructed.task_category, TaskCategory.GRASP)
        self.assertEqual(reconstructed.variant, "side")
        
        np.testing.assert_array_equal(reconstructed.T_ref_tsr, self.template.T_ref_tsr)
        np.testing.assert_array_equal(reconstructed.Tw_e, self.template.Tw_e)
        np.testing.assert_array_almost_equal(reconstructed.Bw, self.template.Bw, decimal=5)

    def test_dict_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip preserves the TSRTemplate."""
        data = self.template.to_dict()
        reconstructed = TSRTemplate.from_dict(data)
        
        self.assertEqual(reconstructed.name, self.template.name)
        self.assertEqual(reconstructed.description, self.template.description)
        self.assertEqual(reconstructed.subject_entity, self.template.subject_entity)
        self.assertEqual(reconstructed.reference_entity, self.template.reference_entity)
        self.assertEqual(reconstructed.task_category, self.template.task_category)
        self.assertEqual(reconstructed.variant, self.template.variant)
        
        np.testing.assert_array_equal(reconstructed.T_ref_tsr, self.template.T_ref_tsr)
        np.testing.assert_array_equal(reconstructed.Tw_e, self.template.Tw_e)
        np.testing.assert_array_almost_equal(reconstructed.Bw, self.template.Bw, decimal=5)

    def test_to_yaml(self):
        """Test TSRTemplate.to_yaml() method."""
        result = self.template.to_yaml()
        
        # Check that it's valid YAML
        parsed = yaml.safe_load(result)
        self.assertEqual(parsed['name'], "Test Template")
        self.assertEqual(parsed['subject_entity'], "generic_gripper")
        self.assertEqual(parsed['task_category'], "grasp")

    def test_from_yaml(self):
        """Test TSRTemplate.from_yaml() method."""
        yaml_str = """
name: Test Template
description: Test description
subject_entity: generic_gripper
reference_entity: mug
task_category: grasp
variant: side
T_ref_tsr:
  - [1, 0, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 0]
  - [0, 0, 0, 1]
Tw_e:
  - [0, 0, 1, -0.05]
  - [1, 0, 0, 0]
  - [0, 1, 0, 0.05]
  - [0, 0, 0, 1]
Bw:
  - [0, 0]
  - [0, 0]
  - [-0.01, 0.01]
  - [0, 0]
  - [0, 0]
  - [-3.14159, 3.14159]
"""
        
        reconstructed = TSRTemplate.from_yaml(yaml_str)
        
        self.assertEqual(reconstructed.name, "Test Template")
        self.assertEqual(reconstructed.description, "Test description")
        self.assertEqual(reconstructed.subject_entity, EntityClass.GENERIC_GRIPPER)
        self.assertEqual(reconstructed.reference_entity, EntityClass.MUG)
        self.assertEqual(reconstructed.task_category, TaskCategory.GRASP)
        self.assertEqual(reconstructed.variant, "side")

    def test_yaml_roundtrip(self):
        """Test that to_yaml -> from_yaml roundtrip preserves the TSRTemplate."""
        yaml_str = self.template.to_yaml()
        reconstructed = TSRTemplate.from_yaml(yaml_str)
        
        self.assertEqual(reconstructed.name, self.template.name)
        self.assertEqual(reconstructed.description, self.template.description)
        self.assertEqual(reconstructed.subject_entity, self.template.subject_entity)
        self.assertEqual(reconstructed.reference_entity, self.template.reference_entity)
        self.assertEqual(reconstructed.task_category, self.template.task_category)
        self.assertEqual(reconstructed.variant, self.template.variant)
        
        np.testing.assert_array_equal(reconstructed.T_ref_tsr, self.template.T_ref_tsr)
        np.testing.assert_array_equal(reconstructed.Tw_e, self.template.Tw_e)
        np.testing.assert_array_almost_equal(reconstructed.Bw, self.template.Bw, decimal=5)

    def test_cross_format_roundtrip(self):
        """Test cross-format roundtrip (dict -> YAML -> dict)."""
        data = self.template.to_dict()
        yaml_str = TSRTemplate.from_dict(data).to_yaml()
        reconstructed = TSRTemplate.from_yaml(yaml_str)
        
        self.assertEqual(reconstructed.name, self.template.name)
        self.assertEqual(reconstructed.description, self.template.description)
        self.assertEqual(reconstructed.subject_entity, self.template.subject_entity)
        self.assertEqual(reconstructed.reference_entity, self.template.reference_entity)
        self.assertEqual(reconstructed.task_category, self.template.task_category)
        self.assertEqual(reconstructed.variant, self.template.variant)

    def test_from_dict_missing_optional_fields(self):
        """Test from_dict with missing optional fields."""
        data = {
            'subject_entity': 'generic_gripper',
            'reference_entity': 'mug',
            'task_category': 'grasp',
            'variant': 'side',
            'T_ref_tsr': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            'Tw_e': [[0, 0, 1, -0.05], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]],
            'Bw': [[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-3.14159, 3.14159]]
        }
        
        reconstructed = TSRTemplate.from_dict(data)
        
        self.assertEqual(reconstructed.name, "")
        self.assertEqual(reconstructed.description, "")
        self.assertEqual(reconstructed.subject_entity, EntityClass.GENERIC_GRIPPER)
        self.assertEqual(reconstructed.reference_entity, EntityClass.MUG)
        self.assertEqual(reconstructed.task_category, TaskCategory.GRASP)
        self.assertEqual(reconstructed.variant, "side")


class TestTSRTemplateExamples(unittest.TestCase):
    """Test TSRTemplate with realistic examples."""

    def test_cylinder_grasp_template(self):
        """Test cylinder grasp template creation and instantiation."""
        template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
                [1, 0, 0, 0],      # x-axis perpendicular to cylinder
                [0, 1, 0, 0.05],   # y-axis along cylinder axis
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0],           # x: fixed position
                [0, 0],           # y: fixed position
                [-0.01, 0.01],    # z: small tolerance
                [0, 0],           # roll: fixed
                [0, 0],           # pitch: fixed
                [-np.pi, np.pi]   # yaw: full rotation
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side",
            name="Cylinder Side Grasp",
            description="Grasp a cylindrical object from the side with 5cm approach distance"
        )
        
        # Test instantiation
        cylinder_pose = np.array([
            [1, 0, 0, 0.5],  # Cylinder at x=0.5
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        tsr = template.instantiate(cylinder_pose)
        pose = tsr.sample()
        
        # Verify pose is a valid 4x4 homogeneous transform
        self.assertEqual(pose.shape, (4, 4))
        self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))  # Bottom row
        # Check rotation matrix properties
        R = pose[:3, :3]
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))  # Orthogonal
        self.assertTrue(np.allclose(np.linalg.det(R), 1.0))  # Determinant = 1

    def test_place_on_table_template(self):
        """Test place on table template creation and instantiation."""
        template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [1, 0, 0, 0],      # Object x-axis aligned with table
                [0, 1, 0, 0],      # Object y-axis aligned with table
                [0, 0, 1, 0.02],   # Object 2cm above table surface
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [-0.1, 0.1],       # x: allow sliding on table
                [-0.1, 0.1],       # y: allow sliding on table
                [0, 0],            # z: fixed height
                [0, 0],            # roll: keep level
                [0, 0],            # pitch: keep level
                [-np.pi/4, np.pi/4]  # yaw: allow some rotation
            ]),
            subject_entity=EntityClass.MUG,
            reference_entity=EntityClass.TABLE,
            task_category=TaskCategory.PLACE,
            variant="on",
            name="Table Placement",
            description="Place object on table surface with 2cm clearance"
        )
        
        # Test instantiation
        table_pose = np.eye(4)  # Table at world origin
        tsr = template.instantiate(table_pose)
        pose = tsr.sample()
        
        # Verify pose is a valid 4x4 homogeneous transform
        self.assertEqual(pose.shape, (4, 4))
        self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))  # Bottom row
        # Check rotation matrix properties
        R = pose[:3, :3]
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))  # Orthogonal
        self.assertTrue(np.allclose(np.linalg.det(R), 1.0))  # Determinant = 1


if __name__ == '__main__':
    unittest.main()
