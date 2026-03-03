"""Tests for TSR primitive parser."""

import numpy as np
from numpy import pi
from unittest import TestCase
import tempfile
import os

from tsr.primitives import (
    parse_point,
    parse_line,
    parse_plane,
    parse_box,
    parse_ring,
    parse_disk,
    parse_cylinder,
    parse_shell,
    parse_sphere,
    parse_raw,
    parse_position,
    parse_template,
    load_template_yaml,
    load_template_file,
    deg2rad,
)
from tsr.tsr import TSR


class TestPositionPrimitives(TestCase):
    """Test position primitive parsers."""

    def test_parse_point(self):
        """Test point primitive — returns all-zero Bw (offset goes to T_ref_tsr)."""
        params = {'x': 0.1, 'y': 0.2, 'z': 0.3}
        Bw = parse_point(params)

        self.assertEqual(Bw.shape, (6, 2))
        np.testing.assert_array_equal(Bw, np.zeros((6, 2)))

    def test_parse_point_defaults(self):
        """Test point with defaults."""
        Bw = parse_point({})
        np.testing.assert_array_equal(Bw, np.zeros((6, 2)))

    def test_parse_line_z(self):
        """Test line along z-axis."""
        params = {'axis': 'z', 'range': [-0.1, 0.1]}
        Bw = parse_line(params)

        self.assertEqual(Bw[2, 0], -0.1)
        self.assertEqual(Bw[2, 1], 0.1)
        self.assertEqual(Bw[0, 0], 0)  # x fixed
        self.assertEqual(Bw[1, 0], 0)  # y fixed

    def test_parse_line_x(self):
        """Test line along x-axis."""
        params = {'axis': 'x', 'range': [0, 0.5], 'y': 0.1, 'z': 0.2}
        Bw = parse_line(params)

        self.assertEqual(Bw[0, 0], 0)
        self.assertEqual(Bw[0, 1], 0.5)
        self.assertEqual(Bw[1, 0], 0.1)
        self.assertEqual(Bw[2, 0], 0.2)

    def test_parse_plane_xy(self):
        """Test plane in xy."""
        params = {'x': [-0.1, 0.1], 'y': [-0.2, 0.2], 'z': 0.5}
        Bw = parse_plane(params)

        self.assertEqual(Bw[0, 0], -0.1)
        self.assertEqual(Bw[0, 1], 0.1)
        self.assertEqual(Bw[1, 0], -0.2)
        self.assertEqual(Bw[1, 1], 0.2)
        self.assertEqual(Bw[2, 0], 0.5)
        self.assertEqual(Bw[2, 1], 0.5)

    def test_parse_box(self):
        """Test box primitive."""
        params = {
            'x': [-0.1, 0.1],
            'y': [-0.2, 0.2],
            'z': [0, 0.3]
        }
        Bw = parse_box(params)

        self.assertEqual(Bw[0, 0], -0.1)
        self.assertEqual(Bw[0, 1], 0.1)
        self.assertEqual(Bw[1, 0], -0.2)
        self.assertEqual(Bw[1, 1], 0.2)
        self.assertEqual(Bw[2, 0], 0)
        self.assertEqual(Bw[2, 1], 0.3)

    def test_parse_ring_z(self):
        """Test ring around z-axis — radius NOT in Bw (goes to Tw_e)."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'angle': [0, 360],
            'height': 0.05
        }
        Bw = parse_ring(params)

        self.assertEqual(Bw[0, 0], 0.0)    # x = 0 (radius goes to Tw_e)
        self.assertEqual(Bw[0, 1], 0.0)
        self.assertEqual(Bw[1, 0], 0)      # y = 0
        self.assertEqual(Bw[2, 0], 0.05)   # z = height
        self.assertAlmostEqual(Bw[5, 0], 0)          # yaw min
        self.assertAlmostEqual(Bw[5, 1], 2 * pi)     # yaw max

    def test_parse_ring_partial(self):
        """Test partial ring (arc)."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'angle': [30, 330]
        }
        Bw = parse_ring(params)

        self.assertAlmostEqual(Bw[5, 0], deg2rad(30))
        self.assertAlmostEqual(Bw[5, 1], deg2rad(330))

    def test_parse_disk(self):
        """Test disk primitive — half-thickness in Bw, midpoint goes to Tw_e."""
        params = {
            'axis': 'z',
            'radius': [0.03, 0.05],
            'angle': [0, 360]
        }
        Bw = parse_disk(params)

        half_thickness = (0.05 - 0.03) / 2  # 0.01
        self.assertAlmostEqual(Bw[0, 0], -half_thickness)
        self.assertAlmostEqual(Bw[0, 1], half_thickness)
        self.assertAlmostEqual(Bw[5, 1], 2 * pi)

    def test_parse_cylinder_z(self):
        """Test cylinder around z-axis."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'height': [0.02, 0.08],
            'angle': [0, 360]
        }
        Bw = parse_cylinder(params)

        # Cylinder radius goes into Tw_e standoff, not Bw position
        self.assertEqual(Bw[0, 0], 0.0)    # x = 0 (at axis)
        self.assertEqual(Bw[0, 1], 0.0)
        self.assertEqual(Bw[2, 0], 0.02)   # height range
        self.assertEqual(Bw[2, 1], 0.08)
        self.assertAlmostEqual(Bw[5, 1], 2 * pi)  # full yaw

    def test_parse_cylinder_partial(self):
        """Test cylinder with partial angle (avoiding region)."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'height': [0.02, 0.08],
            'angle': [30, 330]
        }
        Bw = parse_cylinder(params)

        self.assertAlmostEqual(Bw[5, 0], deg2rad(30))
        self.assertAlmostEqual(Bw[5, 1], deg2rad(330))

    def test_parse_shell(self):
        """Test shell (thick cylinder)."""
        params = {
            'axis': 'z',
            'radius': [0.03, 0.05],
            'height': [0.02, 0.08],
            'angle': [0, 360]
        }
        Bw = parse_shell(params)

        # Shell: radius midpoint goes to Tw_e standoff, half-thickness to Bw
        half_thickness = (0.05 - 0.03) / 2  # 0.01
        self.assertAlmostEqual(Bw[0, 0], -half_thickness)
        self.assertAlmostEqual(Bw[0, 1], half_thickness)
        self.assertEqual(Bw[2, 0], 0.02)  # height range
        self.assertEqual(Bw[2, 1], 0.08)

    def test_parse_sphere(self):
        """Test sphere primitive — radius NOT in Bw (goes to Tw_e)."""
        params = {
            'radius': 0.1,
            'pitch': [-90, 90],
            'yaw': [0, 360]
        }
        Bw = parse_sphere(params)

        self.assertEqual(Bw[0, 0], 0.0)  # radius goes to Tw_e
        self.assertAlmostEqual(Bw[4, 0], -pi/2)  # pitch
        self.assertAlmostEqual(Bw[4, 1], pi/2)
        self.assertAlmostEqual(Bw[5, 0], 0)      # yaw
        self.assertAlmostEqual(Bw[5, 1], 2*pi)

    def test_parse_raw(self):
        """Test raw DOF specification."""
        params = {
            'x': 0.1,
            'y': [-0.1, 0.1],
            'z': 0.2,
            'roll': 0,
            'pitch': [-10, 10],
            'yaw': [0, 180]
        }
        Bw = parse_raw(params)

        self.assertEqual(Bw[0, 0], 0.1)
        self.assertEqual(Bw[1, 0], -0.1)
        self.assertEqual(Bw[1, 1], 0.1)
        self.assertAlmostEqual(Bw[4, 0], deg2rad(-10))
        self.assertAlmostEqual(Bw[5, 1], pi)


class TestParsePosition(TestCase):
    """Test the unified position parser."""

    def test_dispatch_point(self):
        """Test dispatch to point parser — Bw is all zeros."""
        position = {'type': 'point', 'x': 0.1, 'y': 0.2, 'z': 0.3}
        Bw = parse_position(position)
        np.testing.assert_array_equal(Bw, np.zeros((6, 2)))

    def test_dispatch_cylinder(self):
        """Test dispatch to cylinder parser."""
        position = {
            'type': 'cylinder',
            'axis': 'z',
            'radius': 0.04,
            'height': [0.02, 0.08],
            'angle': [0, 360]
        }
        Bw = parse_position(position)
        self.assertEqual(Bw[0, 0], 0.0)  # radius goes to Tw_e, not Bw

    def test_unknown_type(self):
        """Test error on unknown type."""
        with self.assertRaises(ValueError):
            parse_position({'type': 'unknown'})


class TestParseTemplate(TestCase):
    """Test full template parsing."""

    def test_simple_template(self):
        """Test parsing a simple template."""
        spec = {
            'name': 'Test Grasp',
            'task': 'grasp',
            'subject': 'gripper',
            'reference': 'mug',
            'position': {
                'type': 'cylinder',
                'axis': 'z',
                'radius': 0.04,
                'height': [0.02, 0.08],
                'angle': [0, 360]
            },
            'orientation': {
                'approach': 'radial'
            },
            'standoff': 0.05
        }

        result = parse_template(spec)

        self.assertEqual(result.name, 'Test Grasp')
        self.assertEqual(result.task, 'grasp')
        self.assertEqual(result.subject, 'gripper')
        self.assertEqual(result.reference, 'mug')
        self.assertEqual(result.Bw.shape, (6, 2))
        self.assertEqual(result.Tw_e.shape, (4, 4))

    def test_yaml_loading(self):
        """Test loading from YAML string."""
        yaml_str = """
name: Side grasp mug
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.02, 0.08]
  angle: [30, 330]

orientation:
  approach: radial

standoff: 0.05

gripper:
  aperture: 0.06
"""
        result = load_template_yaml(yaml_str)

        self.assertEqual(result.name, 'Side grasp mug')
        self.assertEqual(result.Bw[0, 0], 0.0)  # radius goes to Tw_e standoff
        self.assertAlmostEqual(result.Bw[5, 0], deg2rad(30))  # angle min
        self.assertEqual(result.gripper['aperture'], 0.06)

    def test_top_grasp_template(self):
        """Test top grasp template."""
        yaml_str = """
name: Top grasp mug
task: grasp
subject: gripper
reference: mug

position:
  type: point
  x: 0
  y: 0
  z: 0.12

orientation:
  approach: -z

standoff: 0.05
"""
        result = load_template_yaml(yaml_str)

        self.assertEqual(result.name, 'Top grasp mug')
        # Point offset goes to T_ref_tsr, Bw is zero
        np.testing.assert_array_equal(result.Bw[0:3, :], np.zeros((3, 2)))
        self.assertAlmostEqual(result.T_ref_tsr[2, 3], 0.12)

    def test_placement_template(self):
        """Test placement template."""
        yaml_str = """
name: Place on table
task: place
subject: mug
reference: table

position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0.02

standoff: 0
"""
        result = load_template_yaml(yaml_str)

        self.assertEqual(result.task, 'place')
        self.assertEqual(result.Bw[0, 0], -0.1)
        self.assertEqual(result.Bw[0, 1], 0.1)
        self.assertEqual(result.Bw[2, 0], 0.02)

    def test_reference_frame_field(self):
        """Test reference_frame field for placement templates."""
        yaml_str = """
name: Mug on Table
task: place
subject: mug
reference: table
reference_frame: bottom

position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0

standoff: 0
"""
        result = load_template_yaml(yaml_str)

        self.assertEqual(result.reference_frame, 'bottom')
        self.assertEqual(result.task, 'place')

    def test_reference_frame_default_none(self):
        """Test reference_frame defaults to None when not specified."""
        yaml_str = """
name: Side Grasp
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.02, 0.08]

standoff: 0.05
"""
        result = load_template_yaml(yaml_str)

        self.assertIsNone(result.reference_frame)


class TestOrientationFreedoms(TestCase):
    """Test orientation freedom parsing."""

    def test_yaw_freedom(self):
        """Test yaw freedom in placement."""
        yaml_str = """
name: Place with yaw tolerance
task: place
subject: mug
reference: table

position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0.02

orientation:
  approach: -z
  yaw: [-45, 45]

standoff: 0
"""
        result = load_template_yaml(yaml_str)

        # Check yaw bounds are set correctly
        self.assertAlmostEqual(result.Bw[5, 0], deg2rad(-45))
        self.assertAlmostEqual(result.Bw[5, 1], deg2rad(45))

    def test_free_yaw(self):
        """Test free yaw specification."""
        yaml_str = """
name: Free yaw grasp
task: grasp
subject: gripper
reference: mug

position:
  type: point
  x: 0
  y: 0
  z: 0.1

orientation:
  approach: -z
  yaw: free

standoff: 0.05
"""
        result = load_template_yaml(yaml_str)

        # Check yaw is full range
        self.assertAlmostEqual(result.Bw[5, 0], -pi)
        self.assertAlmostEqual(result.Bw[5, 1], pi)

    def test_roll_freedom(self):
        """Test roll freedom."""
        yaml_str = """
name: Roll tolerance grasp
task: grasp
subject: gripper
reference: cylinder

position:
  type: ring
  axis: z
  radius: 0.04
  angle: [0, 360]

orientation:
  approach: radial
  roll: [-15, 15]

standoff: 0.05
"""
        result = load_template_yaml(yaml_str)

        # Check roll bounds
        self.assertAlmostEqual(result.Bw[3, 0], deg2rad(-15))
        self.assertAlmostEqual(result.Bw[3, 1], deg2rad(15))


class TestCylinderAxes(TestCase):
    """Test cylinder primitives around different axes."""

    def test_cylinder_around_x(self):
        """Test cylinder around x-axis."""
        params = {
            'axis': 'x',
            'radius': 0.04,
            'height': [-0.1, 0.1],
            'angle': [0, 360]
        }
        Bw = parse_cylinder(params)

        # For x-axis cylinder: y=0 (at axis), x=height, roll varies
        self.assertEqual(Bw[1, 0], 0.0)   # y = 0 (radius goes to Tw_e)
        self.assertEqual(Bw[0, 0], -0.1)  # x = height range
        self.assertEqual(Bw[0, 1], 0.1)
        self.assertAlmostEqual(Bw[3, 1], 2 * pi)  # roll varies

    def test_cylinder_around_y(self):
        """Test cylinder around y-axis."""
        params = {
            'axis': 'y',
            'radius': 0.04,
            'height': [-0.1, 0.1],
            'angle': [0, 360]
        }
        Bw = parse_cylinder(params)

        # For y-axis cylinder: x=0 (at axis), y=height, pitch varies
        self.assertEqual(Bw[0, 0], 0.0)   # x = 0 (radius goes to Tw_e)
        self.assertEqual(Bw[1, 0], -0.1)  # y = height range
        self.assertAlmostEqual(Bw[4, 1], 2 * pi)  # pitch varies


class TestIntegration(TestCase):
    """Integration tests: parsed templates → working TSRs."""

    def test_cylinder_grasp_creates_valid_tsr(self):
        """Test that a cylinder grasp template creates a working TSR."""
        yaml_str = """
name: Side grasp
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.02, 0.08]
  angle: [0, 360]

orientation:
  approach: radial

standoff: 0.05
"""
        parsed = load_template_yaml(yaml_str)

        # Create TSR from parsed components
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        # Sample should produce valid 4x4 transform
        pose = tsr.sample()
        self.assertEqual(pose.shape, (4, 4))

        # Sample multiple times - all should be valid
        for _ in range(10):
            pose = tsr.sample()
            self.assertEqual(pose.shape, (4, 4))
            # Check it's a valid transform (rotation matrix is orthonormal)
            R = pose[0:3, 0:3]
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))

    def test_plane_placement_creates_valid_tsr(self):
        """Test that a plane placement template creates a working TSR."""
        yaml_str = """
name: Table placement
task: place
subject: mug
reference: table

position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0.02

orientation:
  approach: -z
  yaw: [-45, 45]

standoff: 0
"""
        parsed = load_template_yaml(yaml_str)
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        # Sample and verify bounds
        for _ in range(10):
            xyzrpy = tsr.sample_xyzrpy()
            # x should be in [-0.1, 0.1]
            self.assertGreaterEqual(xyzrpy[0], -0.1 - 1e-6)
            self.assertLessEqual(xyzrpy[0], 0.1 + 1e-6)
            # y should be in [-0.1, 0.1]
            self.assertGreaterEqual(xyzrpy[1], -0.1 - 1e-6)
            self.assertLessEqual(xyzrpy[1], 0.1 + 1e-6)
            # z should be fixed at 0.02
            self.assertAlmostEqual(xyzrpy[2], 0.02, places=5)

    def test_point_grasp_creates_valid_tsr(self):
        """Test that a point grasp template creates a working TSR."""
        yaml_str = """
name: Top grasp
task: grasp
subject: gripper
reference: mug

position:
  type: point
  x: 0
  y: 0
  z: 0.12

orientation:
  approach: -z
  yaw: free

standoff: 0.05
"""
        parsed = load_template_yaml(yaml_str)
        # Point offset is in T_ref_tsr, must compose with T0_w
        T0_w = np.eye(4) @ parsed.T_ref_tsr
        tsr = TSR(T0_w=T0_w, Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        # Bw position should be zero (all freedom is in yaw only)
        np.testing.assert_array_equal(parsed.Bw[0:3, :], np.zeros((3, 2)))

        # Sample and verify the TSR produces valid transforms
        for _ in range(10):
            pose = tsr.sample()
            self.assertEqual(pose.shape, (4, 4))
            R = pose[0:3, 0:3]
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
            # Yaw should vary (free)
            xyzrpy = tsr.sample_xyzrpy()
            self.assertGreaterEqual(xyzrpy[5], -pi - 1e-6)
            self.assertLessEqual(xyzrpy[5], pi + 1e-6)

    def test_ring_grasp_samples_on_circle(self):
        """Test that ring primitive samples gripper positions on a circle."""
        yaml_str = """
name: Ring grasp
task: grasp
subject: gripper
reference: cylinder

position:
  type: ring
  axis: z
  radius: 0.05
  height: 0.1
  angle: [0, 360]

orientation:
  approach: radial

standoff: 0.03
"""
        parsed = load_template_yaml(yaml_str)
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        # Radius (0.05) + standoff (0.03) = 0.08 total offset in Tw_e
        expected_dist = 0.05 + 0.03

        # Sample and verify gripper positions lie on a circle
        for _ in range(20):
            pose = tsr.sample()
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            # Distance from z-axis should be radius + standoff
            dist = np.sqrt(x**2 + y**2)
            self.assertAlmostEqual(dist, expected_dist, places=4)
            # Height should be fixed at 0.1
            self.assertAlmostEqual(z, 0.1, places=5)

    def test_tsr_distance_with_parsed_template(self):
        """Test distance calculation with parsed template."""
        yaml_str = """
name: Placement
task: place
subject: object
reference: table

position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0

standoff: 0
"""
        parsed = load_template_yaml(yaml_str)
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        # Transform at origin should be contained
        contained = np.eye(4)
        self.assertTrue(tsr.contains(contained))

        # Transform outside bounds should not be contained
        outside = np.eye(4)
        outside[0, 3] = 0.5  # x = 0.5, outside [-0.1, 0.1]
        self.assertFalse(tsr.contains(outside))

    def test_load_template_file(self):
        """Test loading template from file."""
        yaml_content = """
name: Test Template
task: grasp
subject: gripper
reference: object

position:
  type: point
  x: 0.1
  y: 0.2
  z: 0.3

standoff: 0.05
"""
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            parsed = load_template_file(temp_path)
            self.assertEqual(parsed.name, 'Test Template')
            # Point offset goes to T_ref_tsr, Bw is zero
            np.testing.assert_array_equal(parsed.Bw[0:3, :], np.zeros((3, 2)))
            self.assertAlmostEqual(parsed.T_ref_tsr[0, 3], 0.1)
            self.assertAlmostEqual(parsed.T_ref_tsr[1, 3], 0.2)
            self.assertAlmostEqual(parsed.T_ref_tsr[2, 3], 0.3)
        finally:
            os.unlink(temp_path)


class TestErrorHandling(TestCase):
    """Test error handling for invalid inputs."""

    def test_unknown_position_type(self):
        """Test error on unknown position type."""
        with self.assertRaises(ValueError) as ctx:
            parse_position({'type': 'invalid_type'})
        self.assertIn('invalid_type', str(ctx.exception))

    def test_unknown_approach_direction(self):
        """Test error on unknown approach direction."""
        from tsr.primitives import approach_to_rotation
        with self.assertRaises(ValueError) as ctx:
            approach_to_rotation('invalid_direction')
        self.assertIn('invalid_direction', str(ctx.exception))

    def test_missing_position_defaults_to_point(self):
        """Test that missing position defaults to point at origin."""
        spec = {
            'name': 'Test',
            'task': 'grasp',
            'subject': 'gripper',
            'reference': 'object'
        }
        parsed = parse_template(spec)
        # Should default to point at origin
        np.testing.assert_array_equal(parsed.Bw[0:3, :], np.zeros((3, 2)))

    def test_empty_yaml(self):
        """Test parsing empty/minimal YAML."""
        yaml_str = """
name: Minimal
"""
        parsed = load_template_yaml(yaml_str)
        self.assertEqual(parsed.name, 'Minimal')
        self.assertEqual(parsed.task, '')

    def test_invalid_yaml_syntax(self):
        """Test error on invalid YAML syntax."""
        yaml_str = """
name: Bad YAML
position:
  type: cylinder
  radius: [unclosed bracket
"""
        with self.assertRaises(Exception):
            load_template_yaml(yaml_str)


class TestEdgeCases(TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_radius_ring(self):
        """Test ring with zero radius (degenerates to point on axis)."""
        params = {
            'axis': 'z',
            'radius': 0,
            'angle': [0, 360]
        }
        Bw = parse_ring(params)
        self.assertEqual(Bw[0, 0], 0)  # radius = 0

    def test_negative_height_range(self):
        """Test cylinder with negative height range."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'height': [-0.1, -0.02],
            'angle': [0, 360]
        }
        Bw = parse_cylinder(params)
        self.assertEqual(Bw[2, 0], -0.1)
        self.assertEqual(Bw[2, 1], -0.02)

    def test_angle_greater_than_360(self):
        """Test angle range exceeding 360 degrees."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'angle': [0, 720]
        }
        Bw = parse_ring(params)
        # Should convert to radians (720 deg = 4*pi)
        self.assertAlmostEqual(Bw[5, 1], 4 * pi)

    def test_negative_angles(self):
        """Test negative angle range."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'angle': [-180, 180]
        }
        Bw = parse_ring(params)
        self.assertAlmostEqual(Bw[5, 0], -pi)
        self.assertAlmostEqual(Bw[5, 1], pi)

    def test_single_value_as_range(self):
        """Test that single values work where ranges are expected."""
        params = {
            'axis': 'z',
            'radius': 0.04,
            'height': 0.05,  # Single value, not a range
            'angle': 90     # Single value, not a range
        }
        Bw = parse_cylinder(params)
        # Height should be fixed
        self.assertEqual(Bw[2, 0], 0.05)
        self.assertEqual(Bw[2, 1], 0.05)
        # Angle should be fixed
        self.assertAlmostEqual(Bw[5, 0], deg2rad(90))
        self.assertAlmostEqual(Bw[5, 1], deg2rad(90))

    def test_box_with_mixed_values(self):
        """Test box with mix of fixed and range values."""
        params = {
            'x': [-0.1, 0.1],  # range
            'y': 0,            # fixed
            'z': [0, 0.5]      # range
        }
        Bw = parse_box(params)
        self.assertEqual(Bw[0, 0], -0.1)
        self.assertEqual(Bw[0, 1], 0.1)
        self.assertEqual(Bw[1, 0], 0)
        self.assertEqual(Bw[1, 1], 0)
        self.assertEqual(Bw[2, 0], 0)
        self.assertEqual(Bw[2, 1], 0.5)

    def test_sphere_partial_coverage(self):
        """Test sphere with partial pitch/yaw coverage."""
        params = {
            'radius': 0.1,
            'pitch': [0, 45],     # Only upper hemisphere, partial
            'yaw': [0, 180]       # Only half rotation
        }
        Bw = parse_sphere(params)
        self.assertEqual(Bw[0, 0], 0.0)  # radius goes to Tw_e
        self.assertAlmostEqual(Bw[4, 0], 0)
        self.assertAlmostEqual(Bw[4, 1], deg2rad(45))
        self.assertAlmostEqual(Bw[5, 0], 0)
        self.assertAlmostEqual(Bw[5, 1], pi)

    def test_standoff_affects_tw_e(self):
        """Test that standoff creates correct offset in Tw_e."""
        yaml_str = """
name: Test
task: grasp
subject: gripper
reference: object

position:
  type: point
  z: 0.1

orientation:
  approach: -z

standoff: 0.05
"""
        parsed = load_template_yaml(yaml_str)
        # Standoff should create offset in Tw_e
        # For -z approach, offset should be in +z direction
        self.assertEqual(parsed.Tw_e.shape, (4, 4))
        # The translation should reflect the standoff
        self.assertAlmostEqual(np.linalg.norm(parsed.Tw_e[0:3, 3]), 0.05, places=4)


class TestGeometricCorrectness(TestCase):
    """Tests that verify correct geometric behavior of primitives."""

    def test_ring_positions_trace_circle(self):
        """Ring with radial approach: sampled positions should trace a circle."""
        yaml_str = """
name: Ring test
task: grasp
subject: gripper
reference: object
position:
  type: ring
  axis: z
  radius: 0.08
  height: 0
  angle: [0, 360]
orientation:
  approach: radial
standoff: 0.05
"""
        parsed = load_template_yaml(yaml_str)
        expected_radius = 0.08 + 0.05  # radius + standoff

        # Verify Tw_e has the combined offset
        self.assertAlmostEqual(parsed.Tw_e[0, 3], expected_radius, places=6)

        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)
        for _ in range(20):
            pose = tsr.sample()
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            dist = np.sqrt(x**2 + y**2)
            self.assertAlmostEqual(dist, expected_radius, places=4)
            self.assertAlmostEqual(z, 0.0, places=5)

    def test_ring_nonradial_approach(self):
        """Ring with non-radial approach: radius and standoff separate correctly."""
        yaml_str = """
name: Ring axial approach
task: grasp
subject: gripper
reference: object
position:
  type: ring
  axis: z
  radius: 0.05
  height: 0
  angle: [0, 360]
orientation:
  approach: -z
standoff: 0.03
"""
        parsed = load_template_yaml(yaml_str)
        # Standoff goes along -z approach → Tw_e z offset
        # Radius goes radially → Tw_e x offset
        self.assertAlmostEqual(parsed.Tw_e[0, 3], 0.05, places=6)  # radius in x
        self.assertAlmostEqual(parsed.Tw_e[2, 3], 0.03, places=6)  # standoff in z

    def test_sphere_positions_on_sphere_surface(self):
        """Sphere: sampled positions should lie on a sphere surface."""
        yaml_str = """
name: Sphere test
task: handover
subject: object
reference: human
position:
  type: sphere
  radius: 0.45
  pitch: [-30, 30]
  yaw: [-45, 45]
orientation:
  approach: -x
standoff: 0
"""
        parsed = load_template_yaml(yaml_str)
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        for _ in range(20):
            pose = tsr.sample()
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            dist = np.sqrt(x**2 + y**2 + z**2)
            self.assertAlmostEqual(dist, 0.45, places=3)

    def test_disk_positions_within_disk(self):
        """Disk: sampled positions should be within the disk area."""
        yaml_str = """
name: Disk test
task: place
subject: cup
reference: coaster
position:
  type: disk
  axis: z
  radius: [0, 0.02]
  height: 0
  angle: [0, 360]
orientation:
  approach: -z
  yaw: free
standoff: 0
"""
        parsed = load_template_yaml(yaml_str)
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        for _ in range(20):
            pose = tsr.sample()
            x, y = pose[0, 3], pose[1, 3]
            dist = np.sqrt(x**2 + y**2)
            # Should be within disk radius (0.02) with tolerance
            self.assertLessEqual(dist, 0.02 + 1e-6)

    def test_point_offset_in_T_ref_tsr(self):
        """Point: offset should be in T_ref_tsr, Bw should be all zeros."""
        yaml_str = """
name: Point test
task: grasp
subject: gripper
reference: mug
position:
  type: point
  x: 0.06
  y: 0
  z: 0.05
orientation:
  approach: -x
standoff: 0.03
"""
        parsed = load_template_yaml(yaml_str)

        # Bw should be all zeros for point
        np.testing.assert_array_equal(parsed.Bw, np.zeros((6, 2)))

        # T_ref_tsr should have the point offset
        self.assertAlmostEqual(parsed.T_ref_tsr[0, 3], 0.06)
        self.assertAlmostEqual(parsed.T_ref_tsr[1, 3], 0.0)
        self.assertAlmostEqual(parsed.T_ref_tsr[2, 3], 0.05)

        # Tw_e should have only the standoff
        self.assertAlmostEqual(np.linalg.norm(parsed.Tw_e[0:3, 3]), 0.03, places=4)

    def test_ring_x_axis(self):
        """Ring around x-axis: positions trace circle in yz plane."""
        yaml_str = """
name: Valve ring
task: grasp
subject: gripper
reference: valve
position:
  type: ring
  axis: x
  radius: 0.08
  height: 0
  angle: [0, 360]
orientation:
  approach: radial
standoff: 0.03
"""
        parsed = load_template_yaml(yaml_str)
        expected_radius = 0.08 + 0.03
        tsr = TSR(T0_w=np.eye(4), Tw_e=parsed.Tw_e, Bw=parsed.Bw)

        for _ in range(20):
            pose = tsr.sample()
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            # Circle in yz plane
            dist = np.sqrt(y**2 + z**2)
            self.assertAlmostEqual(dist, expected_radius, places=4)
            self.assertAlmostEqual(x, 0.0, places=5)
