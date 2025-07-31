#!/usr/bin/env python
"""
Tests for utility functions in tsr.core.utils.
"""

import numpy as np
import unittest
from numpy import pi

from tsr.core.utils import wrap_to_interval, geodesic_error, geodesic_distance


class TestWrapToInterval(unittest.TestCase):
    """Test the wrap_to_interval function."""
    
    def test_basic_wrapping(self):
        """Test basic angle wrapping."""
        angles = np.array([0, pi/2, pi, 3*pi/2, 2*pi])
        wrapped = wrap_to_interval(angles)
        
        # The function wraps to [-pi, pi] interval starting at -pi
        # So pi gets wrapped to -pi, and 2*pi gets wrapped to 0
        expected = np.array([0, pi/2, -pi, -pi/2, 0])
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_custom_lower_bound(self):
        """Test wrapping with custom lower bound."""
        angles = np.array([0, pi/2, pi, 3*pi/2, 2*pi])
        lower = np.array([0, 0, 0, 0, 0])
        wrapped = wrap_to_interval(angles, lower)
        
        expected = np.array([0, pi/2, pi, 3*pi/2, 0])
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_negative_angles(self):
        """Test wrapping of negative angles."""
        angles = np.array([-pi, -pi/2, 0, pi/2, pi])
        wrapped = wrap_to_interval(angles)
        
        # The function wraps to [-pi, pi] interval starting at -pi
        # So pi gets wrapped to -pi
        expected = np.array([-pi, -pi/2, 0, pi/2, -pi])
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_large_angles(self):
        """Test wrapping of angles larger than 2*pi."""
        angles = np.array([3*pi, 4*pi, 5*pi])
        wrapped = wrap_to_interval(angles)
        
        # The function wraps to [-pi, pi] interval
        expected = np.array([-pi, 0, -pi])
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_single_angle(self):
        """Test wrapping of a single angle."""
        angle = np.array([3*pi])
        wrapped = wrap_to_interval(angle)
        
        # The function wraps to [-pi, pi] interval
        expected = np.array([-pi])
        np.testing.assert_array_almost_equal(wrapped, expected)
    
    def test_empty_array(self):
        """Test wrapping of empty array."""
        angles = np.array([])
        wrapped = wrap_to_interval(angles)
        
        self.assertEqual(len(wrapped), 0)


class TestGeodesicError(unittest.TestCase):
    """Test the geodesic_error function."""
    
    def test_identical_transforms(self):
        """Test error between identical transforms."""
        t1 = np.eye(4)
        t2 = np.eye(4)
        
        error = geodesic_error(t1, t2)
        
        expected = np.array([0, 0, 0, 0])
        np.testing.assert_array_almost_equal(error, expected)
    
    def test_translation_only(self):
        """Test error with translation only."""
        t1 = np.eye(4)
        t2 = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        
        error = geodesic_error(t1, t2)
        
        # Translation error should be [1, 2, 3]
        np.testing.assert_array_almost_equal(error[:3], [1, 2, 3])
        # Rotation error should be 0
        self.assertAlmostEqual(error[3], 0)
    
    def test_rotation_only(self):
        """Test error with rotation only."""
        t1 = np.eye(4)
        # 90 degree rotation around z-axis
        t2 = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        error = geodesic_error(t1, t2)
        
        # Translation error should be 0
        np.testing.assert_array_almost_equal(error[:3], [0, 0, 0])
        # Rotation error should be non-zero
        self.assertGreater(error[3], 0)
    
    def test_combined_transform(self):
        """Test error with both translation and rotation."""
        t1 = np.eye(4)
        t2 = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        
        error = geodesic_error(t1, t2)
        
        # Translation error should be [1, 2, 3]
        np.testing.assert_array_almost_equal(error[:3], [1, 2, 3])
        # Rotation error should be non-zero
        self.assertGreater(error[3], 0)
    
    def test_reverse_direction(self):
        """Test that geodesic_error is not symmetric."""
        t1 = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        t2 = np.eye(4)
        
        error_forward = geodesic_error(t1, t2)
        error_reverse = geodesic_error(t2, t1)
        
        # The errors should be different
        self.assertFalse(np.allclose(error_forward, error_reverse))


class TestGeodesicDistance(unittest.TestCase):
    """Test the geodesic_distance function."""
    
    def test_identical_transforms(self):
        """Test distance between identical transforms."""
        t1 = np.eye(4)
        t2 = np.eye(4)
        
        distance = geodesic_distance(t1, t2)
        
        self.assertAlmostEqual(distance, 0)
    
    def test_translation_only(self):
        """Test distance with translation only."""
        t1 = np.eye(4)
        t2 = np.array([
            [1, 0, 0, 3],
            [0, 1, 0, 4],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        distance = geodesic_distance(t1, t2)
        
        # Distance should be sqrt(3^2 + 4^2 + 0^2) = 5
        self.assertAlmostEqual(distance, 5.0)
    
    def test_rotation_only(self):
        """Test distance with rotation only."""
        t1 = np.eye(4)
        # 90 degree rotation around z-axis
        t2 = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        distance = geodesic_distance(t1, t2)
        
        # Distance should be non-zero due to rotation
        self.assertGreater(distance, 0)
    
    def test_custom_weight(self):
        """Test distance with custom weight parameter."""
        t1 = np.eye(4)
        t2 = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        distance_r1 = geodesic_distance(t1, t2, r=1.0)
        distance_r2 = geodesic_distance(t1, t2, r=2.0)
        
        # Distance with r=2 should be different from r=1
        self.assertNotEqual(distance_r1, distance_r2)
    
    def test_combined_transform(self):
        """Test distance with both translation and rotation."""
        t1 = np.eye(4)
        t2 = np.array([
            [0, -1, 0, 3],
            [1, 0, 0, 4],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        distance = geodesic_distance(t1, t2)
        
        # Distance should be greater than translation-only distance
        translation_distance = np.sqrt(3**2 + 4**2)
        self.assertGreater(distance, translation_distance)


if __name__ == '__main__':
    unittest.main() 