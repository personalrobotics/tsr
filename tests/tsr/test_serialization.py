#!/usr/bin/env python
"""
Tests for TSR and TSRChain serialization methods.

Tests the to_dict, from_dict, to_json, from_json, to_yaml, and from_yaml methods
for both TSR and TSRChain classes.
"""

import json
import numpy as np
import unittest
import yaml
from numpy import pi

from tsr.core.tsr import TSR
from tsr.core.tsr_chain import TSRChain


class TestTSRSerialization(unittest.TestCase):
    """Test TSR serialization methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test TSR
        self.T0_w = np.array([
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        self.Tw_e = np.array([
            [0, 0, 1, 0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.1],
            [0, 0, 0, 1]
        ])
        
        self.Bw = np.array([
            [-0.01, 0.01],  # x bounds
            [-0.01, 0.01],  # y bounds
            [-0.01, 0.01],  # z bounds
            [-pi/4, pi/4],   # roll bounds
            [-pi/4, pi/4],   # pitch bounds
            [-pi/2, pi/2]    # yaw bounds
        ])
        
        self.tsr = TSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
    
    def test_to_dict(self):
        """Test TSR.to_dict() method."""
        result = self.tsr.to_dict()
        
        # Check structure
        self.assertIsInstance(result, dict)
        self.assertIn('T0_w', result)
        self.assertIn('Tw_e', result)
        self.assertIn('Bw', result)
        
        # Check data types
        self.assertIsInstance(result['T0_w'], list)
        self.assertIsInstance(result['Tw_e'], list)
        self.assertIsInstance(result['Bw'], list)
        
        # Check values
        np.testing.assert_array_almost_equal(
            np.array(result['T0_w']), self.T0_w
        )
        np.testing.assert_array_almost_equal(
            np.array(result['Tw_e']), self.Tw_e
        )
        np.testing.assert_array_almost_equal(
            np.array(result['Bw']), self.Bw
        )
    
    def test_from_dict(self):
        """Test TSR.from_dict() method."""
        # Create dictionary representation
        data = {
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist()
        }
        
        # Reconstruct TSR
        reconstructed = TSR.from_dict(data)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_dict_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip preserves the TSR."""
        # Convert to dict and back
        data = self.tsr.to_dict()
        reconstructed = TSR.from_dict(data)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_to_json(self):
        """Test TSR.to_json() method."""
        result = self.tsr.to_json()
        
        # Check that it's valid JSON
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        
        # Check structure
        self.assertIn('T0_w', parsed)
        self.assertIn('Tw_e', parsed)
        self.assertIn('Bw', parsed)
        
        # Check values
        np.testing.assert_array_almost_equal(
            np.array(parsed['T0_w']), self.T0_w
        )
        np.testing.assert_array_almost_equal(
            np.array(parsed['Tw_e']), self.Tw_e
        )
        np.testing.assert_array_almost_equal(
            np.array(parsed['Bw']), self.Bw
        )
    
    def test_from_json(self):
        """Test TSR.from_json() method."""
        # Create JSON string
        json_str = json.dumps({
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist()
        })
        
        # Reconstruct TSR
        reconstructed = TSR.from_json(json_str)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_json_roundtrip(self):
        """Test that to_json -> from_json roundtrip preserves the TSR."""
        # Convert to JSON and back
        json_str = self.tsr.to_json()
        reconstructed = TSR.from_json(json_str)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_to_yaml(self):
        """Test TSR.to_yaml() method."""
        result = self.tsr.to_yaml()
        
        # Check that it's valid YAML
        self.assertIsInstance(result, str)
        parsed = yaml.safe_load(result)
        
        # Check structure
        self.assertIn('T0_w', parsed)
        self.assertIn('Tw_e', parsed)
        self.assertIn('Bw', parsed)
        
        # Check values
        np.testing.assert_array_almost_equal(
            np.array(parsed['T0_w']), self.T0_w
        )
        np.testing.assert_array_almost_equal(
            np.array(parsed['Tw_e']), self.Tw_e
        )
        np.testing.assert_array_almost_equal(
            np.array(parsed['Bw']), self.Bw
        )
    
    def test_from_yaml(self):
        """Test TSR.from_yaml() method."""
        # Create YAML string
        yaml_str = yaml.dump({
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist()
        })
        
        # Reconstruct TSR
        reconstructed = TSR.from_yaml(yaml_str)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_yaml_roundtrip(self):
        """Test that to_yaml -> from_yaml roundtrip preserves the TSR."""
        # Convert to YAML and back
        yaml_str = self.tsr.to_yaml()
        reconstructed = TSR.from_yaml(yaml_str)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )
    
    def test_cross_format_roundtrip(self):
        """Test roundtrip through different formats."""
        # TSR -> dict -> JSON -> YAML -> TSR
        data = self.tsr.to_dict()
        json_str = json.dumps(data)
        yaml_str = yaml.dump(json.loads(json_str))
        reconstructed = TSR.from_yaml(yaml_str)
        
        # Check that all attributes match
        np.testing.assert_array_almost_equal(
            reconstructed.T0_w, self.tsr.T0_w
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Tw_e, self.tsr.Tw_e
        )
        np.testing.assert_array_almost_equal(
            reconstructed.Bw, self.tsr.Bw
        )


class TestTSRChainSerialization(unittest.TestCase):
    """Test TSRChain serialization methods."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test TSRs
        self.tsr1 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, 0.1],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [-0.01, 0.01],
                [-0.01, 0.01],
                [-0.01, 0.01],
                [-pi/6, pi/6],
                [-pi/6, pi/6],
                [-pi/3, pi/3]
            ])
        )

        self.tsr2 = TSR(
            T0_w=np.array([
                [1, 0, 0, 0.2],
                [0, 1, 0, 0.1],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1]
            ]),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.02, 0.02],
                [-0.02, 0.02],
                [-0.02, 0.02],
                [-pi/4, pi/4],
                [-pi/4, pi/4],
                [-pi/2, pi/2]
            ])
        )

        # Create TSRChain
        self.chain = TSRChain(TSRs=[self.tsr1, self.tsr2])

    def test_to_dict(self):
        """Test TSRChain.to_dict() method."""
        result = self.chain.to_dict()

        # Check structure
        self.assertIsInstance(result, dict)
        self.assertIn('tsrs', result)
        self.assertIsInstance(result['tsrs'], list)
        self.assertEqual(len(result['tsrs']), 2)

        # Check TSRs
        for tsr_data in result['tsrs']:
            self.assertIsInstance(tsr_data, dict)
            self.assertIn('T0_w', tsr_data)
            self.assertIn('Tw_e', tsr_data)
            self.assertIn('Bw', tsr_data)

    def test_from_dict(self):
        """Test TSRChain.from_dict() method."""
        # Create dictionary representation
        data = {'tsrs': [self.tsr1.to_dict(), self.tsr2.to_dict()]}

        # Reconstruct TSRChain
        reconstructed = TSRChain.from_dict(data)
        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

        # Check TSRs
        for original, reconstructed_tsr in zip(self.chain.TSRs, reconstructed.TSRs):
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.T0_w, original.T0_w
            )
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.Tw_e, original.Tw_e
            )
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.Bw, original.Bw
            )

    def test_dict_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip preserves the TSRChain."""
        data = self.chain.to_dict()
        reconstructed = TSRChain.from_dict(data)

        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

        for original, reconstructed_tsr in zip(self.chain.TSRs, reconstructed.TSRs):
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.T0_w, original.T0_w
            )
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.Tw_e, original.Tw_e
            )
            np.testing.assert_array_almost_equal(
                reconstructed_tsr.Bw, original.Bw
            )

    def test_to_json(self):
        """Test TSRChain.to_json() method."""
        result = self.chain.to_json()

        self.assertIsInstance(result, str)
        parsed = json.loads(result)

        self.assertIn('tsrs', parsed)
        self.assertEqual(len(parsed['tsrs']), 2)

    def test_from_json(self):
        """Test TSRChain.from_json() method."""
        json_str = json.dumps({'tsrs': [self.tsr1.to_dict(), self.tsr2.to_dict()]})

        reconstructed = TSRChain.from_json(json_str)
        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

    def test_json_roundtrip(self):
        """Test that to_json -> from_json roundtrip preserves the TSRChain."""
        json_str = self.chain.to_json()
        reconstructed = TSRChain.from_json(json_str)

        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

    def test_to_yaml(self):
        """Test TSRChain.to_yaml() method."""
        result = self.chain.to_yaml()

        self.assertIsInstance(result, str)
        parsed = yaml.safe_load(result)

        self.assertIn('tsrs', parsed)
        self.assertEqual(len(parsed['tsrs']), 2)

    def test_from_yaml(self):
        """Test TSRChain.from_yaml() method."""
        yaml_str = yaml.dump({'tsrs': [self.tsr1.to_dict(), self.tsr2.to_dict()]})

        reconstructed = TSRChain.from_yaml(yaml_str)
        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

    def test_yaml_roundtrip(self):
        """Test that to_yaml -> from_yaml roundtrip preserves the TSRChain."""
        yaml_str = self.chain.to_yaml()
        reconstructed = TSRChain.from_yaml(yaml_str)

        self.assertEqual(len(reconstructed.TSRs), len(self.chain.TSRs))

    def test_empty_chain(self):
        """Test serialization of empty TSRChain."""
        empty_chain = TSRChain()

        # Test dict roundtrip
        data = empty_chain.to_dict()
        reconstructed = TSRChain.from_dict(data)
        self.assertEqual(len(reconstructed.TSRs), 0)

        # Test JSON roundtrip
        json_str = empty_chain.to_json()
        reconstructed = TSRChain.from_json(json_str)
        self.assertEqual(len(reconstructed.TSRs), 0)

        # Test YAML roundtrip
        yaml_str = empty_chain.to_yaml()
        reconstructed = TSRChain.from_yaml(yaml_str)
        self.assertEqual(len(reconstructed.TSRs), 0)


if __name__ == '__main__':
    unittest.main() 