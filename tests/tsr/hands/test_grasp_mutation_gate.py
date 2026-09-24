#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Mutation gate: the property matrix must FAIL on a corrupted generator (issue #73).

A suite that never fails proves nothing, so every major bug class named in #73 is
injected deliberately and the matrix's own check functions must reject it:

* **black-box mutants** wrap the real factory output (approach sign, face origin,
  axis mapping, object extent, clearance term, reach), so they survive refactors;
* **code mutants** patch the real generator (`_resolve_clearance`,
  `_infeasibility_reason`, `_usable_depths`), so a corrupted implementation --
  not just a corrupted result -- is shown to be caught.

Detection is required for every factory where the mutation is geometrically
meaningful. The one exception is listed in ``KNOWN_SYMMETRIC``: it is a true
symmetry of the primitive, so the "mutated" templates are still correct.
"""

import unittest
from typing import Dict, List
from unittest import mock

import numpy as np

from tsr.hands import GripperBase, ParallelJawGripper

from ._grasp_matrix import (
    BLACK_BOX_MUTANTS,
    CORPUS_GRIPPERS,
    FACTORIES,
    GraspCase,
    corpus,
    detected,
    oversize_dims,
    soundness_failures,
)

# (mutant, factory) pairs that must NOT be detected, because the mutation is a
# symmetry of the primitive rather than a defect.
KNOWN_SYMMETRIC = {
    ("axis_mapping", "grasp_sphere"): "a sphere is invariant under rotation, so permuting the TSR frame axes is exact",
}


class TestCorpus(unittest.TestCase):
    def test_every_factory_is_covered_and_feasible(self):
        seen = set()
        for case in corpus():
            templates = case.run()
            self.assertTrue(templates, f"{case.factory} corpus case is empty: {case}")
            seen.add(case.factory)
        self.assertEqual(seen, set(FACTORIES))

    def test_unmutated_corpus_passes(self):
        # Attributability: every detection below is caused by its mutation.
        for case in corpus():
            self.assertFalse(detected(case, case.run()), case)


class TestBlackBoxMutants(unittest.TestCase):
    """Each corrupted result is rejected for every factory it can affect."""

    def test_detection_matrix(self):
        undetected = []
        for name, mutate in BLACK_BOX_MUTANTS.items():
            hits = {}
            for case in corpus():
                base = case.run()
                hits[case.factory] = hits.get(case.factory, False) or detected(case, mutate(case, base))
            for factory, hit in sorted(hits.items()):
                expected = (name, factory) not in KNOWN_SYMMETRIC
                if hit != expected:
                    undetected.append(f"{name} on {factory}: detected={hit}, expected={expected}")
        self.assertEqual(undetected, [])

    def test_known_symmetric_pairs_are_still_sound(self):
        # The excluded pairs are excluded because they remain correct, not ignored.
        for (name, factory), _why in KNOWN_SYMMETRIC.items():
            for case in corpus():
                if case.factory != factory:
                    continue
                mutated = BLACK_BOX_MUTANTS[name](case, case.run())
                self.assertEqual(soundness_failures(case, mutated), [], (name, factory))


def _drop_upper_bound(lo, hi, k):
    """`_usable_depths` with the deep limit removed (the reach condition dropped)."""
    return np.linspace(lo, lo + 1.5 * (hi - lo) + 0.01, k)


class TestCodeMutants(unittest.TestCase):
    """Patching the real generator is caught by the same checks."""

    def _detect_per_factory(self, patch) -> List[str]:
        """Factories where NO corpus case detects the patched generator.

        Aggregated with ``any`` over the corpus, like the black-box matrix: a bug
        class is caught if some geometry exposes it. A deep limit that is merely
        conservative for a given hand (fingers far longer than the object, where
        going deeper is still a sound grasp) must not be reported as a defect.
        """
        hits: Dict[str, bool] = {}
        for case in corpus():
            with patch():
                mutated = case.run()
            hits[case.factory] = hits.get(case.factory, False) or detected(case, mutated)
        return sorted(f for f, hit in hits.items() if not hit)

    def test_ignoring_the_requested_clearance_is_detected(self):
        def patch():
            return mock.patch.object(ParallelJawGripper, "_resolve_clearance", lambda self, c, depth: 0.0)

        self.assertEqual(self._detect_per_factory(patch), [])

    def test_dropping_the_depth_upper_bound_is_detected(self):
        def patch():
            return mock.patch.object(GripperBase, "_usable_depths", staticmethod(_drop_upper_bound))

        self.assertEqual(self._detect_per_factory(patch), [])

    def test_removing_the_aperture_and_straddle_checks_is_detected(self):
        # Objects the gripper cannot straddle: unmutated, every factory returns [].
        # Without the check a factory either still returns [] (another guard, e.g. an
        # empty depth band, caught it) or emits templates the oracle must reject.
        escaped, unsound = [], 0
        for spec in CORPUS_GRIPPERS:
            for factory in sorted(FACTORIES):
                dims = oversize_dims(FACTORIES[factory].primitive, spec.max_aperture)
                case = GraspCase(spec, factory, dims, {"k": 3, "clearance": 0.004})
                self.assertEqual(case.run(), [], f"{factory} oversize case should be empty")
                with mock.patch.object(GripperBase, "_infeasibility_reason", lambda *a, **kw: None):
                    mutated = case.run()
                if not mutated:
                    continue
                unsound += 1
                if not soundness_failures(case, mutated):
                    escaped.append(f"{factory} ({spec.name})")
        self.assertEqual(escaped, [])
        self.assertGreater(unsound, 0, "no factory emitted templates without the aperture check")


class TestGripperMutants(unittest.TestCase):
    """A gripper whose dimensions do not match the certified hand is detected."""

    def test_longer_fingers_than_certified(self):
        misses = []
        for case in corpus():
            longer = ParallelJawGripper(
                finger_length=case.gripper.finger_length * 2.0, max_aperture=case.gripper.max_aperture
            )
            if not detected(case, case.run(gripper=longer)):
                misses.append(case.factory)
        self.assertEqual(misses, [])


if __name__ == "__main__":
    unittest.main()
