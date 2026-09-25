#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The PyVista backend's deprecation contract (issue #139).

Viser is the recommended viewer; PyVista keeps working throughout 2.x and is removed
in 3.0. The promises that are easy to break silently are pinned here: the warning
fires where a user can act on it (construction), not where a library can trigger it on
their behalf (import), and the `viz` extra keeps meaning PyVista for all of 2.x —
repointing it in a minor release would silently change what an install provides.
"""

import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
pyvista = __import__("importlib").util.find_spec("pyvista")


@unittest.skipIf(pyvista is None, "optional extra 'viz' is not installed")
class TestDeprecationWarning(unittest.TestCase):
    def test_constructing_the_visualizer_warns(self):
        from tsr.viz import TSRVisualizer

        with self.assertWarns(DeprecationWarning) as caught:
            TSRVisualizer()
        message = str(caught.warning)
        # An actionable warning names the replacement, the removal version and the guide.
        self.assertIn("3.0", message)
        self.assertIn("sstsr[viser]", message)
        self.assertIn("tsr.viser", message)
        self.assertIn("MIGRATION-VISER.md", message)

    def test_importing_the_module_does_not_warn(self):
        # A library importing tsr.viz must not warn on its users' behalf; only the user
        # constructing a visualizer can act on it.
        code = "import warnings; warnings.simplefilter('error'); import tsr.viz; print('ok')"
        out = subprocess.check_output([sys.executable, "-c", code], text=True, stderr=subprocess.STDOUT)
        self.assertIn("ok", out)

    def test_the_backend_still_works(self):
        # Deprecated is not broken: 2.x users keep a working renderer.
        from tsr.viz import TSRVisualizer, cylinder_renderer, parallel_jaw_renderer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            viz = TSRVisualizer()
        self.assertTrue(callable(cylinder_renderer(radius=0.03, height=0.12)))
        self.assertTrue(callable(parallel_jaw_renderer(finger_length=0.08, half_aperture=0.03)))
        self.assertTrue(hasattr(viz, "render") and hasattr(viz, "render_multi"))


class TestExtrasAreUnchanged(unittest.TestCase):
    """`viz` keeps meaning PyVista for all of 2.x; `viser` is separate and optional."""

    def setUp(self):
        self.extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]

    def test_viz_extra_still_installs_pyvista(self):
        viz = " ".join(self.extras["viz"])
        self.assertIn("pyvista", viz)
        self.assertNotIn("viser", viz)  # repointing viz -> Viser is a 3.0 change, not 2.x

    def test_viser_extra_is_separate(self):
        self.assertIn("viser", " ".join(self.extras["viser"]))

    def test_neither_backend_is_a_core_dependency(self):
        core = " ".join(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
        self.assertNotIn("viser", core)
        self.assertNotIn("pyvista", core)


class TestMigrationGuide(unittest.TestCase):
    """The guide must state a disposition for every public PyVista workflow."""

    def setUp(self):
        self.guide = (ROOT / "docs" / "MIGRATION-VISER.md").read_text()

    def test_every_public_renderer_is_addressed(self):
        import tsr.viz as viz

        public = [n for n in dir(viz) if not n.startswith("_") and (n.endswith("_renderer") or n == "TSRVisualizer")]
        missing = [n for n in public if n not in self.guide]
        self.assertEqual(missing, [], "migration guide does not mention these public names")

    def test_headless_png_disposition_is_explicit(self):
        # #139 requires the guide to say what happens to browser-free rendering.
        self.assertIn("Headless PNG", self.guide)
        self.assertIn("No replacement", self.guide)

    def test_removal_version_is_stated(self):
        self.assertIn("sstsr 3.0", self.guide)


if __name__ == "__main__":
    unittest.main()
