# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for HandRegistry."""

import unittest

from tsr.hands import HandRegistry, ParallelJawGripper, default_registry
from tsr.template import TSRTemplate


class TestHandRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = HandRegistry()

    def test_register_and_get(self):
        def fn(gripper, **kw):
            return gripper.grasp_cylinder(**kw)

        self.registry.register("test_hand", "cylinder", "grasp")(fn)
        self.assertIs(self.registry.get("test_hand", "cylinder", "grasp"), fn)

    def test_register_as_decorator(self):
        @self.registry.register("deco_hand", "sphere", "grasp")
        def _gen(gripper, **kw):
            return []

        self.assertIs(self.registry.get("deco_hand", "sphere", "grasp"), _gen)

    def test_get_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.get("no_hand", "cylinder", "grasp")

    def test_contains(self):
        self.registry.register("h", "r", "t")(lambda g, **kw: [])
        self.assertIn(("h", "r", "t"), self.registry)
        self.assertNotIn(("h", "r", "x"), self.registry)

    def test_list_tasks_sorted(self):
        self.registry.register("b", "cylinder", "grasp")(lambda g, **kw: [])
        self.registry.register("a", "cylinder", "grasp")(lambda g, **kw: [])
        tasks = self.registry.list_tasks()
        self.assertEqual(tasks, sorted(tasks))

    def test_default_registry_has_parallel_jaw(self):
        self.assertIn(("parallel_jaw", "cylinder", "grasp"), default_registry)

    def test_default_registry_has_robotiq(self):
        self.assertIn(("robotiq_2f140", "cylinder", "grasp"), default_registry)

    def test_generator_produces_templates(self):
        gen = default_registry.get("parallel_jaw", "cylinder", "grasp")
        gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        templates = gen(gripper, cylinder_radius=0.040, cylinder_height=0.10)
        self.assertEqual(len(templates), 12)  # 4*k with k=3
        for t in templates:
            self.assertIsInstance(t, TSRTemplate)
