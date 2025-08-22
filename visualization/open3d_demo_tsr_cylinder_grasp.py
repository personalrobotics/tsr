"""Demo: Visualize a TSR sample (as a sphere) for a cylinder grasp using Open3D."""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path

# Ensure src is on sys.path for tsr imports
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsr.core.tsr_template import TSRTemplate
from tsr.schema import EntityClass, TaskCategory

# Cylinder parameters
cyl_height = 0.2
cyl_radius = 0.04
cyl_color = [0.5, 0.5, 0.5]  # RGB [0,1]

# Create Open3D geometries
cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=cyl_height)
cylinder.paint_uniform_color(cyl_color)
# Open3D cylinder is Z-aligned by default, so no rotation needed

# Define a TSRTemplate for a side grasp on a cylinder
cylinder_grasp_template = TSRTemplate(
    T_ref_tsr=np.eye(4),
    Tw_e=np.array([
        [0, 0, 1, -(cyl_radius + 0.05)],  # Approach from -z, offset by radius+5cm
        [1, 0, 0, 0],
        [0, 1, 0, 0],
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
    reference_entity=EntityClass.BOX,
    task_category=TaskCategory.GRASP,
    variant="side",
    name="Cylinder Side Grasp",
    description="Grasp a cylinder from the side with 5cm approach distance"
)

# Cylinder pose (at origin, upright)
cylinder_pose = np.eye(4)

# Instantiate the TSR at the cylinder pose
tsr = cylinder_grasp_template.instantiate(cylinder_pose)

# Sample a pose from the TSR
T_sample = tsr.sample()

# Create a sphere at the TSR sample pose
sphere_radius = 0.025
sphere_color = [1.0, 0.0, 0.0]
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
sphere.paint_uniform_color(sphere_color)

# Set sphere pose
sphere_T = T_sample.copy()
sphere_center = sphere.get_center()
sphere.translate(sphere_T[:3, 3] - sphere_center)

# Visualize using Open3D's web-based visualizer (works on Wayland)
try:
    o3d.visualization.draw([cylinder, sphere], show_ui=True, title="Open3D TSR Cylinder Grasp Demo")
except AttributeError:
    print("[WARN] Open3D 'draw' function not found. Please upgrade Open3D to >=0.17. Falling back to draw_geometries (may not work on Wayland)...")
    o3d.visualization.draw_geometries([cylinder, sphere], window_name="Open3D TSR Cylinder Grasp Demo")
