"""Demo: Visualize a TSR sample (as a sphere) for a cylinder grasp using generate_cylinder_grasp_template."""

import numpy as np
from meshcat import Visualizer
import meshcat.geometry as g
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
cyl_color = 0x888888

# Create Meshcat visualizer
vis = Visualizer().open()

# Cylinder pose: rotate +90 deg about X to stand up along Z
cyl_pose = np.eye(4)
cyl_pose[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
vis["object"].set_object(g.Cylinder(cyl_height, cyl_radius), g.MeshPhongMaterial(color=cyl_color))
vis["object"].set_transform(cyl_pose)


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

# Place the sphere at the TSR sample pose
sphere_radius = 0.025
sphere_color = 0xff0000
vis["tsr_sample"].set_object(g.Sphere(sphere_radius), g.MeshPhongMaterial(color=sphere_color, opacity=0.8))
vis["tsr_sample"].set_transform(T_sample)

print("Meshcat visualizer running. Open http://localhost:7000/static/ if not already open.")
print("Press Ctrl-C to exit.")

try:
    import time
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("Shutting down visualizer.")
