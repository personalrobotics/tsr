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

# Rotate cylinder by +90 deg about X to stand up along Z (like Meshcat demo)
cyl_rot = R.from_euler('x', 90, degrees=True).as_matrix()
cyl_transform = np.eye(4)
# cyl_transform[:3, :3] = cyl_rot
cylinder.transform(cyl_transform)

roll90 = R.from_euler('x', 90, degrees=True).as_matrix()
# Tw_e = np.array([
#     [0, 0, 1, 0],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ])
Tw_e = np.eye(4)
# Tw_e[:3, :3] = Tw_e[:3, :3] @ roll90

cylinder_grasp_template = TSRTemplate(
    T_ref_tsr=np.eye(4),
    Tw_e=Tw_e,
    Bw=np.array([
        [0, 0],
        [0, 0],
        [0, 0],
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


# Cylinder pose (at origin, upright, with 90 deg X rotation)
cylinder_pose = np.eye(4)
cylinder_pose[:3, :3] = cyl_rot

# Instantiate the TSR at the cylinder pose
tsr = cylinder_grasp_template.instantiate(cylinder_pose)


# Sample a pose from the TSR
T_sample = tsr.sample()



# Load 2f140.stl and place at TSR sample pose, scaling to meters
stl_path = str(REPO_ROOT / "visualization" / "2f140.stl")
gripper_mesh = o3d.io.read_triangle_mesh(stl_path)
gripper_mesh.compute_vertex_normals()
gripper_mesh.paint_uniform_color([0.2, 0.2, 0.8])
# Scale gripper mesh to meters (0.001)
gripper_mesh.scale(0.001, center=gripper_mesh.get_center())
# Move gripper mesh to TSR sample pose
gripper_center = gripper_mesh.get_center()
gripper_mesh.translate(T_sample[:3, 3] - gripper_center)

# Debug: print positions
print("Cylinder center:", np.array([0, 0, 0]))
print("Gripper pose:", T_sample[:3, 3])

# Visualize only the cylinder and gripper mesh
try:
    o3d.visualization.draw([cylinder, gripper_mesh], show_ui=True, title="Open3D TSR Cylinder Grasp Demo")
except AttributeError:
    print("[WARN] Open3D 'draw' function not found. Please upgrade Open3D to >=0.17. Falling back to draw_geometries (may not work on Wayland)...")
    o3d.visualization.draw_geometries([cylinder, gripper_mesh], window_name="Open3D TSR Cylinder Grasp Demo")
