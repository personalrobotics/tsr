from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Use existing core TSR implementation without changes.
from .tsr import TSR as CoreTSR  # type: ignore[attr-defined]
from ..schema import EntityClass, TaskCategory


@dataclass(frozen=True)
class TSRTemplate:
    """Neutral TSR template with semantic context (pure geometry, scene-agnostic).

    A TSRTemplate defines a TSR in a reference-relative coordinate frame,
    allowing it to be instantiated at any reference pose in the world.
    This makes templates reusable across different scenes and object poses.

    Attributes:
        T_ref_tsr: 4×4 transform from REFERENCE frame to TSR frame.
                   This defines how the TSR frame is oriented relative to
                   the reference entity (e.g., object).
        Tw_e: 4×4 transform from TSR frame to SUBJECT frame at Bw = 0 (canonical).
              This defines the desired pose of the subject (e.g., end-effector)
              relative to the TSR frame when all bounds are at their nominal values.
        Bw: (6,2) bounds in TSR frame over [x,y,z,roll,pitch,yaw].
            Each row [i,:] defines the min/max bounds for dimension i.
            Translation bounds (rows 0-2) are in meters.
            Rotation bounds (rows 3-5) are in radians using RPY convention.
        subject_entity: The entity whose pose is constrained (e.g., gripper).
        reference_entity: The entity relative to which TSR is defined (e.g., object).
        task_category: The category of task being performed (e.g., GRASP, PLACE).
        variant: The specific variant of the task (e.g., "side", "top").
        name: Optional human-readable name for the template.
        description: Optional detailed description of the template.
        preshape: Optional gripper configuration as DOF values.
                 This specifies the desired gripper joint angles or configuration
                 that should be achieved before or during the TSR execution.
                 For parallel jaw grippers, this might be a single value (aperture).
                 For multi-finger hands, this would be a list of joint angles.
                 None if no specific gripper configuration is required.

    Examples:
        >>> # Create a template for grasping a cylinder from the side
        >>> template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),  # TSR frame aligned with cylinder frame
        ...     Tw_e=np.array([
        ...         [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
        ...         [1, 0, 0, 0],      # x-axis perpendicular to cylinder
        ...         [0, 1, 0, 0.05],   # y-axis along cylinder axis
        ...         [0, 0, 0, 1]
        ...     ]),
        ...     Bw=np.array([
        ...         [0, 0],           # x: fixed position
        ...         [0, 0],           # y: fixed position
        ...         [-0.01, 0.01],    # z: small tolerance
        ...         [0, 0],           # roll: fixed
        ...         [0, 0],           # pitch: fixed
        ...         [-np.pi, np.pi]   # yaw: full rotation
        ...     ]),
        ...     subject_entity=EntityClass.GENERIC_GRIPPER,
        ...     reference_entity=EntityClass.MUG,
        ...     task_category=TaskCategory.GRASP,
        ...     variant="side",
        ...     name="Cylinder Side Grasp",
        ...     description="Grasp a cylindrical object from the side with 5cm approach distance",
        ...     preshape=np.array([0.08])  # 8cm aperture for parallel jaw gripper
        ... )
        >>> 
        >>> # Instantiate at a specific cylinder pose
        >>> cylinder_pose = np.array([
        ...     [1, 0, 0, 0.5],  # Cylinder at x=0.5
        ...     [0, 1, 0, 0.0],
        ...     [0, 0, 1, 0.3],
        ...     [0, 0, 0, 1]
        ... ])
        >>> tsr = template.instantiate(cylinder_pose)
        >>> pose = tsr.sample()  # Sample a grasp pose
    """

    T_ref_tsr: np.ndarray
    Tw_e: np.ndarray
    Bw: np.ndarray
    subject_entity: EntityClass
    reference_entity: EntityClass
    task_category: TaskCategory
    variant: str
    name: str = ""
    description: str = ""
    preshape: Optional[np.ndarray] = None

    def instantiate(self, T_ref_world: np.ndarray) -> CoreTSR:
        """Bind this template to a concrete reference pose in world.

        This method creates a concrete TSR by combining the template's
        reference-relative definition with a specific reference pose in
        the world coordinate frame.

        Args:
            T_ref_world: 4×4 pose of the reference entity in world frame.
                        This is typically the pose of the object being
                        manipulated (e.g., mug, table, valve).

        Returns:
            CoreTSR whose T0_w = T_ref_world @ T_ref_tsr, Tw_e = Tw_e, Bw = Bw.
            The resulting TSR can be used for sampling, distance calculations,
            and other TSR operations.

        Examples:
                    >>> # Create a template for placing objects on a table
        >>> place_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([
        ...         [1, 0, 0, 0],      # Object x-axis aligned with table
        ...         [0, 1, 0, 0],      # Object y-axis aligned with table
        ...         [0, 0, 1, 0.02],   # Object 2cm above table surface
        ...         [0, 0, 0, 1]
        ...     ]),
        ...     Bw=np.array([
        ...         [-0.1, 0.1],       # x: allow sliding on table
        ...         [-0.1, 0.1],       # y: allow sliding on table
        ...         [0, 0],            # z: fixed height
        ...         [0, 0],            # roll: keep level
        ...         [0, 0],            # pitch: keep level
        ...         [-np.pi/4, np.pi/4]  # yaw: allow some rotation
        ...     ]),
        ...     subject_entity=EntityClass.MUG,
        ...     reference_entity=EntityClass.TABLE,
        ...     task_category=TaskCategory.PLACE,
        ...     variant="on",
        ...     name="Table Placement",
        ...     description="Place object on table surface with 2cm clearance"
        ... )
        >>> 
        >>> # Example with multi-finger hand preshape
        >>> multi_finger_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([
        ...         [0, 0, 1, -0.03],  # Approach from -z, 3cm offset
        ...         [1, 0, 0, 0],      # x-axis perpendicular to object
        ...         [0, 1, 0, 0],      # y-axis along object
        ...         [0, 0, 0, 1]
        ...     ]),
        ...     Bw=np.array([
        ...         [0, 0],           # x: fixed position
        ...         [0, 0],           # y: fixed position
        ...         [-0.005, 0.005],  # z: small tolerance
        ...         [0, 0],           # roll: fixed
        ...         [0, 0],           # pitch: fixed
        ...         [-np.pi/6, np.pi/6]  # yaw: limited rotation
        ...     ]),
        ...     subject_entity=EntityClass.GENERIC_GRIPPER,
        ...     reference_entity=EntityClass.BOX,
        ...     task_category=TaskCategory.GRASP,
        ...     variant="precision",
        ...     name="Precision Grasp",
        ...     description="Precision grasp with multi-finger hand",
        ...     preshape=np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])  # 6-DOF hand configuration
        ... )
            >>> 
            >>> # Instantiate at table pose
            >>> table_pose = np.eye(4)  # Table at world origin
            >>> place_tsr = place_template.instantiate(table_pose)
            >>> placement_pose = place_tsr.sample()
        """
        T0_w = T_ref_world @ self.T_ref_tsr
        return CoreTSR(T0_w=T0_w, Tw_e=self.Tw_e, Bw=self.Bw)

    def to_dict(self):
        """Convert this TSRTemplate to a python dict for serialization."""
        result = {
            'name': self.name,
            'description': self.description,
            'subject_entity': self.subject_entity.value,
            'reference_entity': self.reference_entity.value,
            'task_category': self.task_category.value,
            'variant': self.variant,
            'T_ref_tsr': self.T_ref_tsr.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist(),
        }
        if self.preshape is not None:
            result['preshape'] = self.preshape.tolist()
        return result

    @staticmethod
    def from_dict(x):
        """Construct a TSRTemplate from a python dict."""
        preshape = None
        if 'preshape' in x and x['preshape'] is not None:
            preshape = np.array(x['preshape'])
        
        return TSRTemplate(
            name=x.get('name', ''),
            description=x.get('description', ''),
            subject_entity=EntityClass(x['subject_entity']),
            reference_entity=EntityClass(x['reference_entity']),
            task_category=TaskCategory(x['task_category']),
            variant=x['variant'],
            T_ref_tsr=np.array(x['T_ref_tsr']),
            Tw_e=np.array(x['Tw_e']),
            Bw=np.array(x['Bw']),
            preshape=preshape,
        )

    def to_json(self):
        """Convert this TSRTemplate to a JSON string."""
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSRTemplate from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json
        x_dict = json.loads(x, *args, **kw_args)
        return TSRTemplate.from_dict(x_dict)

    def to_yaml(self):
        """Convert this TSRTemplate to a YAML string."""
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x, *args, **kw_args):
        """
        Construct a TSRTemplate from a YAML string.

        This method internally forwards all arguments to `yaml.safe_load`.
        """
        import yaml
        x_dict = yaml.safe_load(x, *args, **kw_args)
        return TSRTemplate.from_dict(x_dict)
