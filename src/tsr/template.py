from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .tsr import TSR


@dataclass(frozen=True)
class TSRTemplate:
    """Neutral TSR template with semantic context (pure geometry, scene-agnostic).

    A TSRTemplate defines a TSR in a reference-relative coordinate frame,
    allowing it to be instantiated at any reference pose in the world.
    This makes templates reusable across different scenes and object poses.

    Required fields: T_ref_tsr, Tw_e, Bw, name, description, task, subject, reference
    Optional fields: variant, preshape, stability_margin, any user-defined metadata

    Gripper frame convention (for grasp templates):
        z = approach direction (toward object surface)
        y = finger opening direction
        x = palm normal (right-hand rule: x = y × z)

    Attributes:
        T_ref_tsr: 4×4 transform from REFERENCE frame to TSR frame.
        Tw_e: 4×4 transform from TSR frame to SUBJECT frame at Bw = 0 (canonical).
        Bw: (6,2) bounds in TSR frame over [x,y,z,roll,pitch,yaw].
        task: The task being performed (e.g., "grasp", "place", "pour").
        subject: The entity whose pose is constrained (e.g., "gripper").
        reference: The entity relative to which TSR is defined (e.g., "mug").
        name: Human-readable name for the template.
        description: Detailed description of the template.
        variant: Optional variant identifier (e.g., "side", "top").
        preshape: Optional gripper configuration as DOF values.
        stability_margin: For placement templates, the stability margin in radians
            (arctan(d_min / h_com)). None for grasp templates or analytic primitives.
    """

    T_ref_tsr: np.ndarray
    Tw_e: np.ndarray
    Bw: np.ndarray
    task: str
    subject: str
    reference: str
    name: str = ""
    description: str = ""
    variant: str = ""
    preshape: Optional[np.ndarray] = None
    stability_margin: Optional[float] = None

    def __repr__(self) -> str:
        parts = [f"task={self.task!r}", f"subject={self.subject!r}"]
        if self.variant:
            parts.append(f"variant={self.variant!r}")
        if self.stability_margin is not None:
            parts.append(f"margin={np.degrees(self.stability_margin):.1f}°")
        return f"TSRTemplate({', '.join(parts)})"

    def instantiate(self, T_ref_world: np.ndarray) -> TSR:
        """Bind this template to a concrete reference pose in world.

        Args:
            T_ref_world: 4×4 pose of the reference entity in world frame.

        Returns:
            TSR whose T0_w = T_ref_world @ T_ref_tsr, Tw_e = Tw_e, Bw = Bw.
        """
        T0_w = T_ref_world @ self.T_ref_tsr
        return TSR(T0_w=T0_w, Tw_e=self.Tw_e, Bw=self.Bw)

    def sample(self, T_ref_world: np.ndarray) -> np.ndarray:
        """Bind to a reference pose and sample one end-effector pose.

        Shorthand for ``self.instantiate(T_ref_world).sample()``.

        Args:
            T_ref_world: 4×4 pose of the reference entity in world frame.

        Returns:
            4×4 sampled end-effector pose in world frame.
        """
        return self.instantiate(T_ref_world).sample()

    def to_dict(self):
        """Convert this TSRTemplate to a python dict for serialization."""
        result = {
            'name': self.name,
            'description': self.description,
            'task': self.task,
            'subject': self.subject,
            'reference': self.reference,
            'T_ref_tsr': self.T_ref_tsr.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist(),
        }
        if self.variant:
            result['variant'] = self.variant
        if self.preshape is not None:
            result['preshape'] = self.preshape.tolist()
        if self.stability_margin is not None:
            result['stability_margin'] = float(self.stability_margin)
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
            task=x['task'],
            subject=x['subject'],
            reference=x['reference'],
            variant=x.get('variant', ''),
            T_ref_tsr=np.array(x['T_ref_tsr']),
            Tw_e=np.array(x['Tw_e']),
            Bw=np.array(x['Bw']),
            preshape=preshape,
            stability_margin=x.get('stability_margin', None),
        )

    def to_json(self):
        """Convert this TSRTemplate to a JSON string."""
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """Construct a TSRTemplate from a JSON string."""
        import json
        x_dict = json.loads(x, *args, **kw_args)
        return TSRTemplate.from_dict(x_dict)

    def to_yaml(self):
        """Convert this TSRTemplate to a YAML string."""
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x):
        """Construct a TSRTemplate from a YAML string."""
        import yaml
        x_dict = yaml.safe_load(x)
        return TSRTemplate.from_dict(x_dict)
