#!/usr/bin/env python
"""One-time migration script: convert primitives DSL templates to serialized format.

Usage:
    uv run scripts/migrate_templates.py

This script is intentionally NOT committed to the repo. It's a one-time tool
to convert templates/*.yaml from the human-DSL format to the serialized matrix
format used by TSRTemplate.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import yaml
import numpy as np
from tsr.primitives import load_template_file, ParsedTemplate
from tsr.template import TSRTemplate
from tsr.io import save_template


def parsed_to_tsr_template(parsed: ParsedTemplate, spec: dict) -> TSRTemplate:
    """Convert a ParsedTemplate to TSRTemplate, preserving metadata."""
    preshape = None
    if parsed.gripper and 'aperture' in parsed.gripper:
        preshape = np.array([parsed.gripper['aperture']])

    return TSRTemplate(
        name=parsed.name,
        description=parsed.description,
        task=parsed.task,
        subject=parsed.subject,
        reference=parsed.reference,
        T_ref_tsr=parsed.T_ref_tsr,
        Tw_e=parsed.Tw_e,
        Bw=parsed.Bw,
        preshape=preshape,
    )


def migrate_directory(src_dir: Path, dst_dir: Path) -> None:
    """Migrate all .yaml files in src_dir to serialized format in dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    for yaml_path in sorted(src_dir.glob("*.yaml")):
        print(f"  Migrating {yaml_path.name}...")

        with open(yaml_path) as f:
            spec = yaml.safe_load(f)

        parsed = load_template_file(str(yaml_path))
        template = parsed_to_tsr_template(parsed, spec)

        out_path = dst_dir / yaml_path.name
        save_template(template, out_path)
        print(f"    -> {out_path}")

        # Print matrices for verification
        print(f"    T_ref_tsr diag: {np.diag(template.T_ref_tsr)}")
        print(f"    Tw_e:\n{np.array2string(template.Tw_e, precision=4, suppress_small=True)}")
        print(f"    Bw:\n{np.array2string(template.Bw, precision=4, suppress_small=True)}")
        if template.preshape is not None:
            print(f"    preshape: {template.preshape}")
        print()


def main():
    templates_dir = repo_root / "templates"
    subdirs = ["grasps", "places", "tasks"]

    print(f"Migrating templates from {templates_dir}")
    print("=" * 60)

    for subdir in subdirs:
        src = templates_dir / subdir
        if not src.exists():
            continue
        print(f"\n[{subdir}]")
        migrate_directory(src, src)  # overwrite in-place

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
