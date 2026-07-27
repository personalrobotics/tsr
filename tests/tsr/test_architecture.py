# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Architecture-enforcement tests (see docs/ARCHITECTURE.md).

These freeze the layering so a careless future import can't rot it:

* Layer 0 (``tsr.core``) is the pure-math heart and must not import from any
  upper layer (template, hands, placement, io, sampling, viz).
* ``import tsr`` must stay lightweight — it must not transitively pull the
  heavy optional viz dependencies (pyvista / vtk / matplotlib).
"""

import ast
import subprocess
import sys
from pathlib import Path

import tsr.core

_UPPER_LAYERS = {"template", "hands", "placement", "io", "sampling", "viz"}


def _imported_names(py_file: Path):
    """Yield top-level module tokens imported by a source file."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # 'from tsr.template import X' -> module 'tsr.template'
            # 'from ..hands import Y'      -> level>0, module 'hands'
            yield ("." * node.level) + (node.module or "")


def test_core_has_no_upward_imports():
    core_dir = Path(tsr.core.__file__).parent
    offenders = []
    for py_file in core_dir.rglob("*.py"):
        for name in _imported_names(py_file):
            tail = name.lstrip(".")
            head = tail.split(".")[-1] if name.startswith(".") else tail
            # Catch both absolute 'tsr.hands...' and relative '..hands' forms.
            if any(part in _UPPER_LAYERS for part in (head, tail.split(".")[0])):
                if tail.startswith("tsr.") or name.startswith("."):
                    offenders.append(f"{py_file.name}: imports {name}")
    assert not offenders, "tsr.core must not import upper layers:\n" + "\n".join(offenders)


def test_import_tsr_does_not_pull_heavy_viz_deps():
    code = "import tsr, sys; print([m for m in ('pyvista', 'vtk', 'matplotlib') if m in sys.modules])"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "[]", f"`import tsr` unexpectedly imported heavy deps: {out}"
