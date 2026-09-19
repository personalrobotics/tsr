# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Back-compat shim: the ``TSR`` class now lives in :mod:`tsr.core.tsr`.

Importing ``from tsr.tsr import TSR`` keeps working. New code should prefer
``from tsr import TSR`` or ``from tsr.core import TSR``.
"""

from .core.tsr import NANBW, TSR

__all__ = ["TSR", "NANBW"]
