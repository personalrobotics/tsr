# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Back-compat shim: ``TSRChain`` now lives in :mod:`tsr.core.tsr_chain`.

Importing ``from tsr.tsr_chain import TSRChain`` keeps working. New code should
prefer ``from tsr import TSRChain`` or ``from tsr.core import TSRChain``.
"""

from .core.tsr_chain import TSRChain

__all__ = ["TSRChain"]
