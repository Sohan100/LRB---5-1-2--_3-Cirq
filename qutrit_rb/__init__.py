"""Core package for qutrit randomized benchmarking simulations."""

import sys

from . import gates as _gates
from . import logical_rb_seq as _logical_rb_seq
from . import noise as _noise
from . import pauli as _pauli
from . import qutrit_clifford as _qutrit_clifford
from . import qutrit_folded_logical_clifford as _qutrit_folded_logical_clifford
from . import qutrit_folded_logical_plus_state as _qutrit_folded_logical_plus_state
from . import qutrit_logical_clifford as _qutrit_logical_clifford
from . import qutrit_logical_pauli as _qutrit_logical_pauli
from . import qutrit_logical_rb_logical_noise_sim as _qutrit_logical_rb_logical_noise_sim
from . import qutrit_logical_rb_terminal_check_sim as _qutrit_logical_rb_terminal_check_sim
from . import qutrit_physical_rb_sim as _qutrit_physical_rb_sim
from . import qutrit_rb_plotting as _qutrit_rb_plotting
from . import rb_checkpoint as _rb_checkpoint
from . import rb_seq as _rb_seq
from .qutrit_logical_rb_logical_noise_sim import (
    LogicalRBSim,
    LogicalRbLogicalNoiseSim,
)
from .qutrit_logical_rb_terminal_check_sim import LogicalRbTerminalCheckSim
from .qutrit_physical_rb_sim import PhysicalRBSim, PhysicalRbSim
from .qutrit_rb_plotting import LogicalRbPlotter


_LEGACY_MODULE_ALIASES = {
    "gates": _gates,
    "logical_rb_seq": _logical_rb_seq,
    "noise": _noise,
    "pauli": _pauli,
    "qutrit_clifford": _qutrit_clifford,
    "qutrit_folded_logical_clifford": _qutrit_folded_logical_clifford,
    "qutrit_folded_logical_plus_state": _qutrit_folded_logical_plus_state,
    "qutrit_logical_clifford": _qutrit_logical_clifford,
    "qutrit_logical_pauli": _qutrit_logical_pauli,
    "qutrit_logical_rb_logical_noise_sim": (
        _qutrit_logical_rb_logical_noise_sim
    ),
    "qutrit_logical_rb_terminal_check_sim": (
        _qutrit_logical_rb_terminal_check_sim
    ),
    "qutrit_physical_rb_sim": _qutrit_physical_rb_sim,
    "qutrit_rb_plotting": _qutrit_rb_plotting,
    "rb_checkpoint": _rb_checkpoint,
    "rb_seq": _rb_seq,
}

for _legacy_name, _module in _LEGACY_MODULE_ALIASES.items():
    sys.modules.setdefault(_legacy_name, _module)

__all__ = [
    "LogicalRBSim",
    "LogicalRbLogicalNoiseSim",
    "LogicalRbTerminalCheckSim",
    "LogicalRbPlotter",
    "PhysicalRBSim",
    "PhysicalRbSim",
]
