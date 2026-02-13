"""
qutrit_folded_logical_clifford.py

This module defines the logical Clifford operators for a single qutrit encoding
on a folded 5-qutrit grid. Each logical operator is provided as a
static method returning a cirq.Circuit. The grid is arranged as
follows:
"""

import cirq
from gates import (
    QuditCZ,
    QuditHadamard,
    QuditI,
    QuditS,
    QuditSwap,
    QuditX,
    QuditZ,
)


class QutritFoldedLogicalClifford:
    """
    Static factory for folded-layout logical Clifford operator circuits.

    This class provides one static constructor per primitive logical Clifford
    operation used elsewhere in the codebase to build full logical groups.

    Logical layout (fixed index mapping):
        q0 -> GridQid(0,0)
        q1 -> GridQid(0,2)
        q2 -> GridQid(1,1)
        q3 -> GridQid(2,0)
        q4 -> GridQid(2,2)

    Methods:
        logical_I(dimension=3): Logical identity operation on the folded block.
        logical_Hadamard(dimension=3): Logical Hadamard construction.
        logical_S(dimension=3): Logical phase-gate construction.
        logical_X(dimension=3): Logical Pauli-X construction.
        logical_Z(dimension=3): Logical Pauli-Z construction.
    """
    @staticmethod
    def logical_I(dimension: int = 3) -> cirq.Circuit:
        """
Returns the logical Identity operator on the folded 5â€“qutrit grid.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (returns the logical identity operator on the
                  folded 5â€“qutrit grid.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        # Logical identity is realized as per-site identity operations on the
        # full folded block.
        return cirq.Circuit(
            QuditI(dimension).on_each([
                cirq.GridQid(0, 0, dimension=dimension),
                cirq.GridQid(0, 2, dimension=dimension),
                cirq.GridQid(1, 1, dimension=dimension),
                cirq.GridQid(2, 0, dimension=dimension),
                cirq.GridQid(2, 2, dimension=dimension)
            ])
        )

    @staticmethod
    def logical_Hadamard(dimension: int = 3) -> cirq.Circuit:
        """
Returns the logical Hadamard operator on the folded 5â€“qutrit grid.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (returns the logical hadamard operator on the
                  folded 5â€“qutrit grid.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        # This sequence implements the folded-code logical Hadamard using local
        # basis transforms plus one data-route swap.
        return cirq.Circuit(
            QuditHadamard(dimension).on(
                cirq.GridQid(0, 0, dimension=dimension)),
            (QuditHadamard(dimension)**(-1)
             ).on(cirq.GridQid(0, 2, dimension=dimension)),
            QuditHadamard(dimension).on(
                cirq.GridQid(1, 1, dimension=dimension)),
            QuditHadamard(dimension).on(
                cirq.GridQid(2, 0, dimension=dimension)),
            QuditHadamard(dimension).on(
                cirq.GridQid(2, 2, dimension=dimension)),
            QuditSwap(dimension).on(
                cirq.GridQid(1, 1, dimension=dimension),
                cirq.GridQid(2, 0, dimension=dimension)
            )
        )

    @staticmethod
    def logical_S(dimension: int = 3) -> cirq.Circuit:
        """
Returns the logical S operator on the folded 5â€“qutrit grid.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (returns the logical s operator on the folded
                  5â€“qutrit grid.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        # Logical phase construction combines local S/S^-1 with one entangling
        # correction on the data-support pair.
        return cirq.Circuit(
            QuditS(dimension).on(cirq.GridQid(0, 0, dimension=dimension)),
            (QuditS(dimension) ** (-1)).on(
                cirq.GridQid(0, 2, dimension=dimension)
            ),
            (QuditCZ(dimension)**-1).on(
                cirq.GridQid(1, 1, dimension=dimension),
                cirq.GridQid(2, 0, dimension=dimension)
            ),
            QuditS(dimension).on(cirq.GridQid(2, 2, dimension=dimension))
        )

    @staticmethod
    def logical_X(dimension: int = 3) -> cirq.Circuit:
        """
Returns the logical X operator on the folded 5â€“qutrit grid.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (returns the logical x operator on the folded
                  5â€“qutrit grid.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        # Logical X support is restricted to the encoded-data subset {q0, q3}.
        return cirq.Circuit(
            QuditX(dimension).on_each([
                cirq.GridQid(0, 0, dimension=dimension),
                cirq.GridQid(2, 0, dimension=dimension)
            ])
        )

    @staticmethod
    def logical_Z(dimension: int = 3) -> cirq.Circuit:
        """
Returns the logical Z operator on the folded 5â€“qutrit grid.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (returns the logical z operator on the folded
                  5â€“qutrit grid.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        # Logical Z support is restricted to the encoded-data subset {q0, q2}.
        return cirq.Circuit(
            QuditZ(dimension).on_each([
                cirq.GridQid(0, 0, dimension=dimension),
                cirq.GridQid(1, 1, dimension=dimension)
            ])
        )
