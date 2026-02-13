"""
qutrit_folded_logical_plus_state.py

This module defines the QutritFoldedLogicalPlusState class, which encapsulates
the setup
of a logical plus state on a folded 5-qutrit grid arranged as follows:

             q0   q1
               q2
             q3   q4

The class provides methods to generate the state preparation circuit and to
verify X-stabilizers and Z-stabilizers as well as a logical X
operator.
"""

import cirq
import numpy as np
from gates import QuditHadamard, QuditCNOT, QuditX, QuditZ


class QutritFoldedLogicalPlusState:
    """
    Represents a folded 5-qutrit grid for preparing a logical plus
    state (|+>).
    The grid is defined from a 3x3 array by excluding positions:
      (0,1), (1,0), (1,2), (2,1)

    The remaining positions (in order) are:
      q0: (0,0), q1: (0,2), q2: (1,1), q3: (2,0), q4: (2,2)
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize folded-layout geometry and canonical qutrit ordering.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.dimension = dimension
        self.excluded_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
        self.qutrits = sorted(
            [cirq.GridQid(x, y, dimension=self.dimension)
             for x in range(3) for y in range(3)
             if (x, y) not in self.excluded_positions],
            key=lambda q: (q.row, q.col)
        )
        # Expected order: [q0, q1, q2, q3, q4]. All stabilizer and logical
        # operator checks in this file assume this exact index mapping.

    def generate_state_circuit(self):
        """
        Generate the logical plus state circuit and compute its final state
        vector. The circuit performs: 1. Applies the
        conjugate-transpose of the Hadamard gate (denoted **'ct')
        on all qutrits with row > 0. 2. Applies an inverse CNOT
        from q2 to q1. 3. Applies a CNOT from q1 to q0. 4. Applies
        two CNOTs: one from q3 to q0 and one from q4 to q2.
        Args:
        None: `generate_state_circuit` relies on object state and accepts no
        additional inputs.
        
        Returns:
            object: Constructed object generated from the provided inputs.
        
        Raises:
        ValueError: If `generate_state_circuit` receives inputs that are
        incompatible with its expected configuration.
        """
        circuit = cirq.Circuit()
        # Apply Hadamard-dagger on qutrits with row > 0.
        plus_targets = [q for q in self.qutrits if q.row > 0]
        plus_moment = cirq.Moment(
            (QuditHadamard(self.dimension)**'ct').on_each(plus_targets)
        )
        circuit.append(plus_moment)
        # Inverse CNOT: q2 -> q1.
        circuit.append([
            (QuditCNOT(self.dimension)**-1).on(self.qutrits[2],
                                               self.qutrits[1])
        ])
        # CNOT: q1 -> q0.
        circuit.append([
            QuditCNOT(self.dimension).on(self.qutrits[1],
                                         self.qutrits[0])
        ])
        # CNOTs: q3 -> q0 and q4 -> q2.
        circuit.append([
            QuditCNOT(self.dimension).on(self.qutrits[3],
                                         self.qutrits[0]),
            QuditCNOT(self.dimension).on(self.qutrits[4],
                                         self.qutrits[2])
        ])
        # Return both the construction circuit and the resulting state vector
        # so callers can run algebraic verification directly.
        simulator = cirq.Simulator()
        final_state = simulator.simulate(circuit, qubit_order=self.qutrits)
        return circuit, final_state.final_state_vector

    @staticmethod
    def kron_list(operators):
        """
Compute the Kronecker product of a list of operators.

Args:
    operators (Any): Input argument consumed by `kron_list` to perform this
                     operation.

Returns:
    object: Output produced by this routine according to the behavior described
            above (compute the kronecker product of a list of operators.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        result = operators[0]
        for op in operators[1:]:
            # Left-associative Kronecker chaining over ordered subsystem ops.
            result = np.kron(result, op)
        return result

    def verify_x_stabilizers(self, state_vector):
        """
Verify that the X-type stabilizers leave the state invariant.

Stabilizers: S1 = X0 * X1 * X2^dagger * I * I;
S2 = I * X1^dagger * I * X3 * X4^dagger
Args:
    state_vector (Any): Input argument consumed by `verify_x_stabilizers` to
                        perform this operation.

Returns:
    object: Output produced by this routine according to the behavior described
            above (verify that the X-type stabilizers leave the
            state invariant.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        X = cirq.unitary(QuditX(self.dimension))
        X_inv = cirq.unitary(QuditX(self.dimension)**-1)
        I = np.eye(self.dimension)
        stab1_ops = [X, X, X_inv, I, I]
        stab2_ops = [I, X_inv, I, X, X_inv]
        stab1 = self.kron_list(stab1_ops)
        stab2 = self.kron_list(stab2_ops)
        # Stabilizer validation criterion: S|psi> = |psi>.
        valid1 = np.allclose(stab1 @ state_vector, state_vector)
        valid2 = np.allclose(stab2 @ state_vector, state_vector)
        return valid1, valid2

    def verify_z_stabilizers(self, state_vector):
        """
Verify that the Z-type stabilizers leave the state invariant.

Stabilizers: S1 = Z0 * Z1^dagger * I * Z3^dagger * I;
S2 = I * Z1 * Z2 * I * Z4^dagger
Args:
    state_vector (Any): Input argument consumed by `verify_z_stabilizers` to
                        perform this operation.

Returns:
    object: Output produced by this routine according to the behavior described
            above (verify that the Z-type stabilizers leave the
            state invariant.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        Z = cirq.unitary(QuditZ(self.dimension))
        Z_inv = cirq.unitary(QuditZ(self.dimension)**'ct')
        I = np.eye(self.dimension)
        stab1_ops = [Z, Z_inv, I, Z_inv, I]
        stab2_ops = [I, Z, Z, I, Z_inv]
        stab1 = self.kron_list(stab1_ops)
        stab2 = self.kron_list(stab2_ops)
        # Stabilizer validation criterion: S|psi> = |psi>.
        valid1 = np.allclose(stab1 @ state_vector, state_vector)
        valid2 = np.allclose(stab2 @ state_vector, state_vector)
        return valid1, valid2

    def verify_logical_x(self, state_vector):
        """
Verify that the logical X operator (X0 * I * I * X3 * I) leaves the
state invariant.

Args:
    state_vector (Any): Input argument consumed by `verify_logical_x` to
                        perform this operation.

Returns:
    object: Output produced by this routine according to the behavior described
            above (verify that the logical X operator
            (X0 * I * I * X3 * I) leaves the state invariant.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        X = cirq.unitary(QuditX(self.dimension))
        I = np.eye(self.dimension)
        ops = [X, I, I, X, I]
        logical_x = self.kron_list(ops)
        return np.allclose(logical_x @ state_vector, state_vector)
