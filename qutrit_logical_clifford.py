"""
qutrit_logical_clifford.py

This module defines the QutritLogicalClifford class for generating the full
logical
Clifford group for a single qutrit encoding on a folded 5–qutrit grid. It
obtains the full
(unencoded) group expressions from the QutritCliffordGroup (in
qutrit_clifford.py) and then
maps each basic operation to its corresponding logical operator defined in
qutrit_folded_logical_clifford.py.
"""

import cirq
import re
from typing import Dict, List
from qutrit_clifford import QutritCliffordGroup
from qutrit_folded_logical_clifford import QutritFoldedLogicalClifford
from gates import ProductGate


class QutritLogicalClifford:
    """
    Builder for the full logical Clifford group on the folded 5-qutrit code.

    The class maps unencoded Clifford expressions (from
    `QutritCliffordGroup`) onto encoded logical circuits using folded logical
    primitives, then optionally merges adjacent single-qudit operations for a
    more compact circuit representation.

    Attributes:
        dimension (int): Local qudit dimension.
        full_expressions (List[str]): Unencoded Clifford expression labels.
        logicalOperatorMap (Dict[str, Callable[[], cirq.Circuit]]): Mapping of
            primitive symbols (`I`, `H`, `S`, `X`, `Z`) to logical circuit
            constructors.

    Methods:
        __init__(dimension=3): Initialize expression set and operator mapping.
        generateLogicalCircuit(expression): Convert one expression to circuit.
        mergeSingleQubitGates(circuit): Merge adjacent one-qudit operations.
        createMergedLogicalGroup(logicalGroup): Apply merging to all entries.
        get_group(): Return expression-to-logical-circuit dictionary.
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize logical Clifford expression sources and operator mapping.
        
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
        # Get the full (unencoded) group expressions from the
        # QutritCliffordGroup.
        qc = QutritCliffordGroup(dimension)
        self.full_expressions = qc.get_full_group_expressions()
        # Map basic symbols to their logical operator circuits (as lambda
        # functions so they’re freshly created)
        self.logicalOperatorMap = {
            'I': lambda: QutritFoldedLogicalClifford.logical_I(self.dimension),
            'H': lambda: QutritFoldedLogicalClifford.logical_Hadamard(
                self.dimension),
            'S': lambda: QutritFoldedLogicalClifford.logical_S(self.dimension),
            'X': lambda: QutritFoldedLogicalClifford.logical_X(self.dimension),
            'Z': lambda: QutritFoldedLogicalClifford.logical_Z(self.dimension),
        }

    def generateLogicalCircuit(self, expression: str) -> cirq.Circuit:
        """
Converts a group expression string (e.g. "H^2" or "X^2Z^2S^2HSH") into
a logical circuit.

Args:
    expression (str): Symbolic group expression string that will be parsed into
                      an executable circuit.

Returns:
    cirq.Circuit: Constructed object generated from the provided inputs.

Raises:
    KeyError: If this method encounters an invalid state while processing the
              provided inputs.
"""
        pattern = re.compile(r'([A-Z])(\^\d+)?')
        matches = pattern.findall(expression)
        circuit = cirq.Circuit()
        # Process matches in reverse order so that the circuit is built in the
        # intended order.
        for match in reversed(matches):
            base = match[0]
            power = int(match[1][1:]) if match[1] else 1
            if base in self.logicalOperatorMap:
                for _ in range(power):
                    # Each call produces a fresh circuit for that logical
                    # operator.
                    circuit += self.logicalOperatorMap[base]()
            else:
                raise KeyError(f"Base {base} not found in logicalOperatorMap.")
        return circuit

    def mergeSingleQubitGates(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """
Merges successive single–qudit operations in a circuit.

Args:
    circuit (cirq.Circuit): Circuit object consumed, transformed, or analyzed
                            by this method.

Returns:
    cirq.Circuit: Merged representation produced by combining compatible items.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        mergedCircuit = cirq.Circuit()
        qudits = sorted(circuit.all_qubits())
        currentGates: Dict[cirq.Qid, List[cirq.Gate]] = {q: [] for q in qudits}

        for moment in circuit:
            newOperations = []
            for op in moment:
                if len(op.qubits) == 1:
                    # Buffer single-qudit gates so each wire segment can be
                    # merged into one ProductGate later.
                    currentGates[op.qubits[0]].append(op.gate)
                else:
                    # Flush single-qudit gates before applying a multi–qudit
                    # operation.
                    for q in op.qubits:
                        if currentGates[q]:
                            if len(currentGates[q]) == 1:
                                newOperations.append(currentGates[q][0].on(q))
                            else:
                                newOperations.append(ProductGate(
                                    list(reversed(currentGates[q]))).on(q))
                            currentGates[q].clear()
                    newOperations.append(op)
            for op in newOperations:
                mergedCircuit.append(op)
        # Flush any remaining single-qudit gates.
        for q, gates in currentGates.items():
            if gates:
                if len(gates) == 1:
                    mergedCircuit.append(gates[0].on(q))
                else:
                    mergedCircuit.append(ProductGate(
                        list(reversed(gates))).on(q))
        return mergedCircuit

    def createMergedLogicalGroup(
            self,
            logicalGroup: Dict[str, cirq.Circuit]) -> Dict[str, cirq.Circuit]:
        """
For each logical group circuit, merges successive single–qudit gates.

Args:
    logicalGroup (Dict[str, cirq.Circuit]): Mapping from symbolic labels to
                                            logical circuits before or after
                                            single-qudit-gate merging.

Returns:
    Dict[str, cirq.Circuit]: Constructed object generated from the provided
                             inputs.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        mergedGroup = {}
        for expr, circuit in logicalGroup.items():
            mergedGroup[expr] = self.mergeSingleQubitGates(circuit)
        return mergedGroup

    def get_group(self) -> Dict[str, cirq.Circuit]:
        """
        Generates the full logical Clifford group as a dictionary mapping each
        group expression (string) to its merged logical circuit.
        
        Args:
        None: `get_group` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Dict[str, cirq.Circuit]:
                Requested data object loaded or assembled by this
                                     method.
        
        Raises:
        ValueError: If `get_group` receives inputs that are incompatible with
        its expected configuration.
        """
        # Create a dictionary mapping each expression to its (unmerged) logical
        # circuit.
        logicalGroup = {}
        for expr in self.full_expressions:
            # Convert each symbolic Clifford expression into a concrete folded
            # logical circuit.
            logicalGroup[expr] = self.generateLogicalCircuit(expr)
        # Merge single-qudit operations.
        mergedGroup = self.createMergedLogicalGroup(logicalGroup)
        return mergedGroup
