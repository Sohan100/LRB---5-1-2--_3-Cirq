"""
qutrit_logical_pauli.py

This module defines the QutritLogicalPauli class for generating the full
logical
Pauli group for a single qutrit encoding on a folded 5-qutrit
grid. It obtains the
Pauli group expressions from the PauliGroup class (in pauli.py) and then maps
each
basic operation (I, X, Z) to its corresponding logical operator defined in
qutrit_folded_logical_clifford.py.

Two public methods are provided:
    - get_group(): returns a dictionary mapping each logical Pauli expression
      (string)
      to the merged logical circuit.
    - get_group_with_active_indices(): returns a dictionary mapping each
      logical Pauli
      expression to a tuple (active_indices, merged logical circuit), where
      active indices
      are assigned as follows:
          'I'      â†’ [0, 1, 2, 3, 4]
          'X'      â†’ [0, 3]
          'Z'      â†’ [0, 2]
          'XZ'     â†’ [0, 2, 3]
          'X^2'    â†’ [0, 3]
          'Z^2'    â†’ [0, 2]
          'X^2Z'   â†’ [0, 2, 3]
          'Z^2X'   â†’ [0, 2, 3]
          'X^2Z^2' â†’ [0, 2, 3]
"""

import cirq
import re
from typing import List, Dict, Tuple
from pauli import PauliGroup
from qutrit_folded_logical_clifford import QutritFoldedLogicalClifford
from gates import ProductGate
from typing import Dict, List, Tuple


class QutritLogicalPauli:
    """
    Builder for encoded logical Pauli circuits on the folded qutrit code.

    The class starts from unencoded Pauli labels and maps each symbol to its
    folded-layout logical counterpart. It supports both:
    - expression-to-circuit mapping, and
    - expression-to-(`active_indices`, circuit) mapping used by logical noise
      channels.

    Attributes:
        dimension (int): Local qudit dimension.
        P3Group (List[str]): Pauli expression list imported from `PauliGroup`.
        logicalOperatorMap (Dict[str, Callable[[], cirq.Circuit]]): Symbol map
            for folded logical `I`, `X`, `Z` constructors.

    Methods:
        __init__(dimension=3): Initialize source expression list and mappings.
        generateCircuit(expression): Parse one Pauli expression to circuit.
        mergeSingleQubitGates(circuit): Merge adjacent single-qudit operations.
        createMergedLogicalGroup(logicalGroup): Merge all group entries.
        get_group(): Return expression-to-circuit dictionary.
        convert_group_with_active_indices(group): Attach active index metadata.
        get_group_with_active_indices(): Return metadata-enriched group map.
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize logical Pauli expression source and operator mapping.
        
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
        # Retrieve the Pauli group expressions from the PauliGroup class.
        pg = PauliGroup(dimension)
        # PauliGroup.group_expressions is expected to be:
        # ['I', 'X', 'Z', 'XZ', 'X^2', 'Z^2', 'X^2Z', 'Z^2X', 'X^2Z^2']
        self.P3Group = pg.group_expressions

        # Map basic symbols to their logical operator circuits using the folded
        # logical definitions.
        self.logicalOperatorMap = {
            'I': lambda: QutritFoldedLogicalClifford.logical_I(self.dimension),
            'X': lambda: QutritFoldedLogicalClifford.logical_X(self.dimension),
            'Z': lambda: QutritFoldedLogicalClifford.logical_Z(self.dimension),
        }

    def generateCircuit(self, expression: str) -> cirq.Circuit:
        """
Converts an expression (e.g. "X^2Z^2" or "XZ") into a logical circuit
by parsing each symbol (and its exponent) and concatenating the
corresponding circuits.

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
        # Process in reverse order so that the leftmost symbol ends up applied
        # last.
        for match in reversed(matches):
            base = match[0]
            power = int(match[1][1:]) if match[1] else 1
            if base in self.logicalOperatorMap:
                for _ in range(power):
                    circuit += self.logicalOperatorMap[base]()
            else:
                raise KeyError(f"Base {base} not found in logicalOperatorMap.")
        return circuit

    def mergeSingleQubitGates(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """
Merges successive singleâ€“qudit operations in the circuit.

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
            newOps = []
            for op in moment:
                if len(op.qubits) == 1:
                    # Buffer one-qudit gates to reduce depth via ProductGate
                    # merging on each wire.
                    currentGates[op.qubits[0]].append(op.gate)
                else:
                    for q in op.qubits:
                        if currentGates[q]:
                            if len(currentGates[q]) == 1:
                                newOps.append(currentGates[q][0].on(q))
                            else:
                                newOps.append(ProductGate(
                                    list(reversed(currentGates[q]))).on(q))
                            currentGates[q].clear()
                    newOps.append(op)
            for op in newOps:
                mergedCircuit.append(op)
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
For each logical group circuit, merge successive singleâ€“qudit
operations.

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
        Returns a dictionary mapping each logical Pauli expression (string) to
        the merged logical circuit.
        
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
        logicalGroup = {}
        for expr in self.P3Group:
            # Build each logical Pauli from its symbolic expression.
            logicalGroup[expr] = self.generateCircuit(expr)
        mergedGroup = self.createMergedLogicalGroup(logicalGroup)
        return mergedGroup

    def convert_group_with_active_indices(self,
                                          group: Dict[str, cirq.Circuit]) -> \
        Dict[str,
             Tuple[List[int], cirq.Circuit]]:
        """
Converts a dictionary mapping Pauli labels to circuits into a
dictionary mapping each label to a tuple (active_indices, circuit). The
active indices correspond to the following encoded ordering on the
folded grid: Index 0: GridQid(0,0) Index 1: GridQid(0,2) Index 2:
GridQid(1,1) Index 3: GridQid(2,0) Index 4: GridQid(2,2)

Args:
    group (Dict[str, cirq.Circuit]): Dictionary-like group representation
                                     consumed by this converter.

Returns:
    Dict[str,
             Tuple[List[int], cirq.Circuit]]: Converted
                                                            representation
                                                            derived from the
                                                            source input.

Raises:
    ValueError: If this method encounters an invalid state while processing the
                provided inputs.
"""
        new_group = {}
        for key, circuit in group.items():
            # Active indices specify where sampled logical noise should be
            # embedded in the full encoded block.
            if key == 'I':
                active = [0, 1, 2, 3, 4]
            elif key == 'X':
                active = [0, 3]
            elif key == 'Z':
                active = [0, 2]
            elif key == 'XZ':
                active = [0, 2, 3]
            elif key == 'X^2':
                active = [0, 3]
            elif key == 'Z^2':
                active = [0, 2]
            elif key == 'X^2Z':
                active = [0, 2, 3]
            elif key == 'Z^2X':
                active = [0, 2, 3]
            elif key == 'X^2Z^2':
                active = [0, 2, 3]
            else:
                raise ValueError(f"Unknown key: {key}")
            new_group[key] = (active, circuit)
        return new_group

    def get_group_with_active_indices(
            self) -> Dict[str, Tuple[List[int], cirq.Circuit]]:
        """
        Returns a dictionary mapping each logical Pauli expression to a tuple
        (active_indices, merged logical circuit).
        
        Args:
        None: `get_group_with_active_indices` relies on object state and
        accepts no additional inputs.
        
        Returns:
            Any: Requested data object loaded or assembled by this method.
        
        Raises:
        ValueError: If `get_group_with_active_indices` receives inputs that are
        incompatible with its expected configuration.
        """
        group = self.get_group()
        return self.convert_group_with_active_indices(group)
