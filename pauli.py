"""
pauli.py

This module defines the PauliGroup class for single–qutrit Pauli operations.
It uses your custom gate classes (e.g. QuditI, QuditX, QuditZ, ProductGate)
to build the group from string expressions.

The group expressions are defined as:
    ['I', 'X', 'Z', 'XZ', 'X^2', 'Z^2', 'X^2Z', 'Z^2X', 'X^2Z^2']
"""

import cirq
from gates import (
    ProductGate,
    QuditI,
    QuditX,
    QuditZ,
)  # Ensure these are defined.


class PauliGroup:
    """
    Builder for single-qutrit Pauli group expressions and gate objects.

    The class stores canonical Pauli-string expressions and provides parsing
    utilities to convert each string into composed custom gate objects from
    `gates.py`.

    Attributes:
        dimension (int): Local qudit dimension.
        operation_map (Dict[str, cirq.Gate]): Symbol-to-gate lookup for `I`,
            `X`, and `Z`.
        group_expressions (List[str]): Canonical expression ordering used to
            build the Pauli group dictionary.

    Methods:
        __init__(dimension=3): Configure dimension and expression catalog.
        parse_and_multiply(expression): Parse one string and return a composed
            gate.
        convert_strings_to_operations(group_strings): Batch-convert many
            expressions.
        get_group(): Return expression-to-gate dictionary.
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize the single-qutrit Pauli group helper.
        
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
        # Map symbols to the corresponding Pauli gate objects.
        self.operation_map = {
            'I': QuditI(dimension),
            'X': QuditX(dimension),
            'Z': QuditZ(dimension)
        }
        # Define the Pauli group expressions.
        self.group_expressions = [
            'I', 'X', 'Z', 'XZ', 'X^2', 'Z^2', 'X^2Z', 'Z^2X', 'X^2Z^2'
        ]

    def parse_and_multiply(self, expression: str):
        """
Parse an expression (e.g., "X^2", "XZ", "X^2Z^2") and return the
corresponding gate.

Args:
    expression (str): Symbolic group expression string that will be parsed into
                      an executable circuit.

Returns:
    object: Output produced by this routine according to the behavior described
            above (parse an expression (e.g., "x^2", "xz", "x^2z^2") and return
            the).

Raises:
    ValueError: If this method encounters an invalid state while processing the
                provided inputs.
"""
        i = 0
        result = None
        # Scan the expression left-to-right and compose one gate block at a
        # time (including optional power suffix).
        while i < len(expression):
            char = expression[i]
            if char not in self.operation_map:
                raise ValueError(f"Unknown operation symbol: {char}")
            # Default exponent is 1.
            exponent = 1
            i += 1
            if i < len(expression) and expression[i] == '^':
                i += 1
                exp_str = ''
                while i < len(expression) and expression[i].isdigit():
                    exp_str += expression[i]
                    i += 1
                exponent = int(exp_str)
            # Build the gate raised to the required exponent.
            gate = self.operation_map[char]
            current_gate = gate
            for _ in range(exponent - 1):
                current_gate = current_gate @ gate
            # Multiply sequentially.
            if result is None:
                result = current_gate
            else:
                # Preserve expression order: existing product then next block.
                result = result @ current_gate
        return result

    def convert_strings_to_operations(self, group_strings):
        """
Convert a list of Pauli expression strings into gate objects.

Args:
    group_strings (Any): Input argument consumed by
                         `convert_strings_to_operations` to perform this
                         operation.

Returns:
    object: Converted representation derived from the source input.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        results = []
        for expr in group_strings:
            # Reuse single-expression parser so all syntax validation lives in
            # one place.
            results.append(self.parse_and_multiply(expr))
        return results

    def get_group(self):
        """
        Return a dictionary mapping each Pauli group expression to the
        corresponding gate.
        
        Args:
        None: `get_group` relies on object state and accepts no additional
        inputs.
        
        Returns:
            object: Requested data object loaded or assembled by this method.
        
        Raises:
        ValueError: If `get_group` receives inputs that are incompatible with
        its expected configuration.
        """
        ops = self.convert_strings_to_operations(self.group_expressions)
        return dict(zip(self.group_expressions, ops))
