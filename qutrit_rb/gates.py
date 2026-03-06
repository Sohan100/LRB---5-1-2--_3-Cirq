"""
gates.py
========

Core qudit gate and channel primitives used across physical and logical qutrit
simulations in this repository.

This module defines:
- Single-qudit gates (`QuditI`, `QuditX`, `QuditZ`, `QuditS`, `QuditHadamard`)
- Multi-qudit gates (`QuditCNOT`, `QuditCZ`, `QuditSwap`)
- Gate-composition helpers (`ProductGate`)

Design goals:
- Provide Cirq-compatible custom gate classes for arbitrary qudit dimension.
- Support algebraic gate composition with readable diagram symbols.
- Keep matrix/unitary behavior explicit for debugging and verification tasks.
"""

import cirq
import itertools
import numpy as np
from typing import List, Union, Tuple, Dict
from scipy.linalg import block_diag
from typing import Type
import random
import copy
from scipy.optimize import curve_fit, root_scalar
import matplotlib.pyplot as plt
from scipy import interpolate
import re


class QuditI(cirq.Gate):
    """This class represents the Identity Gate for qudits of a given
    dimension.

    Attributes:
        dimension (int): The dimension of the qudit this gate acts upon.
        dagger (int): Indicates whether the matrix is Hermitian adjoint.
        matrix (np.ndarray): The unitary matrix of the Identity Gate.

    Methods:
        _unitary_(): Returns the unitary matrix of the gate.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a given power.
        _circuit_diagram_info_(args): Provides a label for the circuit.
        _qid_shape_(): Returns the qudit shape.
        _num_qudits_(): Returns the number of qudits (always 1).
    """

    def __init__(self, dimension: int):
        """
        Initializes the QuditI gate with a given dimension.
        
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
        self.dagger = 0
        self.matrix = self._unitary_()

    def _unitary_(self):
        """
        Returns the identity matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return np.eye(self.dimension)

    def __matmul__(self, other):
        """Overrides the @ operator for matrix multiplication.

        Args:
            other (cirq.Gate): Another gate.

        Returns:
            ProductGate: A product gate representing self @ other.

        Raises:
            ValueError: If other is not a cirq.Gate.
        """
        if isinstance(other, cirq.Gate):
            return ProductGate([self, other])
        else:
            raise ValueError("Can only matrix multiply Gates")

    def __pow__(self, exponent):
        """
Raises the Identity gate to any power (remains unchanged).

Args:
    exponent (Any): Power applied to the current gate; negative values request
                    inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return self

    def _circuit_diagram_info_(self, args):
        """
Return the concise circuit-diagram symbol for Identity (`I`).

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return "I"

    def _qid_shape_(self):
        """
        Returns the qudit shape as a one-element tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qudits_(self):
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies the Identity gate to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies the identity gate
                          to each qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class ProductGate(cirq.Gate):
    """Represents the product of two or more gates in d dimensions.

    Attributes:
        gates (List[cirq.Gate]): The list of gates.
        dimension (int): The qudit dimension.
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The product unitary matrix.
        symbol (str): The symbol used in diagrams.

    Methods:
        _create_matrix(): Computes the unitary matrix.
        _unitary_(): Returns the unitary matrix.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a power or adjoint.
        _create_symbol(): Creates the diagram symbol.
        _format_gate_symbol(symbol, gate, power): Helper to format symbols.
        _circuit_diagram_info_(args): Provides a diagram label.
        _qid_shape_(): Returns the qudit shape.
        _num_qubits_(): Returns the number of qudits (always 1).
        on_each(qudits): Applies the gate to a list of qudits.
    """

    def __init__(self, gates: List[cirq.Gate]):
        """
        Initializes the ProductGate with a list of gates.
        
        Args:
            gates (List[cirq.Gate]):
                Collection of gates supplied to the randomized-
                                     benchmarking sequence builder.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            TypeError: If validation fails: gates must be a list.
            TypeError:
                If validation fails: All elements in gates must be instances of
                       cirq.Gate.
            ValueError:
                If validation fails: gates list must contain more than one
                        gate.
        """
        super().__init__()
        if not isinstance(gates, list):
            raise TypeError("gates must be a list")
        if not all(isinstance(gate, cirq.Gate) for gate in gates):
            raise TypeError(
                "All elements in gates must be instances of cirq.Gate")
        if len(gates) <= 1:
            raise ValueError("gates list must contain more than one gate")
        self.gates = gates
        # Infer qudit dimension from first operand; constructor validation
        # assumes all operands are dimension-compatible.
        self.dimension = cirq.unitary(self.gates[0]).shape[0]
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()
        self.symbol = self._create_symbol()

    def _create_matrix(self) -> np.ndarray:
        """
        Computes the unitary matrix of the product.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        mats = [cirq.unitary(gate) for gate in self.gates]
        productMat = np.eye(self.dimension, dtype=np.complex128)
        # Compose gate matrices in listed order to match ProductGate semantics.
        for mat in mats:
            productMat @= mat
        return productMat

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix of the product gate.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __matmul__(self, other: cirq.Gate) -> 'ProductGate':
        """
Overrides the @ operator for matrix multiplication.

Args:
    other (cirq.Gate): Gate instance to compose with this gate using matrix-
                       product order `self @ other`.

Returns:
    ProductGate: Composite gate representing matrix-product composition `self @
                 other`.

Raises:
    ValueError: If validation fails: Can only matrix multiply gates.
"""
        if isinstance(other, cirq.Gate):
            if isinstance(other, ProductGate):
                return ProductGate(self.gates + other.gates)
            else:
                return ProductGate(self.gates + [other])
        else:
            raise ValueError("Can only matrix multiply gates")

    def __pow__(self, exponent: Union[int, str]) -> 'ProductGate':
        """
Raises the product gate to a given power or adjoint.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger += 1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _create_symbol(self) -> str:
        """
        Creates the symbol for circuit diagrams.
        
        Args:
        None: `_create_symbol` relies on object state and accepts no additional
        inputs.
        
        Returns:
            str: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_symbol` receives inputs that are incompatible
        with its expected configuration.
        """
        symbol = ""
        currentGate = None
        currentPower = 0
        for gate in self.gates:
            gateSymbol = cirq.circuit_diagram_info(gate).wire_symbols[0]
            isInvertedProduct = gateSymbol.startswith("([") and \
                gateSymbol.endswith("^-1)")
            if isInvertedProduct:
                symbol += gateSymbol
                continue
            baseSymbol = "".join(filter(str.isalpha, gateSymbol))
            if baseSymbol == "I":
                continue
            powerMatch = re.search(r"\^(-?\d+)", gateSymbol)
            power = int(powerMatch.group(1)) if powerMatch else 1
            if baseSymbol == currentGate:
                currentPower += power
            else:
                if currentGate:
                    symbol += self._format_gate_symbol(symbol,
                                                       currentGate,
                                                       currentPower)
                currentGate = baseSymbol
                currentPower = power
        if currentGate:
            symbol += self._format_gate_symbol(symbol, currentGate,
                                               currentPower)
        return symbol if symbol else "I"

    def _format_gate_symbol(self, symbol: str, gate: str,
                            power: int) -> str:
        """
Helper to format gate symbols.

Args:
    symbol (str): Input argument consumed by `_format_gate_symbol` to perform
                  this operation.
    gate (str): Input argument consumed by `_format_gate_symbol` to perform
                this operation.
    power (int): Input argument consumed by `_format_gate_symbol` to perform
                 this operation.

Returns:
    str: Output produced by this routine according to the behavior described
         above (helper to format gate symbols.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if power == 0:
            return ""
        elif power == 1:
            return gate
        else:
            return f"{gate}^{power}"

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides a label for the circuit diagram.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return self.symbol + "†"
        if self.expo != 1:
            return "([" + self.symbol + "]^" + str(self.expo) + ")"
        return cirq.CircuitDiagramInfo(wire_symbols=(self.symbol,))

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the qudit shape as a one-element tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qubits_(self) -> int:
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qubits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Result object produced by this method.
        
        Raises:
        ValueError: If `_num_qubits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies the product gate to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies the product gate to
                          each qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class QuditZ(cirq.Gate):
    """Represents the generalized Pauli Z gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks if the matrix is Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The QuditZ gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides a diagram label.
        _qid_shape_(): Returns the qudit shape.
        _num_qudits_(): Returns the number of qudits (always 1).
        on_each(qudits): Applies the gate to a list of qudits.
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditZ with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the QuditZ gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        matrix = np.diag([np.exp(2j * np.pi * k / self.dimension)
                          for k in range(self.dimension)])
        return matrix

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __matmul__(self, other: cirq.Gate) -> 'ProductGate':
        """
Overrides the @ operator for matrix multiplication.

Args:
    other (cirq.Gate): Gate instance to compose with this gate using matrix-
                       product order `self @ other`.

Returns:
    ProductGate: Composite gate representing matrix-product composition `self @
                 other`.

Raises:
    ValueError: If validation fails: Can only matrix multiply Gates.
"""
        if isinstance(other, cirq.Gate):
            return ProductGate([self, other])
        else:
            raise ValueError("Can only matrix multiply Gates")

    def __pow__(self, exponent: Union[int, str]) -> 'QuditZ':
        """
Raises the QuditZ gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = + 1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides the diagram label.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "Z†"
        if self.expo != 1:
            return "(Z^" + str(self.expo) + ")"
        return "Z"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the qudit shape as a tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies QuditZ to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies quditz to each
                          qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class QuditX(cirq.Gate):
    """Represents the generalized Pauli X gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The QuditX gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides a diagram label.
        _qid_shape_(): Returns the qudit shape.
        _num_qudits_(self): Returns the number of qudits (always 1).
        on_each(qudits): Applies the gate to a list of qudits.
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditX with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the QuditX gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        shift_matrix = np.zeros((self.dimension, self.dimension),
                                dtype=np.complex128)
        # Cyclic computational-basis shift: |i> -> |i+1 mod d>.
        for i in range(self.dimension):
            shift_matrix[(i + 1) % self.dimension, i] = 1
        return shift_matrix

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __matmul__(self, other: cirq.Gate) -> 'ProductGate':
        """
Overrides the @ operator for matrix multiplication.

Args:
    other (cirq.Gate): Gate instance to compose with this gate using matrix-
                       product order `self @ other`.

Returns:
    ProductGate: Composite gate representing matrix-product composition `self @
                 other`.

Raises:
    ValueError: If validation fails: Can only matrix multiply Gates.
"""
        if isinstance(other, cirq.Gate):
            return ProductGate([self, other])
        else:
            raise ValueError("Can only matrix multiply Gates")

    def __pow__(self, exponent: Union[int, str]) -> 'QuditX':
        """
Raises the QuditX gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = +1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides the diagram label.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "X†"
        if self.expo != 1:
            return "(X^" + str(self.expo) + ")"
        return "X"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the qudit shape as a tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies QuditX to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies quditx to each
                          qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class QuditS(cirq.Gate):
    """Generalizes the Phase (S) gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The Qudit Phase gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides a diagram label.
        _qid_shape_(): Returns the qudit shape.
        _num_qudits_(self): Returns the number of qudits (always 1).
        on_each(qudits): Applies the gate to a list of qudits.
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditS with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the Qudit Phase gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        w = np.exp(-2j * np.pi / self.dimension)
        # Diagonal phase polynomial for generalized S.
        matrix = np.diag([w**((i - self.dimension - 2) * i / 2)
                          for i in range(self.dimension)])
        return matrix

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __matmul__(self, other: cirq.Gate) -> 'ProductGate':
        """
Overrides the @ operator for matrix multiplication.

Args:
    other (cirq.Gate): Gate instance to compose with this gate using matrix-
                       product order `self @ other`.

Returns:
    ProductGate: Composite gate representing matrix-product composition `self @
                 other`.

Raises:
    ValueError: If validation fails: Can only matrix multiply Gates.
"""
        if isinstance(other, cirq.Gate):
            return ProductGate([self, other])
        else:
            raise ValueError("Can only matrix multiply Gates")

    def __pow__(self, exponent: Union[int, str]) -> 'QuditS':
        """
Raises the QuditS gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = +1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides the diagram label.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "S†"
        if self.expo != 1:
            return "(S^" + str(self.expo) + ")"
        return "S"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the qudit shape as a tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies QuditS to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies qudits to each
                          qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class QuditHadamard(cirq.Gate):
    """Generalizes the Hadamard gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The Qudit Hadamard gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __matmul__(other): Matrix multiplies with another gate.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides a diagram label.
        _qid_shape_(): Returns the qudit shape.
        _num_qudits_(self): Returns the number of qudits (always 1).
        on_each(qudits): Applies the gate to a list of qudits.
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditHadamard with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the Qudit Hadamard gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        w = np.exp(2j * np.pi / self.dimension)
        matrix = np.zeros((self.dimension, self.dimension),
                          dtype=np.complex128)
        # Discrete Fourier transform over d-level computational basis.
        for i in range(self.dimension):
            for j in range(self.dimension):
                matrix[i, j] = w**(i * j)
        normalized = matrix / np.sqrt(self.dimension)
        return normalized

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __matmul__(self, other: cirq.Gate) -> 'ProductGate':
        """
Overrides the @ operator for matrix multiplication.

Args:
    other (cirq.Gate): Gate instance to compose with this gate using matrix-
                       product order `self @ other`.

Returns:
    ProductGate: Composite gate representing matrix-product composition `self @
                 other`.

Raises:
    ValueError: If validation fails: Can only matrix multiply Gates.
"""
        if isinstance(other, cirq.Gate):
            return ProductGate([self, other])
        else:
            raise ValueError("Can only matrix multiply Gates")

    def __pow__(self, exponent: Union[int, str]) -> 'QuditHadamard':
        """
Raises the QuditHadamard gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = + 1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides the diagram label.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "H†"
        if self.expo != 1:
            return "(H^" + str(self.expo) + ")"
        return "H"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the qudit shape as a tuple.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 1).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 1

    def on_each(self, qudits: List[cirq.Qid]) -> List[cirq.Operation]:
        """
Applies QuditHadamard to each qudit in the list.

Args:
    qudits (List[cirq.Qid]): Ordered qudit register used by this experiment or
                             circuit constructor.

Returns:
    List[cirq.Operation]: Output produced by this routine according to the
                          behavior described above (applies qudithadamard to
                          each qudit in the list.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return [self(q) for q in qudits]


class QuditSwap(cirq.Gate):
    """Represents the generalized Swap gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        matrix (np.ndarray): The Qudit Swap gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides diagram labels.
        _qid_shape_(): Returns the qudit shapes.
        _num_qudits_(self): Returns the number of qudits (always 2).
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditSwap with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the Qudit Swap gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        swapMatrix = np.zeros((self.dimension**2, self.dimension**2),
                              dtype=np.complex128)
        # Basis permutation implementing |i,j> -> |j,i>.
        for i in range(self.dimension):
            for j in range(self.dimension):
                swapMatrix[i * self.dimension + j][j * self.dimension + i] = 1
        return swapMatrix

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __pow__(self, exponent: Union[int, str]) -> 'QuditSwap':
        """
Raises the QuditSwap gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides the diagram labels for the swap gate.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return "x", "x"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the shapes of the two qudits.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension, self.dimension)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 2).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 2


class QuditCZ(cirq.Gate):
    """Represents the generalized CZ gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The QuditCZ gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides diagram labels.
        _qid_shape_(): Returns the qudit shapes.
        num_qudits(): Returns the number of qudits (always 2).
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditCZ with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the QuditCZ gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        d = self.dimension
        matrix = np.eye(d**2, dtype=complex)
        w = np.exp(2j * np.pi / d)
        # Controlled phase applies only when both control/target are non-zero.
        for x in range(d):
            for y in range(d):
                if x > 0 and y > 0:
                    index = x * d + y
                    matrix[index, index] = w ** (x * y)
        return matrix

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __pow__(self, exponent: Union[int, str]) -> 'QuditCZ':
        """
Raises the QuditCZ gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = +1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides diagram labels for the CZ gate.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "@", "Z†"
        if self.expo != 1:
            return "@", "(Z^" + str(self.expo) + ")"
        return "@", "Z"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the shapes of the two qudits.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension, self.dimension)

    def num_qudits(self) -> int:
        """
        Returns the number of qudits (always 2).
        
        Args:
        None: `num_qudits` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Result object produced by this method.
        
        Raises:
        ValueError: If `num_qudits` receives inputs that are incompatible with
        its expected configuration.
        """
        return 2


class QuditCNOT(cirq.Gate):
    """Represents the generalized CNOT (CX) gate for d-dimensional qudits.

    Attributes:
        dimension (int): The qudit dimension (>= 2).
        dagger (int): Tracks Hermitian adjoint.
        expo (int): The gate exponent.
        matrix (np.ndarray): The QuditCNOT gate matrix.

    Methods:
        _create_matrix(): Computes the matrix.
        _unitary_(): Returns the unitary matrix.
        __pow__(exponent): Raises the gate to a power.
        _circuit_diagram_info_(args): Provides diagram labels.
        _qid_shape_(): Returns the qudit shapes.
        _num_qudits_(self): Returns the number of qudits (always 2).
    """

    def __init__(self, dimension: int):
        """
        Initializes QuditCNOT with a given dimension.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        self.dimension = dimension
        self.dagger = 0
        self.expo = 1
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> np.ndarray:
        """
        Generates the QuditCNOT gate matrix.
        
        Args:
        None: `_create_matrix` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Result object produced by this method.
        
        Raises:
        ValueError: If `_create_matrix` receives inputs that are incompatible
        with its expected configuration.
        """
        CX = np.zeros((self.dimension**2, self.dimension**2))
        # Explicit basis-action construction of generalized CNOT.
        for x in range(self.dimension):
            for y in range(self.dimension):
                CX += np.kron(
                    np.outer(
                        np.eye(
                            self.dimension)[
                            :, x], np.eye(
                            self.dimension)[
                            :, x]), np.outer(
                        np.eye(
                            self.dimension)[
                            :, (x + y) %
                            self.dimension], np.eye(
                            self.dimension)[
                            :, y]))
        return CX

    def _unitary_(self) -> np.ndarray:
        """
        Returns the unitary matrix.
        
        Args:
        None: `_unitary_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            np.ndarray: Unitary matrix representation of the gate.
        
        Raises:
        ValueError: If `_unitary_` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.matrix

    def __pow__(self, exponent: Union[int, str]) -> 'QuditCNOT':
        """
Raises the QuditCNOT gate to a given power.

Args:
    exponent (Union[int, str]): Power applied to the current gate; negative
                                values request inverse behavior when supported.

Returns:
    cirq.Gate: Gate instance implementing the requested exponentiation
               behavior.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if exponent == "ct":
            self.matrix = np.array(np.asmatrix(self.matrix).getH())
            self.dagger = + 1
            return self
        self.matrix = np.linalg.matrix_power(self.matrix, exponent)
        self.expo *= exponent
        return self

    def _circuit_diagram_info_(self, args) -> str:
        """
Provides diagram labels for the CNOT gate.

Args:
    args (Any): Formatting metadata supplied by Cirq when requesting printable
                circuit-diagram labels.

Returns:
    Tuple[str, ...] | str: Circuit-diagram label(s) that Cirq renders for this
                           gate.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if self.dagger % 2 != 0:
            return "@", "Z†"
        if self.expo != 1:
            return "@", "(X^" + str(self.expo) + ")"
        return "@", "X"

    def _qid_shape_(self) -> Tuple[int]:
        """
        Returns the shapes of the two qudits.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension, self.dimension)

    def _num_qudits_(self) -> int:
        """
        Returns the number of qudits (always 2).
        
        Args:
        None: `_num_qudits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Number of qudits this gate acts on.
        
        Raises:
        ValueError: If `_num_qudits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return 2
