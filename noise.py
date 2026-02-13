"""
noise.py
========

Noise-channel definitions for physical and logical qutrit benchmarking.

Contents:
- `QuditDepolarizingChannel`: single-qudit depolarizing channel represented as
  a probabilistic mixture over generalized Pauli operators.
- `kraus(...)`: helper to construct Kraus operators for depolarizing noise.
- `LogicalDepolarizingNoise`: logical-channel sampler that maps logical Pauli
  templates onto an encoded qudit block using active-index remapping.
"""

import itertools
import numpy as np
import cirq
from gates import QuditX, QuditZ


class QuditDepolarizingChannel(cirq.Gate):
    """
    This class represents a depolarizing channel for qudits of a given
    dimension.

    Attributes:
        dimension (int): The dimension of the qudit must be at least 2.
        errorProbability (float): The probability of depolarizing error,
                                  between
            0 and 1.
        errorProbabilities (dict): A dictionary mapping error operator
                                   exponents
            to their probabilities.

    Methods:
        _calcProbs_(): Computes the error probabilities for the depolarizing
            channel.
        _mixture_(): Returns a list of (probability, operator) tuples
                     representing
            the depolarizing channel.
        _has_mixture_(): Indicates that this channel can be represented as a
            mixture of operators.
        _value_equality_values_(): Returns the value used for equality
                                   comparison.
        _circuit_diagram_info_(args): Returns the gate's label for circuit
            diagrams.
        _qid_shape_(): Returns the shape of the qudit (dimension).
        _num_qudits_(): Returns the number of qudits this channel acts on.
    """

    def __init__(self, dimension: int, errorProbability: float):
        """
        Initializes the QuditDepolarizingChannel class through assigning it a
        dimension, error probability and explicit matrix.
        
        Args:
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
            errorProbability (float):
                Input argument consumed by `__init__` to perform
                                      this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If validation fails: Dimension must be at least 2.
            ValueError:
                If validation fails: Error probability must be between 0 and 1
                        inclusively.
        """
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
        if errorProbability > 1 or errorProbability < 0:
            raise ValueError(
                "Error probability must be between 0 and 1 inclusively")
        self.dimension = dimension
        self.errorProbability = errorProbability
        self.errorProbabilities = self._calcProbs_()

    def _calcProbs_(self) -> dict:
        """
        Computes the error probabilities for the depolarizing channel composed
        of different Kraus operators.
        
        Args:
        None: `_calcProbs_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            dict: Result object produced by this method.
        
        Raises:
        ValueError: If `_calcProbs_` receives inputs that are incompatible with
        its expected configuration.
        """
        errorProbabilities = {}
        # Depolarizing model: identity gets residual probability, all non-
        # identity generalized Pauli errors share the remainder uniformly.
        probI = 1 - self.errorProbability * (1 - 1 / (self.dimension**2))
        probKraus = self.errorProbability / (self.dimension**2)
        for exponents in itertools.product(range(self.dimension),
                                           range(self.dimension)):
            if exponents == (0, 0):
                errorProbabilities[exponents] = probI
            else:
                errorProbabilities[exponents] = probKraus
        return errorProbabilities

    def _mixture_(self) -> list:
        """
        Returns a list of (probability, operator) tuples representing the
        depolarizing channel.
        
        Args:
        None: `_mixture_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            list: Result object produced by this method.
        
        Raises:
        ValueError: If `_mixture_` receives inputs that are incompatible with
        its expected configuration.
        """

        def operators(xExponent, zExponent): return \
            cirq.unitary(QuditX(self.dimension)**xExponent) @ \
            cirq.unitary(QuditZ(self.dimension)**zExponent)
        # Build Cirq mixture tuples (probability, unitary).
        sequences = [(self.errorProbabilities[(xExponent, zExponent)],
                      operators(xExponent, zExponent))
                     for (xExponent, zExponent)
                     in self.errorProbabilities.keys()]
        return sequences

    def _has_mixture_(self) -> bool:
        """
        Indicates that this channel can be represented as a mixture of Kraus
        operators.
        
        Args:
        None: `_has_mixture_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            bool: Result object produced by this method.
        
        Raises:
        ValueError: If `_has_mixture_` receives inputs that are incompatible
        with its expected configuration.
        """
        return True

    def _value_equality_values_(self) -> float:
        """
        Returns the value used for equality comparison.
        
        Args:
        None: `_value_equality_values_` relies on object state and accepts no
        additional inputs.
        
        Returns:
            float: Result object produced by this method.
        
        Raises:
        ValueError: If `_value_equality_values_` receives inputs that are
        incompatible with its expected configuration.
        """
        return self.errorProbability

    def _circuit_diagram_info_(self, args) -> tuple:
        """
Provides a label for the channel in circuit diagrams.

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
        if args.precision is not None:
            label = (
                f"D{self.dimension}"
                f"({self.errorProbability:.{args.precision}g})"
            )
            return (label,)
        else:
            return (f"D{self.dimension}({self.errorProbability})",)

    def _qid_shape_(self) -> tuple:
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
        Returns the number of qudits this channel acts upon.
        
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


def kraus(dimension: int, errorProbability: float):
    """
Constructs a list of Kraus operators for the depolarizing channel.

Args:
    dimension (int): Local Hilbert-space dimension for the qudit system
                     represented by this operation.
    errorProbability (float): Input argument consumed by `kraus` to perform
                              this operation.

Returns:
    object: Output produced by this routine according to the behavior described
            above (constructs a list of kraus operators for the depolarizing
            channel.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
    kraussList = []
    # Identity component captures "no-error" branch probability.
    K0 = np.sqrt(1 - errorProbability * ((dimension**2 - 1) /
                                         (dimension**2))) * np.eye(dimension)
    kraussList.append(K0)
    for i in range(dimension):
        for j in range(dimension):
            if not (i == 0 and j == 0):
                # Each non-identity generalized Pauli gets equal depolarizing
                # weight in this model.
                pCoeff = np.sqrt(errorProbability / (dimension**2))
                kOperator = pCoeff * (cirq.unitary(QuditX(dimension)**i) @
                                      cirq.unitary(QuditZ(dimension)**j)
                                      )
                kOperator /= np.linalg.norm(kOperator)
                kraussList.append(kOperator)
    return kraussList


class LogicalDepolarizingNoise(cirq.Gate):
    """
    A logical depolarizing noise channel that samples a logical Pauli operator
    from a predefined group.

    The group is provided as a dictionary with keys as labels (e.g. 'I', 'X',
    'Z', etc.) and values as tuples of the form (active_indices,
    logical_circuit).
    active_indices is a list of indices on which that logical operator acts,
    and
    logical_circuit is a cirq.Circuit defined on dummy qudits (with length
    equal to
    len(active_indices)).

    The sampling probabilities are defined as:
      p_I = 1 - errorProbability * (1 - 1/(dimension**2))
      p_other = errorProbability / (dimension**2)

    Attributes:
        errorProbability (float): Depolarizing error probability.
        mergedLogicalP3Group (dict): Maps labels to tuples
            (active_indices, logical_circuit).
        dimension (int): Local dimension (default 3).
        full_qudit_count (int): Total number of qudits in the encoded block.
        keys (List[str]): List of keys from the dictionary.
        probs (List[float]): Sampling probabilities for each operator.
        last_sampled_key (str or None): The key of the most recently sampled
            operator.
    """

    def __init__(self, errorProbability: float, mergedLogicalP3Group: dict,
                 full_qudit_count: int, dimension: int = 3):
        """
        Initialize logical depolarizing-channel sampling configuration.
        
        Args:
            errorProbability (float):
                Input argument consumed by `__init__` to perform
                                      this operation.
            mergedLogicalP3Group (dict):
                Input argument consumed by `__init__` to
                                         perform this operation.
            full_qudit_count (int):
                Input argument consumed by `__init__` to perform
                                    this operation.
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError:
                If validation fails: Error probability must be between 0 and 1
                        inclusively.
        """
        if errorProbability < 0 or errorProbability > 1:
            raise ValueError(
                "Error probability must be between 0 and 1 inclusively")
        self.errorProbability = errorProbability
        self.mergedLogicalP3Group = mergedLogicalP3Group
        self.dimension = dimension
        self.full_qudit_count = full_qudit_count
        p_I = 1 - errorProbability * (1 - 1 / (dimension**2))
        p_other = errorProbability / (dimension**2)
        self.keys = list(mergedLogicalP3Group.keys())
        self.probs = [p_I if key == 'I' else p_other for key in self.keys]
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]
        self.last_sampled_key = None

    def _num_qubits_(self) -> int:
        """
        Returns the number of qudits this noise gate acts on.
        
        Args:
        None: `_num_qubits_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            int: Result object produced by this method.
        
        Raises:
        ValueError: If `_num_qubits_` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.full_qudit_count

    def _qid_shape_(self) -> tuple:
        """
        Returns the shape of each qudit.
        
        Args:
        None: `_qid_shape_` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Tuple[int, ...]: Qid shape expected by Cirq for this gate.
        
        Raises:
        ValueError: If `_qid_shape_` receives inputs that are incompatible with
        its expected configuration.
        """
        return (self.dimension,) * self.full_qudit_count

    def _decompose_(self, qubits):
        """
        Decomposes the noise channel by sampling a logical operator from the
        group. The group entry is a tuple (active_indices, noise_circuit). The
        dummy qudits in noise_circuit are remapped onto the qubits at the
        specified active indices.
        
        Args:
            qubits (Any):
                Input argument consumed by `_decompose_` to perform this
                          operation.
        
        Returns:
        None: `_decompose_` updates internal object state and returns no value.
        
        Raises:
            ValueError:
                If this method encounters an invalid state while processing the
                        provided inputs.
        """
        chosen_key = np.random.choice(self.keys, p=self.probs)
        self.last_sampled_key = chosen_key
        # Group entry stores both support indices and the corresponding logical
        # circuit template.
        active_indices, noise_circuit = self.mergedLogicalP3Group[chosen_key]
        dummy_qubits = sorted(
            noise_circuit.all_qubits(),
            key=lambda q: q.name if hasattr(
                q,
                'name') else str(q))
        if len(dummy_qubits) != len(active_indices):
            raise ValueError(
                f"Mismatch: noise circuit for {chosen_key} is defined on "
                f"{len(dummy_qubits)} qudits, but active_indices has length "
                f"{len(active_indices)}."
            )
        mapping = {dummy: qubits[i] for dummy, i in zip(dummy_qubits,
                                                        active_indices)}
        # Remap template-circuit dummy qudits onto actual encoded-block qudits.
        remapped_circuit = noise_circuit.transform_qubits(
            lambda q: mapping.get(q, q))
        for op in remapped_circuit.all_operations():
            yield op

    def _circuit_diagram_info_(self, args) -> tuple:
        """
Provides a label for the noise channel in circuit diagrams. If a
logical operator has been sampled, only the qudits corresponding to its
active indices display its label, while others show a blank symbol.

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
        if self.last_sampled_key is not None:
            active_indices, _ = self.mergedLogicalP3Group[
                self.last_sampled_key
            ]
            labels = []
            for i in range(self.full_qudit_count):
                if i in active_indices:
                    labels.append(f"{self.last_sampled_key}")
                else:
                    labels.append("")
            return tuple(labels)
        else:
            return ("LD?",) * self.full_qudit_count
