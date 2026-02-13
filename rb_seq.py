"""
rb_seq.py

This module defines the RBSeq class which generates randomized
benchmarking sequences for a given set of qudits and gates
(e.g. single-qutrit gates). It follows the standard
protocol for randomized benchmarking. The class supports:
protocol for randomized benchmarking. The class supports:
  - Generating RB sequences for various sequence lengths.
  - Converting sequences into Cirq Moments.
  - Building complete Cirq Circuits from the generated moments.
  - Optionally adding a noise channel after each gate.
  - Optionally applying a measurement basis transformation before measurement.

Attributes:
    qudits (List[cirq.Qid]): The list of qudits on which the circuits will be
                             built.
    gateSet (List[cirq.Gate]): The set of available gates used to construct RB
                               sequences.
    sequenceLengths (List[int]): A list of sequence lengths (number of gates,
                                 plus one for the inverse).
    circuitsPerSequence (int): Number of circuits to generate for each sequence
                               length.
    measurementBasisTransformation (cirq.Gate): Gate for transforming
                                                measurement basis.
    seed (int): Seed for the random number generator.
    noiseChannel (cirq.Gate): Optional noise channel to insert after each gate.

Public Methods:
    getrbSeqs() -> List[List[Union[cirq.Gate, str]]]:
        Returns the generated RB sequences (as lists of gates and a final "M"
        marker).
    getrbMoments() -> List[List[cirq.Moment]]:
        Returns the RB sequences expressed as lists of Cirq Moments.
    getrbCircuits() -> List[cirq.Circuit]:
        Returns the RB sequences as complete Cirq Circuits.
"""

from typing import List, Union
import cirq
import random
import copy
from gates import ProductGate


class RBSeq:
    """
    Physical randomized benchmarking sequence and circuit generator.

    This class creates randomized gate sequences for specified depths, appends
    an inverse block, and converts the resulting sequences into Cirq moments
    and
    final circuits. Optional noise insertion and measurement-basis transforms
    are supported.

    Attributes:
        qudits (List[cirq.Qid]): Target qudits for sequence application.
        measurementBasisTransformation (cirq.Gate): Optional pre-measurement
            basis transform.
        gateSet (List[cirq.Gate]): Primitive gate pool for random sampling.
        sequenceLengths (List[int]): RB depth values.
        circuitsPerSequence (int): Number of randomized circuits per depth.
        rng (random.Random): Seeded random generator for reproducibility.
        noiseChannel (cirq.Gate): Optional channel inserted after each gate.
        rbSeqs (List[List[Union[cirq.Gate, str]]]): Sequence-level
            representation with terminal `"M"` markers.
        rbMoments (List[List[cirq.Moment]]): Moment-level representation.
        rbCircuits (List[cirq.Circuit]): Executable circuit-level
                                         representation.

    Methods:
        __init__(...): Configure generator and build all representations.
        genSeqs(): Generate randomized RB sequences with inverse gates.
        _add_noise(gates): Insert noise operations after gate entries.
        genMoms(): Convert sequence entries into Cirq moments.
        genCircs(): Convert moments into Cirq circuits.
        getrbSeqs(): Return sequence representation.
        getrbMoments()/get_rb_moments(): Return moment representation.
        getrbCircuits()/get_rb_circuits(): Return circuit representation.
    """

    def __init__(self, qudits: List[cirq.Qid], gates: List[cirq.Gate],
                 sequenceLengths: List[int] = [
                     0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                 circuitsPerSequence: int = 30,
                 measurementBasisTransformation: cirq.Gate = None,
                 seed: int = None, noiseChannel: cirq.Gate = None):
        """
        Initializes the RBSeq generator with the given parameters.
        
        Args:
            qudits (List[cirq.Qid]):
                Ordered qudit register used by this experiment or
                                     circuit constructor.
            gates (List[cirq.Gate]):
                Collection of gates supplied to the randomized-
                                     benchmarking sequence builder.
            sequenceLengths (List[int]):
                List of randomized-benchmarking sequence
                                         lengths to generate.
            circuitsPerSequence (int):
                Number of random circuits generated for each
                                       sequence length.
            measurementBasisTransformation (cirq.Gate):
                Optional circuit applied before
                                                        terminal measurement to
                                                        rotate
                                                        into the desired
                                                        readout basis.
            seed (int):
                Random-number-generator seed used to make circuit sampling
                        reproducible.
            noiseChannel (cirq.Gate):
                Noise-channel gate instance inserted according to
                                      this method's placement strategy.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.qudits = qudits
        self.measurementBasisTransformation = measurementBasisTransformation
        self.gateSet = gates
        self.sequenceLengths = sequenceLengths
        self.circuitsPerSequence = circuitsPerSequence
        self.rng = random.Random(seed)
        self.noiseChannel = noiseChannel

        self.rbSeqs: List[List[Union[cirq.Gate, str]]] = []
        self.depthKeys: List[str] = []
        self.rbMoments: List[List[cirq.Moment]] = []
        self.rbCircuits: List[cirq.Circuit] = []

        self.genSeqs()
        self.genMoms()
        self.genCircs()

    def genSeqs(self):
        """
        Generate randomized benchmarking sequences based on the specified
        sequence lengths. Each sequence consists of a random selection of gates
        from the available gate set, followed by a gate that is the inverse of
        the product of all the preceding gates, and a final marker "M" to
        denote measurement.
        
        Args:
        None: `genSeqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
        None: `genSeqs` executes for side effects only and returns no payload.
        
        Raises:
        ValueError: If `genSeqs` receives inputs that are incompatible with its
        expected configuration.
        """
        self.rbSeqs = []
        self.depthKeys = []
        # Outer loop selects the target RB depth; inner loop samples random
        # circuits at that fixed depth.
        for length in self.sequenceLengths:
            for i in range(self.circuitsPerSequence):
                cliffGates = []
                if length == 0:
                    # For zero-length sequence, use the identity.
                    cliffGates = [cirq.IdentityGate(qid_shape=(
                        cirq.unitary(self.gateSet[0]).shape[0],))]
                else:
                    # Randomly choose 'length' gates from the gateSet.
                    cliffGates = [self.rng.choice(
                        self.gateSet) for j in range(length)]
                    # Append inverse block so ideal execution returns to the
                    # reference state.
                    cliffGates += [ProductGate([copy.deepcopy(gate)
                                                ** -1 for gate in cliffGates])]
                # Optionally add noise after each gate.
                if self.noiseChannel is not None:
                    cliffGates = self._add_noise(cliffGates)
                # Append the measurement indicator.
                cliffGates += ["M"]
                # Preserve per-circuit depth label so downstream code never
                # relies on hardcoded depth step assumptions.
                self.rbSeqs.append(cliffGates)
                self.depthKeys.append(str(length))

    def _add_noise(
            self,
            gates: List[Union[cirq.Gate, str]]) -> List[Union[cirq.Gate, str]]:
        """
Inserts the noise channel after each gate (except after measurement
markers).

Args:
    gates (List[Union[cirq.Gate, str]]): Collection of gates supplied to the
                                         randomized-benchmarking sequence
                                         builder.

Returns:
    List[Union[cirq.Gate, str]]: Output produced by this routine according to
                                 the behavior described above (inserts the
                                 noise channel after each gate (except after
                                 measurement).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        noisyGates = []
        for gate in gates:
            # Keep original gate/marker ordering.
            noisyGates.append(gate)
            if gate != "M":
                # Insert one noise channel immediately after each gate token.
                noisyGates.append(self.noiseChannel)
        return noisyGates

    def getrbSeqs(self) -> List[List[Union[cirq.Gate, str]]]:
        """
        Returns the generated randomized benchmarking sequences.
        
        Args:
        None: `getrbSeqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[Union[cirq.Gate, str]]]: Requested data object loaded or
                                               assembled by this method.
        
        Raises:
        ValueError: If `getrbSeqs` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.rbSeqs

    def genMoms(self):
        """
        Converts each randomized benchmarking sequence into a list of Cirq
        Moments. Each gate in the sequence is applied to all qudits in a
        separate Moment. The "M" marker is replaced by a measurement moment
        (optionally after a basis transformation).
        
        Args:
        None: `genMoms` relies on object state and accepts no additional
        inputs.
        
        Returns:
        None: `genMoms` executes for side effects only and returns no payload.
        
        Raises:
        ValueError: If `genMoms` receives inputs that are incompatible with its
        expected configuration.
        """
        self.rbMoments = []
        for seq in self.rbSeqs:
            moments = []
            for gate in seq:
                if gate == "M":
                    # Terminal measurement marker.
                    if self.measurementBasisTransformation is None:
                        moments.append(cirq.Moment(
                            [cirq.measure(q) for q in self.qudits]))
                    else:
                        # Rotate measurement basis before measuring.
                        moments.append(
                            cirq.Moment(
                                [
                                    self.measurementBasisTransformation.on(q)
                                    for q in self.qudits
                                ]
                            )
                        )
                        moments.append(cirq.Moment(
                            [cirq.measure(q) for q in self.qudits]))
                else:
                    # Broadcast gate across all target qudits for this model.
                    moments.append(cirq.Moment(
                        [gate.on(q) for q in self.qudits]))
            self.rbMoments.append(moments)

    def getrbMoments(self) -> List[List[cirq.Moment]]:
        """
        Returns the generated Cirq Moments for the RB sequences.
        
        Args:
        None: `getrbMoments` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[cirq.Moment]]:
                Requested data object loaded or assembled by this
                                     method.
        
        Raises:
        ValueError: If `getrbMoments` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.rbMoments

    def get_rb_moments(self) -> List[List[cirq.Moment]]:
        """
        Snake-case alias for `getrbMoments`.
        
        Args:
        None: `get_rb_moments` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[cirq.Moment]]:
                Requested data object loaded or assembled by this
                                     method.
        
        Raises:
        ValueError: If `get_rb_moments` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.getrbMoments()

    def genCircs(self):
        """
        Builds Cirq Circuits from the list of randomized benchmarking moments.
        
        Args:
        None: `genCircs` relies on object state and accepts no additional
        inputs.
        
        Returns:
        None: `genCircs` executes for side effects only and returns no payload.
        
        Raises:
        ValueError: If `genCircs` receives inputs that are incompatible with
        its expected configuration.
        """
        # Materialize executable Cirq circuits from the moment representation.
        self.rbCircuits = [cirq.Circuit(momSeq) for momSeq in self.rbMoments]

    def getrbCircuits(self) -> List[cirq.Circuit]:
        """
        Returns the Cirq Circuits generated from the RB sequences.
        
        Args:
        None: `getrbCircuits` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[cirq.Circuit]:
                Requested data object loaded or assembled by this
                                method.
        
        Raises:
        ValueError: If `getrbCircuits` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.rbCircuits

    def getDepthKeys(self) -> List[str]:
        """
        Return RB depth labels aligned with generated circuit indices.
        
        Args:
        None: `getDepthKeys` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[str]: Depth labels for each generated circuit.
        
        Raises:
        ValueError: If `getDepthKeys` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.depthKeys

    def get_depth_keys(self) -> List[str]:
        """
        Snake-case alias for `getDepthKeys`.
        
        Args:
        None: `get_depth_keys` relies on object state and accepts no
        additional inputs.
        
        Returns:
            List[str]: Depth labels for each generated circuit.
        
        Raises:
        ValueError: If `get_depth_keys` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.getDepthKeys()

    def get_rb_circuits(self) -> List[cirq.Circuit]:
        """
        Snake-case alias for `getrbCircuits`.
        
        Args:
        None: `get_rb_circuits` relies on object state and accepts no
        additional inputs.
        
        Returns:
            List[cirq.Circuit]:
                Requested data object loaded or assembled by this
                                method.
        
        Raises:
        ValueError: If `get_rb_circuits` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.getrbCircuits()
