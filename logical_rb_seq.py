"""
logical_rb_seq.py

Logical Randomized Benchmarking Sequence Class

This class generates randomized benchmarking sequences using logical
Clifford operators.

Attributes:
    measureQudits (List[cirq.Qid]): List of qudits to be measured.
    logicalGateSet (Dict[str, cirq.Circuit]): Logical Clifford circuits.
    sequenceLengths (List[int]): List of sequence lengths.
    circuitsPerSequence (int): Number of circuits to generate per sequence
                               length.
    rng (random.Random): Random number generator (seeded for reproducibility).
    noiseChannel (cirq.Gate, optional): Noise channel to insert into sequences.
    measurementInterval (float, optional): Fraction (0-1) defining intervals at
                                           which to insert measurements.
    syndromeCircuits (cirq.Circuit, optional): Circuit for syndrome extraction.

    rbSeqs (List[List[Union[cirq.Circuit, str]]]): RB sequences as a list of
                                                   gate circuits and
                                                   measurement markers.
    stringSeqs (List[List[str]]): The same RB sequences represented as strings.
    rbMoments (List[List[cirq.Moment]]): RB sequences converted into lists of
                                         Cirq Moments.
    rbCircuits (List[cirq.Circuit]): Final Cirq circuits for the RB experiment.

Methods:
    genSeqs(): Generate randomized benchmarking sequences.
    _invertCircuit(circuit): Invert a given circuit (reverse operations and
                             invert each).
    _invertStrings(stringList): Invert a list of gate string labels.
    _addNoiseToSequence(gateSequence, numNonInverse): Insert noise after
                                                      selected gates.
    _addNoise(logicalGate): Append noise (physical or logical) to a gate
                            circuit.
    _addMeasurementsWithInterval(gateSequence, numNonInverse): Insert
                                                               measurement
                                                               placeholders
                                                               ("M") at regular
                                                               intervals.
    _updateMeasurementKeys(circuit, suffix): Update measurement keys for
                                             uniqueness.
    genMoms(): Convert RB sequences into lists of Cirq Moments.
    genCircs(): Convert the moments into Cirq Circuits.
    getRbSeqs(): Return the RB sequences (as a list of circuit/marker lists).
    getRbMoments(): Return the list of Cirq Moments.
    getRbCircuits(): Return the final Cirq Circuits.
"""

import cirq
import random
import copy
from typing import List, Dict, Union
from gates import (
    ProductGate,  # Assumes ProductGate is defined in your gates module.
)


class LogicalRBSeq:
    """
    Generator for logical randomized benchmarking sequences and circuits.

    This class builds depth-parameterized logical RB experiments from a
    supplied
    logical gate dictionary. It supports optional noise insertion and optional
    interval syndrome/measurement-marker insertion before producing final Cirq
    circuits.

    Attributes:
        measureQudits (List[cirq.Qid]): Qudits measured at terminal (and
            optional interval) stages.
        measurementBasisTransformation (cirq.Circuit): Optional basis-rotation
            circuit applied before terminal measurement.
        logicalGateSet (Dict[str, cirq.Circuit]): Mapping of logical labels to
            logical circuits.
        sequenceLengths (List[int]): Non-negative RB sequence lengths.
        circuitsPerSequence (int): Number of random sequences per depth.
        rng (random.Random): Seeded RNG used for reproducible sequence draws.
        noiseChannel (cirq.Gate): Optional channel inserted after selected
            sequence elements.
        measurementInterval (float): Optional fractional interval for inserting
            `"M"` markers in non-terminal positions.
        syndromeCircuits (cirq.Circuit): Optional syndrome extraction circuit
            inserted at interval markers.
        rbSeqs (List[List[Union[cirq.Circuit, str]]]): Sequence representation
            containing logical gate circuits and measurement markers.
        stringSeqs (List[List[str]]): Human-readable sequence labels.
        rbMoments (List[List[cirq.Moment]]): Sequence representation converted
                                             to
            Cirq moments.
        rbCircuits (List[cirq.Circuit]): Final executable RB circuits.

    Methods:
        __init__(...): Configure the generator and immediately build sequences,
            moments, and circuits.
        genSeqs(): Generate randomized logical sequences and inverse structure.
        genMoms(): Convert generated sequence objects into moment lists.
        genCircs(): Convert moment lists into Cirq circuits.
        getRbSeqs()/get_rb_seqs(): Return sequence-level representation.
        getRbMoments()/get_rb_moments(): Return moment-level representation.
        getRbCircuits()/get_rb_circuits(): Return circuit-level representation.
    """

    def __init__(self, measureQudits: List[cirq.Qid],
                 logicalGateSet: Dict[str, cirq.Circuit],
                 sequenceLengths: List[int] = [
                     0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                 circuitsPerSequence: int = 30,
                 measurementBasisTransformation: cirq.Circuit = None,
                 seed: int = None, noiseChannel: cirq.Gate = None,
                 measurementInterval: float = None,
                 syndromeCircuits: cirq.Circuit = None):
        """
        Initializes the LogicalRBSeq generator.
        
        Args:
            measureQudits (List[cirq.Qid]): Qudits measured in the generated RB
                                            experiment.
            logicalGateSet (Dict[str, cirq.Circuit]):
                Dictionary mapping logical labels
                                                      to circuits.
            sequenceLengths (List[int]): RB sequence lengths to generate.
            circuitsPerSequence (int):
                Number of random circuits per sequence length.
            measurementBasisTransformation (cirq.Circuit):
                Optional basis-rotation
                                                           circuit applied
                                                           before
                                                           terminal
                                                           measurement.
            seed (int): Random seed used for reproducible sequence sampling.
            noiseChannel (cirq.Gate):
                Noise channel inserted according to placement
                                      rules.
            measurementInterval (float):
                Fraction controlling interval-measurement
                                         placement.
            syndromeCircuits (cirq.Circuit):
                Syndrome circuit inserted at interval
                                             checkpoints.
        
        Returns:
            None: None. This method updates internal state in-place.
        
        Raises:
        ValueError: If `__init__` receives inputs that are incompatible with
        its expected configuration.
        """
        self.measureQudits = measureQudits
        self.measurementBasisTransformation = measurementBasisTransformation
        self.logicalGateSet = logicalGateSet
        self.sequenceLengths = sequenceLengths
        self.circuitsPerSequence = circuitsPerSequence
        self.rng = random.Random(seed)
        self.noiseChannel = noiseChannel
        self.measurementInterval = measurementInterval
        self.syndromeCircuits = syndromeCircuits

        self.rbSeqs: List[List[Union[cirq.Circuit, str]]] = []
        self.stringSeqs: List[List[str]] = []
        self.depthKeys: List[str] = []
        self.rbMoments: List[List[cirq.Moment]] = []
        self.rbCircuits: List[cirq.Circuit] = []

        self.genSeqs()
        self.genMoms()
        self.genCircs()

    @property
    def string_seqs(self) -> List[List[str]]:
        """
        Snake-case alias for `stringSeqs`.
        
        Args:
        None: `string_seqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
        List[List[str]]: Return value produced by `string_seqs` for the
        requested operation.
        
        Raises:
        ValueError: If `string_seqs` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.stringSeqs

    @property
    def depth_keys(self) -> List[str]:
        """
        Snake-case alias for `depthKeys`.
        
        Args:
        None: `depth_keys` relies on object state and accepts no additional
        inputs.
        
        Returns:
        List[str]: Depth labels aligned one-to-one with generated RB circuits.
        
        Raises:
        ValueError: If `depth_keys` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.depthKeys

    def genSeqs(self):
        """
        Generate RB sequences with measurement markers. For each sequence
        
        Args:
        None: `genSeqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
            None: None. Populates internal sequence containers.
        
        Raises:
        ValueError: If `genSeqs` receives inputs that are incompatible with its
        expected configuration.
        """
        self.rbSeqs = []
        self.stringSeqs = []
        self.depthKeys = []
        # Cache gate labels once so each sequence draw only samples labels and
        # then resolves them to circuits.
        stringLogicalGateSet = list(self.logicalGateSet.keys())

        # Outer loop controls the target RB depth; inner loop controls how many
        # random circuits are generated at that fixed depth.
        for length in self.sequenceLengths:
            for _ in range(self.circuitsPerSequence):
                if length == 0:
                    # Depth-0 sequence is the identity benchmark point.
                    cliffGates = [self.logicalGateSet['I']]
                    stringCliffGates = ['I']
                    if self.noiseChannel is not None:
                        cliffGates = self._addNoiseToSequence(
                            cliffGates, numNonInverse=1)
                    cliffGates.append("tM")
                    stringCliffGates.append("tM")
                else:
                    # Choose random gates for the non-inverse part.
                    stringCliffGates = [self.rng.choice(
                        stringLogicalGateSet) for _ in range(length)]
                    cliffGates = [self.logicalGateSet[s]
                                  for s in stringCliffGates]

                    # Build the sequence inverse so ideal execution maps the
                    # state back to the reference state before measurement.
                    stringInverseGate = self._invertStrings(stringCliffGates)
                    inverseGates = [self._invertCircuit(
                        gate) for gate in reversed(cliffGates)]
                    cliffGates += inverseGates
                    stringCliffGates.append(stringInverseGate)

                    if self.noiseChannel is not None:
                        # Inject noise after each non-inverse gate and after
                        # the final inverse gate.
                        cliffGates = self._addNoiseToSequence(
                            cliffGates,
                            numNonInverse=len(stringCliffGates) - 1,
                        )
                    if self.measurementInterval is not None:
                        # Insert interval measurement markers used later when
                        # converting sequence tokens into moments.
                        cliffGates = self._addMeasurementsWithInterval(
                            cliffGates, len(stringCliffGates) - 1)
                    cliffGates.append("tM")
                    stringCliffGates.append("tM")

                # Keep circuit objects, human-readable labels, and depth keys
                # index-aligned so downstream code can recover per-circuit
                # depth without assumptions about step size.
                self.rbSeqs.append(cliffGates)
                self.stringSeqs.append(stringCliffGates)
                self.depthKeys.append(str(length))

    def _invertCircuit(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """
        Invert a given circuit by reversing the order and inverting each
        
        Args:
            circuit (cirq.Circuit): Circuit to invert or transform.
        
        Returns:
            cirq.Circuit: Inverse circuit built from reversed operations.
        
        Raises:
        ValueError: If `_invertCircuit` receives inputs that are incompatible
        with its expected configuration.
        """
        invertedOps = [copy.deepcopy(
            op)**-1 for op in reversed(list(circuit.all_operations()))]
        return cirq.Circuit(invertedOps)

    def _invertStrings(self, stringList: List[str]) -> str:
        """
        Invert a list of gate strings.
        
        Args:
            stringList (List[str]): List of symbolic gate labels.
        
        Returns:
            str: Serialized inverse-label expression.
        
        Raises:
        ValueError: If `_invertStrings` receives inputs that are incompatible
        with its expected configuration.
        """
        return ''.join([f"({gate})^-1" for gate in stringList])

    def _addNoiseToSequence(self,
                            gateSequence: List[Union[cirq.Circuit,
                                                     str]],
                            numNonInverse: int) -> List[Union[cirq.Circuit,
                                                              str]]:
        """
        Add noise to each non-inverse gate (and the final inverse) in the
        
        Args:
            gateSequence (List[Union[cirq.Circuit, str]]): Ordered gate/circuit
                                                           sequence to process.
            numNonInverse (int):
                Count of non-inverse gates in the sequence prefix.
        
        Returns:
            List[Union[cirq.Circuit, str]]:
                Sequence with noise inserted at configured
                                            locations.
        
        Raises:
        ValueError: If `_addNoiseToSequence` receives inputs that are
        incompatible with its expected configuration.
        """
        noisySequence = []
        for i, gate in enumerate(gateSequence):
            if isinstance(gate, str):
                noisySequence.append(gate)
            else:
                if self.noiseChannel is not None and (
                        i < numNonInverse or i == len(gateSequence) - 1):
                    noisyGate = self._addNoise(gate)
                    noisySequence.append(noisyGate)
                else:
                    noisySequence.append(gate)
        return noisySequence

    def _addNoise(self, logicalGate: cirq.Circuit) -> cirq.Circuit:
        """
        Add noise to a logical gate. If the noise channel acts on one qudit,
        
        Args:
            logicalGate (cirq.Circuit): Logical gate circuit to which noise is
                                        appended.
        
        Returns:
            cirq.Circuit: Noisy logical-gate circuit.
        
        Raises:
        ValueError: If `_addNoise` receives inputs that are incompatible with
        its expected configuration.
        """
        noisyCircuit = cirq.Circuit(logicalGate)
        if self.noiseChannel._num_qubits_() == 1:
            occupiedQudits = {
                q
                for moment in logicalGate
                for op in moment.operations
                for q in op.qubits
            }
            for q in occupiedQudits:
                noisyCircuit.append(self.noiseChannel.on(q))
        else:
            noisyCircuit.append(self.noiseChannel.on(*self.measureQudits))
        return noisyCircuit

    def _addMeasurementsWithInterval(
            self,
            gateSequence: List[Union[cirq.Circuit, str]],
            numNonInverse: int) -> List[Union[cirq.Circuit, str]]:
        """
        Insert measurement placeholders ("M") at regular intervals.
        
        Args:
            gateSequence (List[Union[cirq.Circuit, str]]): Ordered gate/circuit
                                                           sequence to process.
            numNonInverse (int):
                Count of non-inverse gates in the sequence prefix.
        
        Returns:
            List[Union[cirq.Circuit, str]]:
                Sequence with interval measurement markers
                                            inserted.
        
        Raises:
        ValueError: If `_addMeasurementsWithInterval` receives inputs that are
        incompatible with its expected configuration.
        """
        measureSequence = []
        measurementPeriod = max(1, int(1 / self.measurementInterval))
        for i, gate in enumerate(gateSequence):
            measureSequence.append(gate)
            if i < numNonInverse and (i + 1) % measurementPeriod == 0:
                measureSequence.append("M")
        return measureSequence

    def _updateMeasurementKeys(
            self,
            circuit: cirq.Circuit,
            suffix: str) -> cirq.Circuit:
        """
        Update measurement keys to ensure uniqueness.
        
        Args:
            circuit (cirq.Circuit): Circuit to invert or transform.
            suffix (str): Suffix appended to measurement keys for uniqueness.
        
        Returns:
            cirq.Circuit:
                Circuit with measurement keys rewritten to unique names.
        
        Raises:
        ValueError: If `_updateMeasurementKeys` receives inputs that are
        incompatible with its expected configuration.
        """
        updatedCircuit = cirq.Circuit()
        for moment in circuit:
            updatedOps = []
            for op in moment.operations:
                if isinstance(op.gate, cirq.MeasurementGate):
                    newKey = f"{op.gate.key}_{suffix}"
                    updatedOps.append(cirq.measure(*op.qubits, key=newKey))
                else:
                    updatedOps.append(op)
            updatedCircuit.append(cirq.Moment(updatedOps))
        return updatedCircuit

    def genMoms(self):
        """
        Convert each RB sequence into a list of Cirq Moments. For "M" markers,
        
        Args:
        None: `genMoms` relies on object state and accepts no additional
        inputs.
        
        Returns:
            None: None. Builds moment-level representations in-place.
        
        Raises:
        ValueError: If `genMoms` receives inputs that are incompatible with its
        expected configuration.
        """
        self.rbMoments = []
        for seq in self.rbSeqs:
            moments = []
            measurementCounter = 0
            for gate in seq:
                if gate == "M":
                    # "M" is an interval marker; emit syndrome circuits only
                    # if the caller configured one.
                    if self.syndromeCircuits is not None:
                        uniqueSyndrome = self._updateMeasurementKeys(
                            self.syndromeCircuits,
                            f"interval_{measurementCounter}",
                        )
                        moments.extend(uniqueSyndrome)
                        measurementCounter += 1
                elif gate == "tM":
                    # "tM" is the terminal measurement stage. Optionally apply
                    # basis transformation first, then measure all qudits.
                    if self.measurementBasisTransformation is None:
                        moments.append(
                            cirq.Moment(
                                [
                                    cirq.measure(q, key=f"Terminal_{q}")
                                    for q in self.measureQudits
                                ]
                            )
                        )
                    else:
                        for m in self.measurementBasisTransformation.moments:
                            moments.append(m)
                        moments.append(
                            cirq.Moment(
                                [
                                    cirq.measure(q, key=f"Terminal_{q}")
                                    for q in self.measureQudits
                                ]
                            )
                        )
                elif isinstance(gate, cirq.Circuit):
                    # Regular logical-gate block; append all moments in-order.
                    moments.extend(gate)
            self.rbMoments.append(moments)

    def genCircs(self):
        """
        Generate Cirq circuits from the RB moments.
        
        Args:
        None: `genCircs` relies on object state and accepts no additional
        inputs.
        
        Returns:
            None: None. Builds circuit-level representations in-place.
        
        Raises:
        ValueError: If `genCircs` receives inputs that are incompatible with
        its expected configuration.
        """
        # One circuit per sequence, preserving index alignment with depth keys.
        self.rbCircuits = [cirq.Circuit(momSeq) for momSeq in self.rbMoments]

    def getRbSeqs(self) -> List[List[Union[cirq.Circuit, str]]]:
        """
        Return the RB sequences.
        
        Args:
        None: `getRbSeqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[Union[cirq.Circuit, str]]]: Generated RB sequences.
        
        Raises:
        ValueError: If `getRbSeqs` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.rbSeqs

    def get_rb_seqs(self) -> List[List[Union[cirq.Circuit, str]]]:
        """
        Snake-case alias for `getRbSeqs`.
        
        Args:
        None: `get_rb_seqs` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[Union[cirq.Circuit, str]]]: Generated RB sequences.
        
        Raises:
        ValueError: If `get_rb_seqs` receives inputs that are incompatible with
        its expected configuration.
        """
        return self.getRbSeqs()

    def getRbMoments(self) -> List[List[cirq.Moment]]:
        """
        Return the RB sequences as lists of Cirq Moments.
        
        Args:
        None: `getRbMoments` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[cirq.Moment]]: Generated RB moments.
        
        Raises:
        ValueError: If `getRbMoments` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.rbMoments

    def get_rb_moments(self) -> List[List[cirq.Moment]]:
        """
        Snake-case alias for `getRbMoments`.
        
        Args:
        None: `get_rb_moments` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[List[cirq.Moment]]: Generated RB moments.
        
        Raises:
        ValueError: If `get_rb_moments` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.getRbMoments()

    def getRbCircuits(self) -> List[cirq.Circuit]:
        """
        Return the final RB circuits.
        
        Args:
        None: `getRbCircuits` relies on object state and accepts no additional
        inputs.
        
        Returns:
            List[cirq.Circuit]: Generated RB circuits.
        
        Raises:
        ValueError: If `getRbCircuits` receives inputs that are incompatible
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
        Snake-case alias for `getRbCircuits`.
        
        Args:
        None: `get_rb_circuits` relies on object state and accepts no
        additional inputs.
        
        Returns:
            List[cirq.Circuit]: Generated RB circuits.
        
        Raises:
        ValueError: If `get_rb_circuits` receives inputs that are incompatible
        with its expected configuration.
        """
        return self.getRbCircuits()
