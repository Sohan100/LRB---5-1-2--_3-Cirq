"""
single_qutrit_rb.py

This module defines the SingleQutritRBExperiment class that conducts randomized
benchmarking (RB)
on a physical single qutrit. In this "Plus State RB" experiment, a
noisy RB sequence is run
on a qutrit initially prepared in a chosen state (here using a preparation
gate) and the measurement
results are used to compute an exponentially decaying expectation value from
which the average gate
fidelity is estimated.

Experiment Protocol:
  0) (Optional) Add depolarizing noise to the qutrit.
  1) Prepare the qutrit (e.g. in a |+> state) and run RB sequences.
  2) For each of 11 sequence lengths ([0,2,4,...,20]), generate 30 randomized
  sequences,
     each run for a specified number of repetitions (shots).
  3) Plot the (average) expectation value vs. circuit depth (number of Clifford
  gates).
  4) Fit the data to an exponentially decaying function p(m)= C*f^m.
  5) Compute the average gate fidelity using F = [(d-1)f + 1] / d.

Attributes:
    qudits: List of qudits (cirq.Qid) used in the experiment.
    cliffordGates: List of available Clifford gates for building the RB
                   sequences.
    sequenceLengths: List of desired sequence lengths.
    circuitsPerSequence: Number of circuits per sequence length.
    shots: Number of repetitions for each circuit.
    measurementBasisTransformation: (Optional) Gate to transform measurement
                                    basis.
    noiseChannel: (Optional) Noise channel to add after each gate.
    seed: Random seed for reproducibility.

Methods:
    run() -> Dict[int, List[float]]:
         Simulate all RB circuits and compute expectation values for each
         sequence length.
    fit_decay() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
         Fit the average expectation values to an exponential decay model.
    compute_gate_fidelity() -> float:
         Compute the average gate fidelity from the decay parameter.
    plot_results():
         Plot the measured expectation values and the fitted decay curve.
"""

from typing import List, Dict, Union, Tuple
import cirq
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import copy

# Import our RB sequence generator (assumed to be defined in
# single_qutrit_rb.py)
# (If RBSeq is in another file, adjust the import accordingly.)
# <-- make sure that RBSeq is available from an appropriate module
from rb_seq import RBSeq
# Also import necessary gates and noise channel
from gates import QuditHadamard
from noise import QuditDepolarizingChannel


class SingleQutritRBExperiment:
    """
    End-to-end physical single-qutrit randomized benchmarking experiment class.

    The class wraps RB sequence generation, simulation, exponential-decay
    fitting, and fidelity estimation for a configurable qudit and gate set.

    Attributes:
        qudits (List[cirq.Qid]): Target qudits for RB circuits.
        cliffordGates (List[cirq.Gate]): Gate pool used to generate randomized
            RB sequences.
        sequenceLengths (List[int]): RB depth values.
        circuitsPerSequence (int): Number of random circuits generated per
            depth.
        shots (int): Number of repetitions (samples) per circuit.
        measurementBasisTransformation (cirq.Gate): Optional basis transform
            applied before measurement.
        noiseChannel (cirq.Gate): Optional channel appended by the sequence
            generator.
        seed (int): RNG seed for reproducibility.
        rb_seq (RBSeq): Backing sequence generator instance.
        rb_circuits (List[cirq.Circuit]): Generated circuits to execute.
        expectationValues (Dict[int, List[float]]): Depth-keyed expectation
            values accumulated after simulation.

    Methods:
        __init__(...): Configure experiment and pre-generate RB circuits.
        run(): Execute circuits and populate expectation values.
        fit_decay(): Fit depth-averaged expectations to exponential decay.
        compute_gate_fidelity(): Convert fitted decay parameter to fidelity.
        plot_results(): Visualize raw distributions and fitted trend.
    """

    def __init__(self,
                 qudits: List[cirq.Qid],
                 cliffordGates: List[cirq.Gate],
                 sequenceLengths: List[int] = [
                     0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                 circuitsPerSequence: int = 30,
                 shots: int = 1000,
                 measurementBasisTransformation: cirq.Gate = None,
                 noiseChannel: cirq.Gate = None,
                 seed: int = None):
        """
        Initializes the randomized benchmarking experiment.
        
        Args:
            qudits (List[cirq.Qid]):
                Ordered qudit register used by this experiment or
                                     circuit constructor.
            cliffordGates (List[cirq.Gate]):
                Input argument consumed by `__init__` to
                                             perform this operation.
            sequenceLengths (List[int]):
                List of randomized-benchmarking sequence
                                         lengths to generate.
            circuitsPerSequence (int):
                Number of random circuits generated for each
                                       sequence length.
            shots (int): Input argument consumed by `__init__` to perform this
                         operation.
            measurementBasisTransformation (cirq.Gate):
                Optional circuit applied before
                                                        terminal measurement to
                                                        rotate
                                                        into the desired
                                                        readout basis.
            noiseChannel (cirq.Gate):
                Noise-channel gate instance inserted according to
                                      this method's placement strategy.
            seed (int):
                Random-number-generator seed used to make circuit sampling
                        reproducible.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.qudits = qudits
        self.cliffordGates = cliffordGates
        self.sequenceLengths = sequenceLengths
        self.circuitsPerSequence = circuitsPerSequence
        self.shots = shots
        self.measurementBasisTransformation = measurementBasisTransformation
        self.noiseChannel = noiseChannel
        self.seed = seed

        # Generate RB circuits using the RBSeq class.
        self.rb_seq = RBSeq(
            qudits=self.qudits,
            gates=self.cliffordGates,
            sequenceLengths=self.sequenceLengths,
            circuitsPerSequence=self.circuitsPerSequence,
            measurementBasisTransformation=self.measurementBasisTransformation,
            seed=self.seed,
            noiseChannel=self.noiseChannel)
        self.rb_circuits = self.rb_seq.getrbCircuits()
        # Store per-circuit depth labels so analysis can avoid hardcoded depth
        # step assumptions when available.
        self.depth_keys = (
            self.rb_seq.get_depth_keys()
            if hasattr(self.rb_seq, "get_depth_keys")
            else []
        )

        # Dictionary mapping sequence length (key) to list of expectation
        # values.
        self.expectationValues: Dict[int, List[float]] = {}

    def run(self) -> Dict[int, List[float]]:
        """
        Runs the RB circuits using a DensityMatrixSimulator and computes the
        expectation value for each circuit. The expectation value is calculated
        as: exp_val = | P(0) + P(1)*w + P(2)*w^2 |, where
        w = exp(2*pi*i/3).
        Args:
        None: `run` relies on object state and accepts no additional inputs.
        
        Returns:
            Dict[int, List[float]]:
                Simulation result generated for the requested
                                    configuration.
        
        Raises:
        ValueError: If `run` receives inputs that are incompatible with its
        expected configuration.
        """
        simulator = cirq.DensityMatrixSimulator()
        self.expectationValues = {}
        w = np.exp(2j * np.pi / 3)

        for i, circuit in enumerate(self.rb_circuits):
            # Determine the sequence length key.
            if self.depth_keys and len(self.depth_keys) == (
                    len(self.rb_circuits)):
                key = int(self.depth_keys[i])
            else:
                key = 2 * (i // self.circuitsPerSequence)
            # Build the full circuit by (optionally) preparing the plus state.
            noisy_circuit = cirq.Circuit()
            # For example, if measurementBasisTransformation is given, we
            # assume a preparation gate is applied.
            if self.measurementBasisTransformation is not None:
                prep_gate = QuditHadamard(3)  # Prepare in the plus state.
                noisy_circuit += cirq.Circuit(prep_gate.on(self.qudits[0]))
            noisy_circuit += circuit

            # Run the simulation.
            result = simulator.run(noisy_circuit, repetitions=self.shots)
            # Assume that the measurement key is the only key in
            # result.measurements.
            meas_key = list(result.measurements.keys())[0]
            outcomes = result.measurements[meas_key]
            flat_outcomes = [int(item)
                             for sublist in outcomes for item in sublist]
            N = len(flat_outcomes)
            p0 = flat_outcomes.count(0) / N
            p1 = flat_outcomes.count(1) / N
            p2 = flat_outcomes.count(2) / N
            # Qutrit Bloch-like expectation magnitude used for RB decay fits.
            exp_val = np.abs(p0 * 1 + p1 * w + p2 * (w**2))
            if key not in self.expectationValues:
                self.expectationValues[key] = [exp_val]
            else:
                self.expectationValues[key].append(exp_val)

        return self.expectationValues

    def fit_decay(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the average expectation values to an exponential decay model: p(m)
        = A * f^m.
        
        Args:
        None: `fit_decay` relies on object state and accepts no additional
        inputs.
        
        Returns:
            Any:
                Fitted parameters and/or derived metrics for the provided data.
        
        Raises:
            RuntimeError: If fitting fails to converge.
            ValueError: If input data is invalid for fitting.
        """
        keys = sorted(self.expectationValues.keys())
        xData = np.array(keys)
        yData = np.array([np.mean(self.expectationValues[k]) for k in keys])

        def fitFunc(x, A, f):
            """
            Local exponential RB fit model used for `curve_fit`.

            Args:
                x: RB depth values.
                A: Fit amplitude.
                f: Fit decay parameter.

            Returns:
                Modeled expectation values.

            Raises:
                Exception: Propagates exceptions raised by underlying
                           operations,
                    dependencies, or invalid runtime states.
            """
            return A * (f ** x)

        popt, _ = curve_fit(fitFunc, xData, yData)
        # Cache fit parameters for downstream fidelity and plotting methods.
        self.fitting_params = popt
        return popt, xData, yData

    def compute_gate_fidelity(self) -> float:
        """
        Computes the average gate fidelity using the decay parameter f. For a
        d-dimensional qudit, F = [(d - 1) * f + 1] / d.
        
        Args:
        None: `compute_gate_fidelity` relies on object state and accepts no
        additional inputs.
        
        Returns:
            float: Result object produced by this method.
        
        Raises:
            ValueError:
                If validation fails: Fitting parameters not available; run
                        fit_decay() first..
        """
        d = 3  # For a qutrit.
        if not hasattr(self, 'fitting_params'):
            raise ValueError(
                "Fitting parameters not available; run fit_decay() first.")
        f = self.fitting_params[1]
        F = ((d - 1) * f + 1) / d
        self.average_gate_fidelity = F
        return F

    def plot_results(self):
        """
        Plots the experimental expectation values (using violin plots with
        overlaid scatter points) and the fitted exponential decay curve.
        
        Args:
        None: `plot_results` relies on object state and accepts no additional
        inputs.
        
        Returns:
            object: Result object produced by this method.
        
        Raises:
            OSError: If plot outputs cannot be written to disk.
            ValueError: If plotting inputs have incompatible structure.
        """
        keys = sorted(self.expectationValues.keys())
        xData = np.array(keys)
        yData = np.array([np.mean(self.expectationValues[k]) for k in keys])
        violinData = [self.expectationValues[k] for k in keys]

        plt.figure(figsize=(10, 10))
        plt.violinplot(violinData, xData, showmedians=True)
        for i, data in enumerate(violinData):
            # Overlay raw shot-derived points to show spread at each depth.
            x_vals = [xData[i]] * len(data)
            plt.scatter(x_vals, data, color='blue', alpha=0.4, s=15)
        if hasattr(self, 'fitting_params'):
            def fitFunc(x, A, f):
                """
                Local plotting model matching the RB exponential fit form.

                Args:
                    x: RB depth values.
                    A: Fit amplitude.
                    f: Fit decay parameter.

                Returns:
                    Modeled expectation values.

                Raises:
                    Exception: Propagates exceptions raised by underlying
                               operations,
                        dependencies, or invalid runtime states.
                """
                return A * (f ** x)
            plt.plot(
                xData,
                fitFunc(
                    xData,
                    *
                    self.fitting_params),
                'b-',
                label=(
                    f'C = {self.fitting_params[0]:.4f}\n'
                    f'f = {self.fitting_params[1]:.4f}'
                ))
        plt.xlabel('Circuit Depth (Clifford Gates)')
        plt.ylabel('Expectation Value')
        plt.legend()
        plt.xticks(range(int(xData.min()), int(xData.max()) + 1))
        plt.xlim(-0.1, max(xData) + 1)
        plt.show()


# Example usage:
if __name__ == '__main__':
    # Define a physical qutrit (here using a LineQid with index 5 and
    # dimension 3).
    qutrit = [cirq.LineQid(5, dimension=3)]
    # Import example gates from your gates module.
    from gates import QuditX, QuditZ, QuditS, QuditHadamard
    # For this example, we assume a simple set of Clifford gates.
    cliffordGates = [QuditX(3), QuditZ(3), QuditHadamard(3), QuditS(3)]
    # Create an instance of the experiment with depolarizing noise.
    experiment = SingleQutritRBExperiment(
        qudits=qutrit,
        cliffordGates=cliffordGates,
        sequenceLengths=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        circuitsPerSequence=30,
        shots=5000,
        measurementBasisTransformation=(QuditHadamard(3)**-1),
        noiseChannel=QuditDepolarizingChannel(3, 0.1),
        seed=24
    )
    exp_vals = experiment.run()
    print("Expectation Values:", exp_vals)
    popt, xData, yData = experiment.fit_decay()
    print("Fitting parameters (A, f):", popt)
    avg_gate_fidelity = experiment.compute_gate_fidelity()
    print("Average Gate Fidelity:", avg_gate_fidelity)
    experiment.plot_results()
