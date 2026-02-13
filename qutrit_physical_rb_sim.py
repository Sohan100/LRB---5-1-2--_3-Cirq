#!/usr/bin/env python3
"""
qutrit_physical_rb_sim.py
=========================

Physical (single-qutrit) randomized benchmarking simulation module.

Purpose
-------
This module generates and simulates physical RB datasets used as a baseline for
logical-vs-physical comparisons. It intentionally focuses on simulation and
data persistence only; all visualization is handled by `qutrit_rb_plotting.py`.

Protocol summary
----------------
For each depolarizing probability `p`:
1. Build RB circuits over the unencoded qutrit Clifford set.
2. Insert local physical depolarizing noise via `RBSeq`.
3. Execute circuits with a density-matrix simulator.
4. Compute expectation-value magnitude from trinary outcome frequencies.
5. Fit RB decay and derive average physical gate infidelity.
6. Save raw curves and infidelity summaries for downstream plotting/overlay.

Artifacts written
-----------------
- `physicalRB_SimulationResults.npy`
- `physical_rb_infidelities.npy`

Compatibility
-------------
The class exports the modern `PhysicalRbSim` name and also provides the legacy
`PhysicalRBSim` alias for existing notebooks/scripts.
"""

import os
import numpy as np
import cirq
from scipy.optimize import curve_fit

from qutrit_clifford import QutritCliffordGroup
from gates import QuditHadamard
from noise import QuditDepolarizingChannel
from rb_seq import RBSeq


class PhysicalRbSim:
    """
    This class runs physical (unencoded) single-qutrit Randomized Benchmarking.

    It provides a simulation pipeline parallel to the logical RB workflows so
    physical and logical performance can be compared on matched depolarizing
    rates. The class handles circuit generation, noisy simulation, expectation
    aggregation, infidelity fitting, and persistence of sweep outputs.

    Attributes:
        checkpoint_dir (str): Output directory for simulation result artifacts.
        seed (int): Random seed forwarded to RB sequence generation.
        circuits_per_sequence (int): Number of circuits generated for each RB
            depth.
        dimension (int): Local qudit dimension (3 for qutrit).
        repetitions (int): Number of shots per physical RB circuit.
        clifford_gates (list): Unencoded qutrit Clifford gate collection used
                               by
            `RBSeq`.
        hadamard (cirq.Gate): Measurement-basis transform/preparation gate used
            by the physical RB protocol.

    Methods:
        __init__(checkpoint_dir, seed=24, circuits_per_sequence=30,
                 dimension=3, repetitions=5000):
            Stores configuration and initializes gate resources.
        generate_rb_circuits(physical_error_rate):
            Builds physical RB circuits for one depolarizing probability.
        run_rb_simulation(physical_error_rate):
            Runs all circuits for one rate and returns raw/averaged expectation
            values keyed by RB depth.
        rb_fit_function(x, amplitude, decay):
            Exponential RB decay model used in curve fitting.
        fit_data_and_calculate_infidelity(x_data, y_data):
            Fits decay parameter and converts it to average gate infidelity.
        simulate_for_p_values(p_values):
            Executes a full sweep across multiple depolarizing rates.
        save_physical_rb_infidelities(physical_rb_results, file_path):
            Saves compact physical infidelity summaries.
        run_experiment(p_values):
            End-to-end sweep driver that saves all artifacts for downstream
            plotting/overlay analysis.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        seed: int = 24,
        circuits_per_sequence: int = 30,
        dimension: int = 3,
        repetitions: int = 5000,
    ):
        """
        Initialize physical RB simulation configuration.
        
        Args:
            checkpoint_dir (str):
                Input argument consumed by `__init__` to perform this
                                  operation.
            seed (int):
                Random-number-generator seed used to make circuit sampling
                        reproducible.
            circuits_per_sequence (int):
                Input argument consumed by `__init__` to
                                         perform this operation.
            dimension (int): Local Hilbert-space dimension for the qudit system
                             represented by this operation.
            repetitions (int):
                Number of Monte Carlo shots executed for the circuit
                               simulation.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.seed = seed
        self.circuits_per_sequence = circuits_per_sequence
        self.dimension = dimension
        self.repetitions = repetitions

        self.clifford_gates = QutritCliffordGroup(self.dimension).get_group()
        self.hadamard = QuditHadamard(self.dimension)

    def generate_rb_circuits(self, physical_error_rate: float):
        """
Build physical RB circuits for one depolarizing parameter.

Args:
    physical_error_rate (float): Physical depolarizing probability used for the
                                 current simulation or analysis call.

Returns:
    object: Constructed object generated from the provided inputs.

Raises:
    ValueError: If required experiment parameters are invalid or inconsistent.
"""
        qutrit = [cirq.LineQid(5, self.dimension)]
        noise_channel = QuditDepolarizingChannel(
            self.dimension, physical_error_rate)
        # RBSeq handles randomized gate draws and noise insertion.
        rb_seq = RBSeq(
            qudits=qutrit,
            gates=self.clifford_gates,
            measurementBasisTransformation=self.hadamard,
            seed=self.seed,
            noiseChannel=noise_channel,
        )
        rb_circuits = rb_seq.get_rb_circuits()
        depth_keys = rb_seq.get_depth_keys()
        if len(depth_keys) != len(rb_circuits):
            # Legacy fallback for old sequence objects without explicit depth
            # keys; preserves historical behavior.
            depth_keys = [
                str(2 * (circuit_index // self.circuits_per_sequence))
                for circuit_index in range(len(rb_circuits))
            ]
        return rb_circuits, depth_keys

    def run_rb_simulation(self, physical_error_rate: float):
        """
Run all RB circuits for one depolarizing parameter.

Args:
    physical_error_rate (float): Physical depolarizing probability used for the
                                 current simulation or analysis call.

Returns:
    object: Simulation or analysis result for the requested execution.

Raises:
    ValueError: If required experiment parameters are invalid or inconsistent.
"""
        rb_circuits, depth_keys = self.generate_rb_circuits(
            physical_error_rate)
        simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
        omega = np.exp(2j * np.pi / self.dimension)

        expectation_values = {}

        for circuit_index, circuit in enumerate(rb_circuits):
            # Use explicit per-circuit depth mapping, not hardcoded step size.
            depth_key = str(depth_keys[circuit_index])
            full_circuit = cirq.Circuit(
                self.hadamard.on(cirq.LineQid(5, self.dimension)))
            full_circuit += circuit

            # Run and convert trit outcomes into expectation magnitude.
            result = simulator.run(full_circuit, repetitions=self.repetitions)
            outcomes = [
                int(item)
                for row in result.measurements['q(5) (d=3)']
                for item in row
            ]

            probability_0 = outcomes.count(0) / len(outcomes)
            probability_1 = outcomes.count(1) / len(outcomes)
            probability_2 = outcomes.count(2) / len(outcomes)
            expectation_value = np.abs(
                probability_0
                + probability_1 * omega
                + probability_2 * (omega ** 2)
            )

            expectation_values.setdefault(
                depth_key, []).append(expectation_value)

        # Collapse per-depth sample lists into mean curve for fitting.
        average_expectation = {key: float(np.mean(values))
                               for key, values in expectation_values.items()}
        return expectation_values, average_expectation

    @staticmethod
    def rb_fit_function(x, amplitude, decay):
        """
Exponential RB decay model.

Args:
    x (Any): Independent-variable values (for example RB depth values) used by
             the model function.
    amplitude (Any): Model amplitude parameter used in the fitted decay
                     expression.
    decay (Any): Model decay parameter used in the fitted RB expression.

Returns:
    object: Output produced by this routine according to the behavior described
            above (exponential rb decay model.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return amplitude * (decay ** x)

    def fit_data_and_calculate_infidelity(self, x_data, y_data):
        """
Fit RB decay curve and convert to average gate infidelity.

Args:
    x_data (Any): Independent-variable sample points, typically RB circuit
                  depths.
    y_data (Any): Dependent-variable values associated with `x_data`.

Returns:
    object: Fit parameters and derived metrics computed from input data.

Raises:
    RuntimeError: If model fitting fails to converge with the supplied data.
    ValueError: If input arrays are empty or incompatible for fitting.
"""
        fit_parameters, _ = curve_fit(
            self.rb_fit_function,
            x_data,
            y_data,
            p0=[1.0, 0.99],
            bounds=([0, 0], [2, 1]),
        )
        decay = fit_parameters[1]
        avg_gate_fidelity = (1 + (self.dimension - 1) * decay) / self.dimension
        infidelity = 1 - avg_gate_fidelity
        return fit_parameters, infidelity

    def simulate_for_p_values(self, p_values: list):
        """
Run physical RB across a list of depolarizing rates.

Args:
    p_values (list): Physical depolarizing rates to evaluate in a simulation
                     sweep.

Returns:
    object: Output produced by this routine according to the behavior described
            above (run physical rb across a list of depolarizing rates.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        results = {}
        infidelities = []

        for physical_error_rate in p_values:
            # Each p value is processed independently and recorded in results.
            print(f"Simulating physical RB for p = {physical_error_rate}")
            expectation_values, average_expectation = self.run_rb_simulation(
                physical_error_rate)

            x_data = [int(depth_key)
                      for depth_key in average_expectation.keys()]
            y_data = [average_expectation[depth_key]
                      for depth_key in average_expectation.keys()]
            fit_parameters, infidelity = (
                self.fit_data_and_calculate_infidelity(x_data, y_data)
            )

            results[physical_error_rate] = {
                "x_data": x_data,
                "y_data": y_data,
                "fit_params": fit_parameters,
                "infidelity": infidelity,
                "expectation_values": expectation_values,
                "average": average_expectation,
            }
            infidelities.append(infidelity)

        return results, infidelities

    @staticmethod
    def save_physical_rb_infidelities(physical_rb_results, file_path: str):
        """
Save compact physical infidelity summary.

Args:
    physical_rb_results (Any): Input argument consumed by
                               `save_physical_rb_infidelities` to perform this
                               operation.
    file_path (str): Filesystem path used for loading or saving serialized
                     data.

Returns:
    None: None. This method persists data to storage and does not return a
          payload.

Raises:
    OSError: If the output file cannot be written to disk.
"""
        physical_rates = {p: data["infidelity"]
                          for p, data in physical_rb_results.items()}
        np.save(file_path, physical_rates)
        print(f"Physical RB infidelities saved to {file_path}")

    def run_experiment(self, p_values: list):
        """
        Execute full physical RB sweep and persist artifacts.
        
        Args:
            p_values (list):
                Physical depolarizing rates to evaluate in a simulation
                             sweep.
        
        Returns:
        None: `run_experiment` updates internal object state and returns no
        value.
        
        Raises:
            ValueError:
                If required experiment parameters are invalid or inconsistent.
        """
        results, _ = self.simulate_for_p_values(p_values)

        # Save full physical RB payload used by plotting overlays.
        simulation_file = os.path.join(
            self.checkpoint_dir, "physicalRB_SimulationResults.npy")
        np.save(simulation_file, results)
        print(f"Results saved to '{simulation_file}'")

        # Save compact p -> infidelity mapping for lightweight analysis.
        infidelity_file = os.path.join(
            self.checkpoint_dir, "physical_rb_infidelities.npy")
        self.save_physical_rb_infidelities(results, infidelity_file)


# Backward-compatible alias for existing notebooks/scripts.
PhysicalRBSim = PhysicalRbSim


def main():
    """
    Command-line entrypoint for physical RB simulation sweeps. CLI
    arguments: - `--checkpoint-dir`: output directory. - `--seed`: random
    seed. - `--repetitions`: shots per circuit. - `--error-rate`: optional
    single `p` run.
    
    Args:
    None: `main` relies on object state and accepts no additional inputs.
    
    Returns:
    None: `main` executes for side effects only and returns no payload.
    
    Raises:
    ValueError: If `main` receives inputs that are incompatible with its
    expected configuration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Physical qutrit randomized benchmarking simulation")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoint_physical_rb",
        help="Directory to save physical RB outputs.",
    )
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument("--repetitions", type=int,
                        default=5000, help="Shots per RB circuit.")
    parser.add_argument("--error-rate", type=float,
                        help="Run one p value if specified.")
    args = parser.parse_args()

    sim = PhysicalRbSim(
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        repetitions=args.repetitions,
    )

    if args.error_rate is not None:
        sim.run_experiment([args.error_rate])
        return

    default_error_rates = [
        1.0e-7,
        7.84759970e-6,
        3.35981829e-5,
        6.15848211e-4,
        1.12883789e-2,
        2.06130785e-2,
        2.33572147e-2,
        3.11537409e-2,
        3.62021775e-2,
        4.20687089e-2,
        4.83293024e-2,
        5.47144504e-2,
        6.35808794e-2,
        7.38841056e-2,
        8.58569606e-2,
        9.25524149e-2,
        1.0e-1,
        1.43844989e-1,
        2.06913808e-1,
        3.35981829e-1,
    ]
    sim.run_experiment(default_error_rates)


if __name__ == '__main__':
    main()
