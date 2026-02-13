#!/usr/bin/env python3
"""
qutrit_logical_rb_terminal_check_sim.py
=======================================

Logical randomized benchmarking (LRB) simulation for the folded [[5,1,2]]_3
qutrit code with:
1. terminal-only stabilizer checks, and
2. local physical depolarizing noise inserted after each logical gate block.

Purpose
-------
This module isolates the "terminal-check local-noise" experiment variant from
other LRB implementations. It keeps data generation, checkpointing, and fitting
in one place while delegating all plotting/comparison work to
`qutrit_rb_plotting.py`.

Core design choice
------------------
Unlike the logical-noise model (which samples logical Pauli channels), this
variant explicitly rebuilds each RB sequence and injects *physical* one-qutrit
depolarizing channels immediately after each logical gate block, acting only on
the qutrits touched by that block.

Experiment protocol
-------------------
For each physical depolarizing probability `p`:
1. Generate logical RB sequences from the merged logical Clifford group without
   automatic noise insertion.
2. Rebuild each sequence into a concrete circuit where each logical gate block
   is followed by local physical noise on active qutrits.
3. Append a single terminal readout stage:
   - `H^-1` on each code qutrit (X-basis readout convention),
   - one terminal measurement moment over all five qutrits.
4. Prepend encoded logical plus-state preparation and run shot simulation.
5. Post-select accepted shots using the two folded-code X-stabilizer equations.
6. Extract depth-resolved expectation values and fit RB decay.

Checkpointing
-------------
Per-rate checkpoint:
`<checkpoint_dir>/logicalRB_p{p}.pkl`

Final sweep summary:
`<checkpoint_dir>/final_logical_error_rate.pkl`

Compatibility notes
-------------------
The loader accepts both snake_case and legacy camelCase checkpoint key naming
to support reuse of older generated data.
"""

import os
import numpy as np
import cirq
from scipy.optimize import curve_fit
from typing import Any, Dict, List, Optional, Sequence, Union

from rb_checkpoint import saveCheckpoint, loadCheckpoint
from logical_rb_seq import LogicalRBSeq
from qutrit_folded_logical_plus_state import QutritFoldedLogicalPlusState
from qutrit_logical_clifford import QutritLogicalClifford
from noise import QuditDepolarizingChannel
from gates import QuditHadamard


class LogicalRbTerminalCheckSim:
    """
    This class runs a terminal-check Logical Randomized Benchmarking (LRB)
    experiment for the folded [[5,1,2]]_3 qutrit code.

    In this variant, physical depolarizing channels are injected *locally*
    after
    each logical gate block and only on qutrits touched by that block. The
    experiment uses terminal-only stabilizer checks (no interval syndrome
    extraction), followed by post-selection and RB fitting.

    The class centralizes experiment assembly, local-noise insertion policy,
    checkpoint resume/save behavior, and per-rate logical infidelity
    extraction.

    Attributes:
        checkpoint_dir (str): Directory containing per-rate checkpoints and the
            final sweep summary.
        reps (int): Number of shots per RB circuit.
        seed (int): Seed used for randomized sequence generation.
        dim (int): Qutrit dimension (fixed at 3 in this project).
        circuits_per_sequence (int): Number of circuits per sequence depth for
            depth-key aggregation.
        qutrits (List[cirq.GridQid]): Ordered folded-layout code qutrits.
        logical_gate_set (Dict[str, cirq.Circuit]): Logical Clifford circuits
            used to generate RB sequences.
        logical_plus_state (cirq.Circuit): Encoded logical plus-state
            preparation circuit.
        hadamard_inverse (cirq.Circuit): Terminal basis-transform circuit
            applied before final measurements.

    Methods:
        _fit_func(x, amplitude, decay):
            Static RB decay model `A * f^x` used by curve fitting.
        __init__(checkpoint_dir, reps=10000, seed=24):
            Builds and stores fixed experiment resources.
        _build_rb_sequences():
            Generates logical RB sequences without automatic noise.
        _active_qutrits(circuit):
            Returns the qutrits touched by a logical gate block.
        _append_terminal_measurements(circuit):
            Appends terminal basis transform and final measurements.
        _build_local_noisy_rb_circuit(rb_sequence, noise_gate):
            Reconstructs one RB sequence with local post-gate noise insertion.
        _run_single_circuit_simulation(rb_circuit, repetitions, key, verbose):
            Runs one circuit, applies post-selection, and returns derived
            metrics.
        run_simulation(physical_error_rate, verbose=True):
            Runs one physical-rate job end-to-end and returns logical error
            rate.
        find_logical_error_rate(physical_error_rates, verbose=True):
            Sweeps multiple physical rates and returns a mapping of fitted
            logical error rates.
    """

    @staticmethod
    def _fit_func(x, amplitude, decay):
        """
Standard RB exponential decay model.

Model: y(m) = A * f^m

Args:
    x (Any): Independent-variable values (for example RB depth values) used by
             the model function.
    amplitude (Any): Model amplitude parameter used in the fitted decay
                     expression.
    decay (Any): Model decay parameter used in the fitted RB expression.

Returns:
    object: Output produced by this routine according to the behavior described
            above (standard rb exponential decay model.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return amplitude * (decay ** x)

    def _resolve_depth_keys(
        self,
        total_circuits: int,
        sequence_lengths: Optional[List[int]] = None,
        checkpoint_depth_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """
Resolve one depth key per circuit index for robust aggregation.

Priority:
1. Use checkpoint-provided `depth_keys` when available and aligned.
2. Build from explicit `sequence_lengths` + `circuits_per_sequence`.
3. Fallback to legacy `2 * bucket_index` mapping for old checkpoints.

Args:
    total_circuits (int): Total number of generated RB circuits.
    sequence_lengths (Optional[List[int]]): Ordered depth list used during RB
                                            sequence generation.
    checkpoint_depth_keys (Optional[List[str]]): Depth key list loaded from an
                                                 existing checkpoint.

Returns:
    List[str]: Depth key per circuit index.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if (
            checkpoint_depth_keys is not None
            and len(checkpoint_depth_keys) == total_circuits
        ):
            # Best case: checkpoint already stores one key per circuit index.
            return [str(key) for key in checkpoint_depth_keys]

        if sequence_lengths:
            depth_keys = []
            for depth in sequence_lengths:
                # Expand depth list into per-circuit labels using the
                # configured number of random circuits per depth.
                depth_keys.extend(
                    [str(depth)] * int(self.circuits_per_sequence))
            if len(depth_keys) >= total_circuits:
                return depth_keys[:total_circuits]

        # Legacy fallback for old runs where depth keys were inferred from
        # fixed even steps.
        return [
            str(2 * (circuit_index // self.circuits_per_sequence))
            for circuit_index in range(total_circuits)
        ]

    def __init__(self, checkpoint_dir: str, reps: int = 10000, seed: int = 24):
        """
        Initialize simulation resources.
        
        Args:
            checkpoint_dir (str):
                Input argument consumed by `__init__` to perform this
                                  operation.
            reps (int): Number of shots used for each simulated circuit.
            seed (int):
                Random-number-generator seed used to make circuit sampling
                        reproducible.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.reps = reps
        self.seed = seed
        self.dim = 3
        self.circuits_per_sequence = 30

        # Folded 5-qutrit layout order: q0,q1,q2,q3,q4.
        self.qutrits = [
            cirq.GridQid(0, 0, dimension=self.dim),
            cirq.GridQid(0, 2, dimension=self.dim),
            cirq.GridQid(1, 1, dimension=self.dim),
            cirq.GridQid(2, 0, dimension=self.dim),
            cirq.GridQid(2, 2, dimension=self.dim),
        ]

        self.logical_gate_set = QutritLogicalClifford(
            dimension=self.dim).get_group()
        self.logical_plus_state = (
            QutritFoldedLogicalPlusState().generate_state_circuit()[0]
        )
        self.hadamard_inverse = cirq.Circuit(
            (QuditHadamard(self.dim) ** -1).on_each(self.qutrits))

    def _build_rb_sequences(self) -> LogicalRBSeq:
        """
        Build logical RB sequences without automatic noise insertion.
        
        Args:
        None: `_build_rb_sequences` relies on object state and accepts no
        additional inputs.
        
        Returns:
            LogicalRBSeq: Result object produced by this method.
        
        Raises:
        ValueError: If `_build_rb_sequences` receives inputs that are
        incompatible with its expected configuration.
        """
        return LogicalRBSeq(
            measureQudits=self.qutrits,
            logicalGateSet=self.logical_gate_set,
            measurementBasisTransformation=None,
            seed=self.seed,
            noiseChannel=None,
            measurementInterval=None,
            syndromeCircuits=None,
        )

    @staticmethod
    def _active_qutrits(circuit: cirq.Circuit) -> List[cirq.Qid]:
        """
Extract active qutrits touched by a logical gate block.

Args:
    circuit (cirq.Circuit): Circuit object consumed, transformed, or analyzed
                            by this method.

Returns:
    List[cirq.Qid]: Output produced by this routine according to the behavior
                    described above (extract active qutrits touched by a
                    logical gate block.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        active = {q for op in circuit.all_operations() for q in op.qubits}
        return sorted(active, key=lambda q: (
            getattr(q, "row", -1), getattr(q, "col", -1), str(q)))

    def _append_terminal_measurements(
            self, circuit: cirq.Circuit) -> cirq.Circuit:
        """
Append terminal-only measurement stage.

The stage is: 1) `H^-1` on each code qutrit (for X-basis readout), 2)
one terminal measurement moment on all 5 qutrits.

Args:
    circuit (cirq.Circuit): Circuit object consumed, transformed, or analyzed
                            by this method.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (append terminal-only measurement stage.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        full_circuit = circuit.copy()
        full_circuit += self.hadamard_inverse
        full_circuit.append(cirq.Moment(
            [cirq.measure(q, key=f"Terminal_{q}") for q in self.qutrits]))
        return full_circuit

    def _build_local_noisy_rb_circuit(
        self,
        rb_sequence: Sequence[Union[cirq.Circuit, str]],
        noise_gate: cirq.Gate,
    ) -> cirq.Circuit:
        """
Reconstruct one RB circuit with local noise after each gate block.

For each logical gate block in the sequence: - append the gate block, -
find active qutrits in that block, - append `noise_gate` only on those
active qutrits.

Sequence markers (`"M"`, `"tM"`) are ignored here because this
simulator enforces a single terminal-check stage at the end.

Args:
    rb_sequence (Sequence[Union[cirq.Circuit, str]]): Input argument consumed
                                                      by `_build_local_noisy_rb
                                                      _circuit` to perform this
                                                      operation.
    noise_gate (cirq.Gate): Noise-channel gate instance inserted according to
                            this method's placement strategy.

Returns:
    cirq.Circuit: Output produced by this routine according to the behavior
                  described above (reconstruct one rb circuit with local noise
                  after each gate block.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        noisy_circuit = cirq.Circuit()

        for sequence_item in rb_sequence:
            if isinstance(sequence_item, str):
                continue
            if not isinstance(sequence_item, cirq.Circuit):
                continue

            gate_block = sequence_item.copy()
            noisy_circuit += gate_block
            for qutrit in self._active_qutrits(gate_block):
                noisy_circuit.append(noise_gate.on(qutrit))

        return self._append_terminal_measurements(noisy_circuit)

    def _run_single_circuit_simulation(
        self,
        rb_circuit: cirq.Circuit,
        repetitions: int,
        key: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
Simulate one RB circuit and compute post-selected metrics.

Post-selection checks: (m0 + m1 - m2) mod 3 == 0 (m3 - m1 - m4) mod 3
== 0

Logical-X readout: (m0 + m3) mod 3

Args:
    rb_circuit (cirq.Circuit): Randomized-benchmarking circuit instance
                               evaluated in this run.
    repetitions (int): Number of Monte Carlo shots executed for the circuit
                       simulation.
    key (str): Depth-bin or experiment key used to label returned statistics.
    verbose (bool): When `True`, prints progress updates and intermediate
                    statistics.

Returns:
    Dict[str, Any]: Output produced by this routine according to the behavior
                    described above (simulate one rb circuit and compute post-
                    selected metrics.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)

        full_circuit = self.logical_plus_state.copy()
        full_circuit += rb_circuit

        rb_results = simulator.run(full_circuit, repetitions=repetitions)

        expected_keys = [f"Terminal_{q}" for q in self.qutrits]
        if any(k not in rb_results.measurements for k in expected_keys):
            expected_keys = list(rb_results.measurements.keys())[:5]

        result_columns = []
        measurements = {}

        for meas_key in expected_keys:
            measurement_column = rb_results.measurements[meas_key].reshape(
                repetitions, -1)[:, 0]
            result_columns.append(measurement_column)
            measurements[meas_key] = measurement_column.tolist()

        results_matrix = np.column_stack(result_columns)

        valid_indices = [
            i
            for i in range(repetitions)
            if (
                (
                    results_matrix[i][0]
                    + results_matrix[i][1]
                    - results_matrix[i][2]
                ) % 3 == 0
                and (
                    results_matrix[i][3]
                    - results_matrix[i][1]
                    - results_matrix[i][4]
                ) % 3 == 0
            )
        ]

        survival_ratio = len(valid_indices) / repetitions
        logical_x_readouts = [
            (results_matrix[i][0] + results_matrix[i][3]) %
            3 for i in valid_indices]

        probability_0 = logical_x_readouts.count(
            0) / len(valid_indices) if valid_indices else 0
        probability_1 = logical_x_readouts.count(
            1) / len(valid_indices) if valid_indices else 0
        probability_2 = logical_x_readouts.count(
            2) / len(valid_indices) if valid_indices else 0

        logical_success = probability_0
        omega = np.exp(2j * np.pi / 3)
        expectation_value = np.abs(
            probability_0
            + probability_1 * omega
            + probability_2 * (omega ** 2)
        )

        if verbose:
            print(f"Accepted indices: {len(valid_indices)}/{repetitions}")
            print(f"Effective shots: {len(logical_x_readouts)}")
            print(f"Key: {key}")
            print(f"Expectation value: {expectation_value}\n")

        return {
            key: (
                full_circuit,
                measurements,
                expectation_value,
                survival_ratio,
                logical_success)}

    def run_simulation(self, physical_error_rate: float,
                       verbose: bool = True) -> float:
        """
Run terminal-check logical RB for one physical depolarizing rate.

Args:
    physical_error_rate (float): Physical depolarizing probability used for the
                                 current simulation or analysis call.
    verbose (bool): When `True`, prints progress updates and intermediate
                    statistics.

Returns:
    float: Simulation or analysis result for the requested execution.

Raises:
    ValueError: If required experiment parameters are invalid or inconsistent.
"""
        checkpoint_file = os.path.join(
            self.checkpoint_dir, f"logicalRB_p{physical_error_rate}.pkl")
        checkpoint = loadCheckpoint(checkpoint_file)

        noise_gate = QuditDepolarizingChannel(self.dim, physical_error_rate)

        if checkpoint:
            # Resume previously started rate run from disk.
            expectation_values = checkpoint.get(
                "expectation_values", checkpoint.get("expectationValues", {}))
            survival_ratios = checkpoint.get(
                "survival_ratios", checkpoint.get("survivalRatios", {}))
            logical_successes = checkpoint.get(
                "logical_successes", checkpoint.get("logicalSuccesses", {}))
            rb_sequences = checkpoint.get(
                "rb_sequences", checkpoint.get("rbSequences", []))
            rb_string_sequences = checkpoint.get(
                "rb_string_sequences", checkpoint.get("rbStringSequences", []))
            sequence_lengths = checkpoint.get(
                "sequence_lengths", checkpoint.get("sequenceLengths"))
            checkpoint_depth_keys = checkpoint.get(
                "depth_keys", checkpoint.get("depthKeys"))
            start_index = checkpoint.get("index", 0)
        else:
            # Fresh run: generate RB sequences and initialize empty aggregates.
            rb_builder = self._build_rb_sequences()
            rb_sequences = rb_builder.get_rb_seqs()
            rb_string_sequences = rb_builder.string_seqs
            sequence_lengths = list(rb_builder.sequenceLengths)
            checkpoint_depth_keys = rb_builder.get_depth_keys()
            expectation_values = {}
            survival_ratios = {}
            logical_successes = {}
            start_index = 0

        depth_keys = self._resolve_depth_keys(
            total_circuits=len(rb_sequences),
            sequence_lengths=sequence_lengths,
            checkpoint_depth_keys=checkpoint_depth_keys,
        )

        for circuit_index in range(start_index, len(rb_sequences)):
            sequence_label = rb_string_sequences[circuit_index]
            # Depth label comes from per-circuit mapping, not arithmetic.
            key = depth_keys[circuit_index]

            if verbose:
                print(
                    f"Running circuit {circuit_index}: "
                    f"{sequence_label} with key {key}"
                )

            rb_circuit = self._build_local_noisy_rb_circuit(
                rb_sequences[circuit_index], noise_gate)
            result = self._run_single_circuit_simulation(
                rb_circuit, self.reps, key, verbose=verbose)

            # Accumulate depth-binned metrics used later for RB fitting.
            for result_key, (_, _, expectation_value, survival_ratio,
                             logical_success) in result.items():
                expectation_values.setdefault(
                    result_key, []).append(expectation_value)
                survival_ratios.setdefault(
                    result_key, []).append(survival_ratio)
                logical_successes.setdefault(
                    result_key, []).append(logical_success)

            if circuit_index % 10 == 0:
                # Periodic checkpoint protects long cluster jobs against
                # interruptions.
                saveCheckpoint(
                    {
                        "expectation_values": expectation_values,
                        "survival_ratios": survival_ratios,
                        "logical_successes": logical_successes,
                        "rb_sequences": rb_sequences,
                        "rb_string_sequences": rb_string_sequences,
                        "sequence_lengths": sequence_lengths,
                        "depth_keys": depth_keys,
                        "index": circuit_index + 1,
                    },
                    checkpoint_file,
                )

        saveCheckpoint(
            {
                "expectation_values": expectation_values,
                "survival_ratios": survival_ratios,
                "logical_successes": logical_successes,
                "rb_sequences": rb_sequences,
                "rb_string_sequences": rb_string_sequences,
                "sequence_lengths": sequence_lengths,
                "depth_keys": depth_keys,
                "index": len(rb_sequences),
            },
            checkpoint_file,
        )

        # Fit the average expectation decay across RB depths.
        average_expectation = {key: np.mean(
            expectation_values[key]) for key in expectation_values}
        x_data = [int(key) for key in average_expectation]
        y_data = [average_expectation[key] for key in average_expectation]

        fit_parameters, _ = curve_fit(
            self._fit_func,
            x_data,
            y_data,
            p0=[0.5, 0.5],
            bounds=([0, 0], [1, 1]),
        )
        decay_parameter = fit_parameters[1]
        logical_error_rate = 1 - \
            ((1 + (self.dim - 1) * decay_parameter) / self.dim)

        if verbose:
            print(
                f"Logical error rate for p={physical_error_rate}: "
                f"{logical_error_rate}"
            )

        return logical_error_rate

    def find_logical_error_rate(
        self,
        physical_error_rates: List[float],
        verbose: bool = True,
    ) -> Dict[float, float]:
        """
Run terminal-check logical RB over multiple physical error rates.

Args:
    physical_error_rates (List[float]): Ordered list of physical depolarizing
                                        probabilities used for a sweep
                                        experiment.
    verbose (bool): When `True`, prints progress updates and intermediate
                    statistics.

Returns:
    Dict[float, float]: Output produced by this routine according to the
                        behavior described above (run terminal-check logical rb
                        over multiple physical error rates.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        results = {}

        for physical_error_rate in physical_error_rates:
            if verbose:
                print(
                    f"\n=== Running analysis for physical error rate "
                    f"p={physical_error_rate} ==="
                )

            results[physical_error_rate] = self.run_simulation(
                physical_error_rate, verbose=verbose)
            saveCheckpoint(results, os.path.join(
                self.checkpoint_dir, "final_logical_error_rate.pkl"))

        return results


def main():
    """
    Command-line entrypoint for terminal-check local-noise LRB. CLI
    arguments: - `--checkpoint-dir`: output directory for
    checkpoints/results. - `--reps`: shots per RB circuit. - `--seed`: RB
    sequence random seed. - `--error-rate`: optional single-rate run. If
    `--error-rate` is omitted, the default multi-rate sweep is executed.
    
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
        description=(
            "Logical RB with terminal stabilizer checks and local per-gate "
            "depolarizing noise."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="qutrit_rb_results_terminal_check_local_noise",
        help="Directory to store checkpoints/results.",
    )
    parser.add_argument("--reps", type=int, default=10000,
                        help="Repetitions per RB circuit.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed for RB sequence generation.")
    parser.add_argument(
        "--error-rate",
        type=float,
        help=(
            "Run a single physical error rate p. If omitted, run the "
            "default sweep."
        ),
    )
    args = parser.parse_args()

    sim = LogicalRbTerminalCheckSim(
        checkpoint_dir=args.checkpoint_dir,
        reps=args.reps,
        seed=args.seed,
    )

    if args.error_rate is not None:
        logical_error = sim.run_simulation(args.error_rate, verbose=True)
        print(
            f"Final logical error rate for p={args.error_rate}: "
            f"{logical_error}"
        )
        return

    default_error_rates = [
        1.0e-7,
        7.8476e-6,
        3.3598e-5,
        6.1585e-4,
        1.1288e-2,
        2.0613e-2,
        2.3357e-2,
        3.1154e-2,
        3.6202e-2,
        4.2069e-2,
        4.8329e-2,
        5.4714e-2,
        6.3581e-2,
        7.3884e-2,
        8.5857e-2,
        9.2552e-2,
        1.0e-1,
        1.4384e-1,
        2.0691e-1,
        3.3598e-1,
    ]
    final_results = sim.find_logical_error_rate(
        default_error_rates, verbose=True)

    print("\nAll done. Physical -> logical error rates:")
    for physical_error_rate, logical_error in final_results.items():
        print(f"p={physical_error_rate}: logical_error_rate={logical_error}")


if __name__ == "__main__":
    main()
