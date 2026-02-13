#!/usr/bin/env python3
"""
qutrit_logical_rb_logical_noise_sim.py
======================================

Logical randomized benchmarking (LRB) simulation for the folded [[5,1,2]]_3
qutrit code under a *sampled logical depolarizing* noise model.

Purpose
-------
This file is the simulation driver for the logical-noise LRB experiment. It is
responsible for:
1. Constructing randomized logical Clifford sequences.
2. Injecting sampled logical Pauli noise through `LogicalDepolarizingNoise`.
3. Running shot-based circuit simulation with post-selection.
4. Aggregating depth-resolved expectation/survival/success metrics.
5. Fitting RB decay to estimate average logical gate infidelity.

Scope
-----
- Simulation-only module: plotting is intentionally separated into
  `qutrit_rb_plotting.py`.
- Checkpoint-first workflow: long parameter sweeps can be resumed safely.
- Backward-compatible checkpoint loading: accepts both legacy camelCase keys
  and
  newer snake_case keys.

Experiment protocol summary
---------------------------
For each physical depolarizing parameter `p`:
1. Build a logical RB circuit set over the merged logical Clifford group.
2. Append terminal measurement basis transform (`H^-1`) and terminal
   measurements on all five code qutrits (handled inside `LogicalRBSeq`).
3. For each shot, prepare the folded logical plus state and execute one RB
   circuit.
4. Post-select shots that satisfy the two folded-code X-stabilizer constraints:
   - `(m0 + m1 - m2) mod 3 == 0`
   - `(m3 - m1 - m4) mod 3 == 0`
5. Compute logical-X readout `(m0 + m3) mod 3` on accepted shots.
6. Convert accepted outcome frequencies into an expectation magnitude and group
   values by RB depth key.
7. Fit depth-averaged expectation values to `A * f^m` and convert `f` to
   logical infidelity.

Checkpoint schema
-----------------
Per-`p` checkpoint file:
`<checkpoint_dir>/logicalRB_p{p}.pkl`

Stored fields include:
- `expectation_values` / `expectationValues`
- `survival_ratios` / `survivalRatios`
- `logical_successes` / `logicalSuccesses`
- `rb_circuits` / `rbCircuits`
- `rb_string_sequences` / `rbStringSequences`
- `index`

Sweep summary checkpoint:
`<checkpoint_dir>/final_logical_error_rate.pkl`
"""

import os
import numpy as np
import cirq
from scipy.optimize import curve_fit
from typing import Any, Dict, List, Optional, Tuple

from rb_checkpoint import saveCheckpoint, loadCheckpoint
from logical_rb_seq import LogicalRBSeq
from qutrit_folded_logical_plus_state import QutritFoldedLogicalPlusState
from noise import LogicalDepolarizingNoise
from gates import QuditHadamard
from qutrit_logical_pauli import QutritLogicalPauli
from qutrit_logical_clifford import QutritLogicalClifford


class LogicalRbLogicalNoiseSim:
    """
    This class runs Logical Randomized Benchmarking (LRB) for the folded
    [[5,1,2]]_3 qutrit code using a sampled logical depolarizing channel.

    The class encapsulates full experiment orchestration for one or many
    physical depolarizing probabilities: logical sequence generation,
    checkpoint
    restore/save logic, shot-based simulation, post-selected metric extraction,
    and decay fitting into logical gate infidelity.

    It is designed for long sweeps on local or cluster environments where runs
    may be interrupted. Every rate-specific job stores incremental progress and
    can resume from the last completed circuit index.

    Attributes:
        checkpoint_dir (str): Directory where per-rate and final summary
            checkpoints are stored.
        reps (int): Number of shots used for each RB circuit.
        seed (int): Seed passed to RB sequence generation for reproducible
            sequence sampling.
        dim (int): Local Hilbert-space dimension (3 for qutrit).
        circuits_per_sequence (int): Number of circuits per RB depth used for
            depth-key binning logic.
        logical_pauli_group (dict): Logical Pauli dictionary with active-index
            support used by the logical depolarizing channel.
        logical_clifford_group (dict): Merged logical Clifford group used to
            generate RB sequences.
        qutrits (List[cirq.GridQid]): Ordered folded-layout data qutrits used
            throughout state preparation and measurement.
        logical_plus_state (cirq.Circuit): State-preparation circuit for the
            encoded logical plus state.

    Methods:
        _fit_func(x, amplitude, decay):
            Static RB decay model `A * f^x` used by curve fitting.
        _run_circuit_simulation(circuit, logical_plus_state_circuit,
        repetitions,
                                key, noise_gate, verbose=True):
            Executes one logical RB circuit with post-selection and returns
            depth-keyed metrics.
        __init__(checkpoint_dir, reps=10000, seed=24):
            Initializes static experiment resources and layout.
        run_simulation(physical_error_rate, verbose=True):
            Executes one complete LRB job for a single physical error rate and
            returns the fitted logical error rate.
        find_logical_error_rate(physical_error_rates, verbose=True):
            Runs `run_simulation` across a sweep of physical rates and stores a
            cumulative summary checkpoint.
    """

    @staticmethod
    def _fit_func(x, amplitude, decay):
        """
Standard RB exponential decay model.

This function models the magnitude of the logical expectation value as
a function of RB depth: y(m) = A * f^m where `A` captures state-
preparation and measurement prefactors, and `f` captures per-Clifford
decay.

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
            # Preferred path: explicit per-circuit depth labels loaded from
            # checkpoint.
            return [str(key) for key in checkpoint_depth_keys]

        if sequence_lengths:
            depth_keys = []
            for depth in sequence_lengths:
                # Expand each depth into repeated labels (one per sampled
                # circuit at that depth).
                depth_keys.extend(
                    [str(depth)] * int(self.circuits_per_sequence))
            if len(depth_keys) >= total_circuits:
                return depth_keys[:total_circuits]

        # Backward-compatible fallback for historical fixed-step runs.
        return [
            str(2 * (circuit_index // self.circuits_per_sequence))
            for circuit_index in range(total_circuits)
        ]

    def _run_circuit_simulation(
        self,
        circuit: cirq.Circuit,
        logical_plus_state_circuit: cirq.Circuit,
        repetitions: int,
        key: str,
        noise_gate: cirq.Gate,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
Simulate one logical RB circuit and evaluate post-selected observables.

Why one-shot execution: - This routine executes one repetition at a
time (`repetitions=1`) to force fresh sampling in
`LogicalDepolarizingNoise._decompose_`. - Batched execution can hide
per-shot sampled-operator statistics and can lead to less transparent
diagnostics for logical-channel sampling.

Measurement and post-selection: - All terminal measurement keys are
read into a dense shot matrix. - Shots are accepted iff both folded-
code X-stabilizer constraints are satisfied: 1) `(m0 + m1 - m2) mod 3
== 0` 2) `(m3 - m1 - m4) mod 3 == 0` - Logical-X outcome for accepted
shots is computed as `(m0 + m3) mod 3`.

Derived metrics: - `survival_ratio`: accepted_shots / total_shots -
`logical_success`: probability of logical-X outcome 0 among accepted
shots - `expectation_value`: `|P0 + P1*w + P2*w^2|`, where `w =
exp(2j*pi/3)`

Args:
    circuit (cirq.Circuit): Circuit object consumed, transformed, or analyzed
                            by this method.
    logical_plus_state_circuit (cirq.Circuit): Circuit that prepares the
                                               encoded logical-plus input
                                               state.
    repetitions (int): Number of Monte Carlo shots executed for the circuit
                       simulation.
    key (str): Depth-bin or experiment key used to label returned statistics.
    noise_gate (cirq.Gate): Noise-channel gate instance inserted according to
                            this method's placement strategy.
    verbose (bool): When `True`, prints progress updates and intermediate
                    statistics.

Returns:
    Dict[str, Any]: Output produced by this routine according to the behavior
                    described above (simulate one logical rb circuit and
                    evaluate post-selected observables.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        simulator = cirq.Simulator(dtype=np.complex128)

        full_circuit = logical_plus_state_circuit.copy()
        full_circuit += circuit

        measurements_list = []
        noise_samples = []

        for _ in range(repetitions):
            result = simulator.run(full_circuit.copy(), repetitions=1)
            measurements_list.append(result.measurements)
            noise_samples.append(noise_gate.last_sampled_key or "none")

        measurement_keys = list(measurements_list[0].keys())
        results_matrix = np.zeros(
            (repetitions, len(measurement_keys)), dtype=int)

        for shot_index in range(repetitions):
            for key_index, meas_key in enumerate(measurement_keys):
                results_matrix[shot_index, key_index] = (
                    measurements_list[shot_index][meas_key][0][0]
                )

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
            0) / len(logical_x_readouts) if valid_indices else 0
        probability_1 = logical_x_readouts.count(
            1) / len(logical_x_readouts) if valid_indices else 0
        probability_2 = logical_x_readouts.count(
            2) / len(logical_x_readouts) if valid_indices else 0

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
            print(f"Expectation value: {expectation_value}")
            unique_samples, sample_counts = np.unique(
                np.array(noise_samples), return_counts=True)
            print("Noise sample counts:", dict(
                zip(unique_samples, sample_counts)))

        return {
            key: (
                full_circuit,
                results_matrix,
                expectation_value,
                survival_ratio,
                logical_success,
                np.array(noise_samples),
            )
        }

    def __init__(self, checkpoint_dir: str, reps: int = 10000, seed: int = 24):
        """
        Initialize simulator configuration and static resources.
        
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

        self.logical_pauli_group = QutritLogicalPauli(
            dimension=self.dim).get_group_with_active_indices()
        self.logical_clifford_group = QutritLogicalClifford(
            dimension=self.dim).get_group()

        self.qutrits = [
            cirq.GridQid(0, 0, dimension=self.dim),
            cirq.GridQid(0, 2, dimension=self.dim),
            cirq.GridQid(1, 1, dimension=self.dim),
            cirq.GridQid(2, 0, dimension=self.dim),
            cirq.GridQid(2, 2, dimension=self.dim),
        ]

        self.logical_plus_state = (
            QutritFoldedLogicalPlusState().generate_state_circuit()[0]
        )

    def run_simulation(self, physical_error_rate: float,
                       verbose: bool = True) -> float:
        """
Run logical RB for one physical depolarizing error rate.

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

        logical_noise = LogicalDepolarizingNoise(
            physical_error_rate,
            self.logical_pauli_group,
            full_qudit_count=5,
            dimension=self.dim,
        )

        if checkpoint:
            # Resume in-progress run and reuse saved RB artifacts.
            expectation_values = checkpoint.get(
                "expectation_values", checkpoint.get("expectationValues", {}))
            survival_ratios = checkpoint.get(
                "survival_ratios", checkpoint.get("survivalRatios", {}))
            logical_successes = checkpoint.get(
                "logical_successes", checkpoint.get("logicalSuccesses", {}))
            rb_circuits = checkpoint.get(
                "rb_circuits", checkpoint.get("rbCircuits"))
            rb_string_sequences = checkpoint.get(
                "rb_string_sequences", checkpoint.get("rbStringSequences"))
            sequence_lengths = checkpoint.get(
                "sequence_lengths", checkpoint.get("sequenceLengths"))
            checkpoint_depth_keys = checkpoint.get(
                "depth_keys", checkpoint.get("depthKeys"))
            start_index = checkpoint.get("index", 0)
        else:
            # Fresh generation path: build noisy logical RB circuits.
            hadamard_inverse = cirq.Circuit(
                (QuditHadamard(self.dim) ** -1).on_each(self.qutrits))
            rb_seq = LogicalRBSeq(
                measureQudits=self.qutrits,
                logicalGateSet=self.logical_clifford_group,
                measurementBasisTransformation=hadamard_inverse,
                seed=self.seed,
                noiseChannel=logical_noise,
            )
            rb_circuits = rb_seq.get_rb_circuits()
            rb_string_sequences = rb_seq.string_seqs
            sequence_lengths = list(rb_seq.sequenceLengths)
            checkpoint_depth_keys = rb_seq.get_depth_keys()
            expectation_values = {}
            survival_ratios = {}
            logical_successes = {}
            start_index = 0

        depth_keys = self._resolve_depth_keys(
            total_circuits=len(rb_circuits),
            sequence_lengths=sequence_lengths,
            checkpoint_depth_keys=checkpoint_depth_keys,
        )

        for circuit_index in range(start_index, len(rb_circuits)):
            string_sequence = rb_string_sequences[circuit_index]
            circuit = rb_circuits[circuit_index]
            # Derive depth key from explicit mapping, independent of step size.
            key = depth_keys[circuit_index]

            if verbose:
                print(
                    f"Running circuit {circuit_index}: "
                    f"{string_sequence} with key {key}"
                )

            result = self._run_circuit_simulation(
                circuit,
                self.logical_plus_state,
                self.reps,
                key,
                logical_noise,
                verbose=verbose,
            )

            # Append this circuit's outputs into per-depth metric buckets.
            for result_key, values in result.items():
                expectation_values.setdefault(result_key, []).append(values[2])
                survival_ratios.setdefault(result_key, []).append(values[3])
                logical_successes.setdefault(result_key, []).append(values[4])

            if circuit_index % 10 == 0:
                # Persist progress regularly to support job preemption/restart.
                saveCheckpoint(
                    {
                        "expectation_values": expectation_values,
                        "survival_ratios": survival_ratios,
                        "logical_successes": logical_successes,
                        "rb_circuits": rb_circuits,
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
                "rb_circuits": rb_circuits,
                "rb_string_sequences": rb_string_sequences,
                "sequence_lengths": sequence_lengths,
                "depth_keys": depth_keys,
                "index": len(rb_circuits),
            },
            checkpoint_file,
        )

        # Fit RB decay and convert fitted decay constant to logical infidelity.
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
    ) -> Tuple[List[float], List[float]]:
        """
Sweep logical RB across multiple physical error rates.

Args:
    physical_error_rates (List[float]): Ordered list of physical depolarizing
                                        probabilities used for a sweep
                                        experiment.
    verbose (bool): When `True`, prints progress updates and intermediate
                    statistics.

Returns:
    Tuple[List[float], List[float]]: Output produced by this routine according
                                     to the behavior described above (sweep
                                     logical rb across multiple physical error
                                     rates.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        logical_error_rates = []

        for physical_error_rate in physical_error_rates:
            if verbose:
                print(
                    f"\n=== Running analysis for error rate: "
                    f"{physical_error_rate} ==="
                )

            logical_error = self.run_simulation(
                physical_error_rate, verbose=verbose)
            logical_error_rates.append(logical_error)

            saveCheckpoint(
                {
                    "physical_error_rates": physical_error_rates[
                        : len(logical_error_rates)
                    ],
                    "logical_error_rates": logical_error_rates,
                },
                os.path.join(self.checkpoint_dir,
                             "final_logical_error_rate.pkl"),
            )

        return physical_error_rates, logical_error_rates


# Backward-compatible alias for existing notebooks/scripts.
LogicalRBSim = LogicalRbLogicalNoiseSim


def main():
    """
    Command-line entrypoint for logical-noise LRB runs. CLI options: -
    `--checkpoint-dir`: directory for checkpoints and final summaries. -
    `--reps`: shots per circuit. - `--seed`: RB sequence seed. - `--error-
    rate`: run single physical rate; if omitted, run default sweep.
    
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
        description="Logical randomized benchmarking (logical-noise model).")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoint_logical_rb",
        help="Directory to save checkpoints and results.",
    )
    parser.add_argument("--reps", type=int, default=10000,
                        help="Repetitions per RB circuit.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed for reproducibility.")
    parser.add_argument("--error-rate", type=float,
                        help="Run one physical error rate if specified.")
    args = parser.parse_args()

    sim = LogicalRbLogicalNoiseSim(
        args.checkpoint_dir, reps=args.reps, seed=args.seed)

    if args.error_rate is not None:
        logical_error_rate = sim.run_simulation(args.error_rate)
        print(f"Logical error rate: {logical_error_rate}")
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

    print(f"Running simulations for {len(default_error_rates)} error rates...")
    _, logical_error_rates = sim.find_logical_error_rate(default_error_rates)

    print("Final logical error rates:")
    for physical_error_rate, logical_error_rate in zip(
            default_error_rates, logical_error_rates):
        print(
            f"p={physical_error_rate}: "
            f"logical_error_rate={logical_error_rate}"
        )


if __name__ == "__main__":
    main()
