"""
test_noise.py
=============

Manual smoke-test script for noise primitives in `noise.py`.

This script exercises:
- physical depolarizing channel application,
- Kraus-operator generation utility,
- logical depolarizing channel remapping and execution.
"""

import cirq
import numpy as np
from noise import QuditDepolarizingChannel, kraus, LogicalDepolarizingNoise


def main():
    """
    Execute interactive noise-channel sanity checks and print diagnostics.
    
    Args:
    None: `main` relies on object state and accepts no additional inputs.
    
    Returns:
    None: `main` executes for side effects only and returns no payload.
    
    Raises:
    ValueError: If `main` receives inputs that are incompatible with its
    expected configuration.
    """
    print("Testing QuditDepolarizingChannel...")
    depol = QuditDepolarizingChannel(3, 0.01)
    qudit = cirq.LineQid(0, dimension=3)
    # Apply one noisy channel call, then measure, to verify Cirq integration.
    circuit = cirq.Circuit(depol.on(qudit), cirq.measure(qudit, key="result"))
    print("Circuit:")
    print(circuit)
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=5)
    print("Results:")
    print(result)

    print("\nTesting kraus function...")
    # Ensure helper returns the full operator set for the configured channel.
    kraus_list = kraus(3, 0.01)
    print("Number of Kraus operators:", len(kraus_list))

    print("\nTesting LogicalDepolarizingNoise...")
    # Define a dummy logical operator group.
    dummy_qubit = cirq.NamedQid("dummy", dimension=3)
    noise_circuit = cirq.Circuit(cirq.X(dummy_qubit))
    mergedLogicalP3Group = {
        'X': ([0], noise_circuit),
        'I': ([0],
              cirq.Circuit(cirq.IdentityGate(qid_shape=(3,))(dummy_qubit)))
    }
    logical_noise = LogicalDepolarizingNoise(0.1, mergedLogicalP3Group,
                                             full_qudit_count=1)
    qudit = cirq.LineQid(0, dimension=3)
    # This call triggers sampled logical-noise decomposition on the target.
    circuit = cirq.Circuit(logical_noise.on(qudit))
    print("Logical noise circuit:")
    print(circuit)
    result = sim.run(circuit, repetitions=5)
    print("Logical noise measurement results:")
    print(result)
