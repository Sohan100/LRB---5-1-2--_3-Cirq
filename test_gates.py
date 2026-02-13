"""
test_gates.py
=============

Ad-hoc executable checks for custom gate behavior and small simulation sanity
tests. This file is not a formal unit-test suite; it is intended for manual
inspection of circuit diagrams/unitaries and quick operational smoke tests.
"""

import cirq
import numpy as np
import copy
from gates import (
    QuditI, ProductGate, QuditZ, QuditX, QuditS,
    QuditHadamard, QuditSwap, QuditCZ, QuditCNOT
)


def main():
    """
    Run lightweight gate demonstrations and sanity checks. The routine
    prints: - a simple inverse-Z example, - a `ProductGate` symbol-
    composition example, - a two-qutrit swap/measurement simulation
    snippet.
    
    Args:
    None: `main` relies on object state and accepts no additional inputs.
    
    Returns:
    None: `main` executes for side effects only and returns no payload.
    
    Raises:
    ValueError: If `main` receives inputs that are incompatible with its
    expected configuration.
    """
    # Create a 3-dimensional qudit and apply QuditZ^-1.
    qudit = cirq.LineQid(0, dimension=3)
    circuit = cirq.Circuit((QuditZ(3)**-1).on(qudit))
    print("Circuit with QuditZ^-1:")
    print(circuit)
    print("\nUnitary of QuditZ:")
    print(cirq.unitary(QuditZ(3)))

    # Test the ProductGate with a few simple cases.
    def testProductGate():
        """
        Minimal inline check that `ProductGate` renders expected diagram
        labels.

        Args:
            None: This helper does not accept runtime parameters.

        Returns:
            None: Prints the diagram symbol for a representative `ProductGate`
                  construction.

        Raises:
            ValueError: If gate construction fails due invalid gate list
                        contents.
        """
        # Test 1: Product of X, Z, and Hadamard.
        # This specifically checks composed-symbol readability in diagrams.
        pg1 = ProductGate([QuditX(3), QuditZ(3), QuditHadamard(3)])
        info = cirq.circuit_diagram_info(pg1).wire_symbols
        print("Test 1 ProductGate diagram symbol:", info[0])

    testProductGate()

    # Additional tests and simulations can be added here.
    print("\nSimulation demo:")
    # For example, create a simple circuit with a QuditSwap.
    qudit1 = cirq.LineQid(0, dimension=3)
    qudit2 = cirq.LineQid(1, dimension=3)
    circuit2 = cirq.Circuit(
        QuditSwap(3).on(qudit1, qudit2),
        cirq.measure(qudit1, key='q1'),
        cirq.measure(qudit2, key='q2')
    )
    print(circuit2)
    sim = cirq.Simulator()
    # Run a short sample count only; this is a smoke test, not a statistics
    # benchmark.
    result = sim.run(circuit2, repetitions=10)
    print("Measurement results:")
    print(result.measurements)
