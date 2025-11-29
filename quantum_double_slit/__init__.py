"""
Quantum Double-Slit Experiment Simulation

A comprehensive toolkit for simulating quantum wave packet evolution,
interference patterns, and the observer effect.
"""

__version__ = "0.1.0"

from .core.wavefunction import WaveFunction1D, WaveFunction2D
from .core.evolution import SplitOperatorSolver1D, SplitOperatorSolver2D

__all__ = [
    "WaveFunction1D",
    "WaveFunction2D",
    "SplitOperatorSolver1D",
    "SplitOperatorSolver2D",
]
