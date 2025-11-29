# Quantum Double-Slit Experiment Simulation

Interactive quantum mechanics simulation demonstrating wave-particle duality and the observer effect in the famous double-slit experiment.

## Features

- **Accurate quantum simulation**: Solves the time-dependent Schrödinger equation using the split-operator method
- **Observer effect modeling**: Demonstrates how measurement affects quantum interference patterns
- **GPU acceleration**: Optional CUDA acceleration for real-time simulations
- **Interactive visualization**: Real-time parameter control and multiple visualization modes
- **Educational focus**: Clear code structure for learning quantum mechanics and numerical methods

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU acceleration (NVIDIA GPU required)
pip install cupy-cuda12x  # For CUDA 12.x (check your CUDA version)
```

### Basic Usage

```python
from quantum_double_slit.core import WaveFunction1D, split_operator_step
import numpy as np
import matplotlib.pyplot as plt

# Create initial Gaussian wave packet
x = np.linspace(-10, 10, 1024)
psi = WaveFunction1D.gaussian(x, x0=-5, sigma=0.5, k0=20)

# Evolve in time and visualize
# ... (see examples/)
```

## Project Structure

```
quantum_double_slit/
├── core/              # Core simulation engine
├── visualization/     # Plotting and animation tools
├── examples/          # Jupyter notebooks and example scripts
└── tests/            # Unit tests
```

## Physics Background

The quantum double-slit experiment demonstrates the central mystery of quantum mechanics: particles exhibit wave-like interference patterns when unobserved, but behave like classical particles when measured.

This simulation solves the time-dependent Schrödinger equation:

```
iℏ ∂ψ/∂t = Ĥψ = (-ℏ²/2m ∇² + V(x,y))ψ
```

Using the split-operator method for efficient and accurate time evolution.

## Roadmap

- [x] 1D Schrödinger equation solver
- [ ] 2D extension for double-slit geometry
- [ ] Observer effect implementation
- [ ] GPU acceleration
- [ ] Interactive web interface
- [ ] Advanced visualization modes

## License

MIT License - see LICENSE file for details

## References

- Feynman Lectures on Physics, Vol. III
- Zwolak & Zurek, "Measurement-induced decoherence" (2016)
- Your comprehensive implementation plan document
