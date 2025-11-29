# Quick Start Guide

Get up and running with the quantum double-slit simulation in 5 minutes!

## Installation

```bash
# Navigate to project directory
cd /home/cal/double-slit-experiment

# Activate virtual environment
source venv/bin/activate

# Already installed: numpy, scipy, matplotlib
# To add GPU support (optional):
pip install cupy-cuda12x  # for CUDA 12.x
```

## Run Your First Simulation

### Example 1: Free Particle (1D)
```bash
cd quantum_double_slit/examples
python example_01_free_particle_1d.py
```

**Output:** 5 plots + 1 animation showing quantum wave packet evolution
- Perfect momentum conservation (10⁻¹² error!)
- Wave packet spreading
- Heisenberg uncertainty principle

### Example 2: Double-Slit Experiment (2D) 🌟
```bash
python example_02_double_slit_2d.py
```

**Output:** 6 plots + 1 animation showing **quantum interference**
- Clear interference pattern with 14 fringes
- Wave-particle duality visualization
- Complete physics analysis

## Use in Your Own Code

```python
import numpy as np
from quantum_double_slit.core import WaveFunction2D, SplitOperatorSolver2D
from quantum_double_slit.core.evolution import double_slit_potential_2d
from quantum_double_slit.visualization.plot2d import plot_probability_density

# Create spatial grid
x = np.linspace(-6, 6, 256)
y = np.linspace(-5, 15, 384)

# Initialize Gaussian wave packet
psi = WaveFunction2D.gaussian(x, y,
                              x0=0, y0=-3.5,
                              sigma_x=4.0, sigma_y=0.8,
                              kx0=0, ky0=10.0)

# Create double-slit potential
solver = SplitOperatorSolver2D(
    psi,
    lambda X, Y: double_slit_potential_2d(X, Y,
                                         slit_separation=3.0,
                                         slit_width=0.8),
    dt=0.002
)

# Evolve and visualize
snapshots, times = solver.evolve(t_final=2.0, n_snapshots=100)
plot_probability_density(snapshots[-1], title="Interference Pattern")
```

## View Generated Files

All examples save plots and animations to `quantum_double_slit/examples/`:

```bash
cd quantum_double_slit/examples
ls -lh *.png *.gif

# View with your favorite image viewer
# Example on Linux:
eog 02_complete_analysis.png
animate 02_evolution.gif
```

## Generated Files Overview

### Example 1 (Free Particle)
- `01_initial_wavefunction.png` - Starting wave function
- `01_initial_probability.png` - Initial probability density
- `01_final_probability.png` - After spreading
- `01_dynamics.png` - Position and momentum vs time
- `01_evolution.gif` - Full animation (1.5 MB)

### Example 2 (Double-Slit) ⭐
- `02_initial_state.png` - Incoming wave
- `02_potential.png` - Barrier visualization
- `02_final_state.png` - **Interference pattern!**
- `02_interference_pattern.png` - 1D cross-section
- `02_complete_analysis.png` - Multi-panel analysis
- `02_evolution.gif` - Watch interference form! (5 MB)

## Interactive Jupyter Notebook (Coming Soon)

```bash
# Install Jupyter (if not already)
pip install jupyter ipywidgets

# Start notebook
jupyter notebook
```

Then create a new notebook:

```python
from quantum_double_slit.core import *
from quantum_double_slit.visualization import *
import ipywidgets as widgets

# Interactive sliders for parameters
# Real-time visualization updates
# Live physics exploration!
```

## Enable GPU Acceleration

```bash
# Check CUDA version
nvidia-smi

# Install CuPy (for CUDA 12.x)
pip install cupy-cuda12x

# Test GPU
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Then modify code:
```python
# Future: Just set backend='gpu'
# from quantum_double_slit.core import WaveFunction2D
# psi = WaveFunction2D.gaussian(..., backend='gpu')
#
# Expected: 50-200x speedup on NVIDIA 3050!
```

## Troubleshooting

### matplotlib backend issues
```bash
export MPLBACKEND=TkAgg  # or Qt5Agg, Agg for no display
```

### Animation not saving
```bash
# Install pillow for GIF
pip install pillow

# Install ffmpeg for MP4
sudo dnf install ffmpeg  # Fedora/RHEL
sudo apt install ffmpeg  # Ubuntu/Debian
```

### Import errors
```bash
# Make sure you're in the project root
cd /home/cal/double-slit-experiment

# Activate environment
source venv/bin/activate

# Run examples from examples directory
cd quantum_double_slit/examples
python example_02_double_slit_2d.py
```

## What's Next?

See [STATUS.md](STATUS.md) for roadmap and [README.md](README.md) for full documentation.

### Recommended Next Steps:
1. ✅ Run both examples and explore the output
2. 📊 Modify parameters (slit separation, wave momentum, etc.)
3. 🚀 Enable GPU acceleration for real-time simulation
4. 🔬 Implement observer effect (wave function collapse)
5. 🎮 Create interactive interface with sliders

## Performance Tips

### For faster 2D simulations:
- Reduce grid size: `256×384` → `128×192` (4x faster)
- Increase timestep: `dt=0.002` → `dt=0.005` (2.5x faster)
- Fewer snapshots: `n_snapshots=100` → `n_snapshots=50` (2x faster)

### For higher quality:
- Increase grid: `256×384` → `512×768` (needs GPU!)
- Decrease timestep: `dt=0.002` → `dt=0.001` (more accurate)
- More snapshots: `n_snapshots=100` → `n_snapshots=200` (smoother animation)

## Learning Resources

1. **Code examples** - Start with `example_01` then `example_02`
2. **Inline documentation** - Read docstrings in `core/wavefunction.py`
3. **Visualization** - Explore different modes in `plot2d.py`
4. **Physics background** - See implementation plan document

## Support

For questions or issues:
1. Check [STATUS.md](STATUS.md) for known issues
2. Review example code in `quantum_double_slit/examples/`
3. Check inline documentation (docstrings)

## Key Results to Expect

- ✓ Momentum conservation to 10⁻¹² precision
- ✓ Norm conservation to 10⁻¹² precision
- ✓ Clear quantum interference fringes
- ✓ Smooth, accurate time evolution
- ✓ Beautiful visualizations

Enjoy exploring quantum mechanics! 🚀
