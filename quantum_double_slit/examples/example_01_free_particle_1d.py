"""
Example 1: Free Particle Evolution in 1D

Demonstrates:
- Creating a Gaussian wave packet
- Time evolution of a free particle (V=0)
- Basic visualization
- Verification of spreading and momentum conservation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from quantum_double_slit.core.wavefunction import WaveFunction1D
from quantum_double_slit.core.evolution import SplitOperatorSolver1D, free_particle_potential_1d
from quantum_double_slit.visualization.plot1d import (
    plot_wavefunction_1d, plot_probability_density_1d, animate_evolution_1d
)


def main():
    print("=" * 60)
    print("Free Particle in 1D - Quantum Wave Packet Evolution")
    print("=" * 60)

    # Set up spatial grid
    x = np.linspace(-20, 20, 512)
    print(f"\nGrid: {len(x)} points from x = {x[0]:.1f} to {x[-1]:.1f}")

    # Create Gaussian wave packet
    x0 = -10.0      # Initial center position
    sigma = 1.0     # Initial width
    k0 = 5.0        # Initial wave number (momentum p = ℏk)
    hbar = 1.0      # Atomic units
    m = 1.0         # Particle mass

    print(f"\nInitial wave packet:")
    print(f"  Position: x₀ = {x0:.1f}")
    print(f"  Width: σ = {sigma:.1f}")
    print(f"  Momentum: ℏk₀ = {hbar * k0:.1f}")

    psi0 = WaveFunction1D.gaussian(x, x0=x0, sigma=sigma, k0=k0, hbar=hbar, m=m)

    # Calculate initial properties
    x_init = psi0.expectation_position()
    p_init = psi0.expectation_momentum()
    dx_init = psi0.uncertainty_position()
    dp_init = psi0.uncertainty_momentum()

    print(f"\nInitial quantum state:")
    print(f"  ⟨x⟩ = {x_init:.3f}")
    print(f"  ⟨p⟩ = {p_init:.3f}")
    print(f"  Δx = {dx_init:.3f}")
    print(f"  Δp = {dp_init:.3f}")
    print(f"  Δx·Δp = {dx_init * dp_init:.3f} (Heisenberg limit: ℏ/2 = {hbar/2:.3f})")

    # Plot initial wave function
    fig1 = plot_wavefunction_1d(psi0, title="Initial Wave Function", show_phase=True)
    plt.savefig("01_initial_wavefunction.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved initial wave function plot: 01_initial_wavefunction.png")

    fig2 = plot_probability_density_1d(psi0, title="Initial Probability Density")
    plt.savefig("01_initial_probability.png", dpi=150, bbox_inches='tight')
    print("✓ Saved initial probability density: 01_initial_probability.png")

    # Set up time evolution
    dt = 0.01        # Time step
    t_final = 5.0    # Final time
    n_snapshots = 50

    print(f"\nTime evolution:")
    print(f"  Time step: dt = {dt}")
    print(f"  Final time: t = {t_final}")
    print(f"  Number of steps: {int(t_final / dt)}")

    # Create solver for free particle
    solver = SplitOperatorSolver1D(psi0, free_particle_potential_1d, dt=dt)

    # Evolve
    print("\nEvolving wave function...")
    snapshots, times = solver.evolve(t_final, n_snapshots=n_snapshots)
    print(f"✓ Evolution complete! Generated {len(snapshots)} snapshots")

    # Analyze final state
    psi_final = snapshots[-1]
    x_final = psi_final.expectation_position()
    p_final = psi_final.expectation_momentum()
    dx_final = psi_final.uncertainty_position()
    dp_final = psi_final.uncertainty_momentum()

    print(f"\nFinal quantum state:")
    print(f"  ⟨x⟩ = {x_final:.3f} (expected: ~{x0 + p_init * t_final / m:.3f})")
    print(f"  ⟨p⟩ = {p_final:.3f} (should be conserved: {p_init:.3f})")
    print(f"  Δx = {dx_final:.3f} (spreading from {dx_init:.3f})")
    print(f"  Δp = {dp_final:.3f} (should be conserved: {dp_init:.3f})")

    # Verify momentum conservation
    momentum_error = abs(p_final - p_init) / abs(p_init) * 100
    print(f"\nMomentum conservation: {momentum_error:.2e}% error")

    # Plot final state
    fig3 = plot_probability_density_1d(psi_final, title=f"Final Probability Density (t = {t_final})")
    plt.savefig("01_final_probability.png", dpi=150, bbox_inches='tight')
    print("✓ Saved final probability density: 01_final_probability.png")

    # Create animation
    print("\nCreating animation...")
    anim = animate_evolution_1d(snapshots, times, interval=100,
                                title="Free Particle Evolution")
    anim.save("01_evolution.gif", writer='pillow', fps=10)
    print("✓ Saved animation: 01_evolution.gif")

    # Plot spreading over time
    fig4, axes = plt.subplots(2, 1, figsize=(10, 8))

    positions = [wf.expectation_position() for wf in snapshots]
    widths = [wf.uncertainty_position() for wf in snapshots]
    momenta = [wf.expectation_momentum() for wf in snapshots]

    axes[0].plot(times, positions, 'b-', linewidth=2, label='⟨x(t)⟩')
    axes[0].fill_between(times,
                        [p - w for p, w in zip(positions, widths)],
                        [p + w for p, w in zip(positions, widths)],
                        alpha=0.3, label='⟨x⟩ ± Δx')
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Position and Width vs Time', fontsize=13, fontweight='bold')

    axes[1].plot(times, momenta, 'r-', linewidth=2, label='⟨p(t)⟩')
    axes[1].axhline(p_init, color='gray', linestyle='--', label='⟨p⟩ initial')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Momentum', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Momentum Conservation', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig("01_dynamics.png", dpi=150, bbox_inches='tight')
    print("✓ Saved dynamics plot: 01_dynamics.png")

    print("\n" + "=" * 60)
    print("Example complete! Check the generated plots and animation.")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
