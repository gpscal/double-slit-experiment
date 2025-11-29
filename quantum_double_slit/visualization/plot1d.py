"""
1D wave function visualization tools.

Provides functions for plotting and animating 1D quantum states.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from quantum_double_slit.core.wavefunction import WaveFunction1D


def plot_wavefunction_1d(wf: WaveFunction1D, title: str = "Wave Function",
                         show_phase: bool = True, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot a 1D wave function showing real/imaginary parts or magnitude/phase.

    Args:
        wf: WaveFunction1D to plot
        title: Plot title
        show_phase: If True, plot magnitude and phase; if False, plot real and imaginary parts
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    if show_phase:
        # Top: Magnitude |ψ(x)|
        axes[0].plot(wf.x, np.abs(wf.psi), 'b-', linewidth=2, label='|ψ(x)|')
        axes[0].fill_between(wf.x, 0, np.abs(wf.psi), alpha=0.3)
        axes[0].set_ylabel('Magnitude |ψ|', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Bottom: Phase arg(ψ(x))
        phase = np.angle(wf.psi)
        axes[1].plot(wf.x, phase, 'r-', linewidth=2, label='arg(ψ(x))')
        axes[1].set_ylabel('Phase (radians)', fontsize=12)
        axes[1].set_ylim([-np.pi, np.pi])
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        # Top: Real part
        axes[0].plot(wf.x, wf.psi.real, 'b-', linewidth=2, label='Re(ψ)')
        axes[0].set_ylabel('Re(ψ)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Bottom: Imaginary part
        axes[1].plot(wf.x, wf.psi.imag, 'r-', linewidth=2, label='Im(ψ)')
        axes[1].set_ylabel('Im(ψ)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    axes[1].set_xlabel('Position x', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_probability_density_1d(wf: WaveFunction1D, title: str = "Probability Density",
                                figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot probability density |ψ(x)|².

    Args:
        wf: WaveFunction1D to plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    prob_density = np.abs(wf.psi)**2
    ax.plot(wf.x, prob_density, 'b-', linewidth=2.5)
    ax.fill_between(wf.x, 0, prob_density, alpha=0.4, color='blue')

    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Probability Density |ψ(x)|²', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    x_mean = wf.expectation_position()
    x_std = wf.uncertainty_position()
    ax.axvline(x_mean, color='red', linestyle='--', linewidth=2, label=f'⟨x⟩ = {x_mean:.2f}')
    ax.axvspan(x_mean - x_std, x_mean + x_std, alpha=0.2, color='red',
               label=f'Δx = {x_std:.2f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def animate_evolution_1d(snapshots: List[WaveFunction1D], times: List[float],
                        interval: int = 50, title: str = "Wave Function Evolution",
                        save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animation of wave function time evolution.

    Args:
        snapshots: List of WaveFunction1D at different times
        times: List of corresponding times
        interval: Time between frames in milliseconds
        title: Animation title
        save_path: If provided, save animation to this path (e.g., 'animation.gif' or 'animation.mp4')

    Returns:
        FuncAnimation object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Get x range from first snapshot
    x = snapshots[0].x

    # Initialize plot elements
    line_real, = axes[0].plot([], [], 'b-', linewidth=2, label='Re(ψ)')
    line_imag, = axes[0].plot([], [], 'r-', linewidth=2, label='Im(ψ)')
    line_prob, = axes[1].plot([], [], 'g-', linewidth=2.5, label='|ψ|²')
    fill_prob = axes[1].fill_between([], [], [], alpha=0.4, color='green')

    # Set up axes
    max_real = max(np.max(np.abs(wf.psi.real)) for wf in snapshots)
    max_imag = max(np.max(np.abs(wf.psi.imag)) for wf in snapshots)
    max_prob = max(np.max(np.abs(wf.psi)**2) for wf in snapshots)

    axes[0].set_xlim(x[0], x[-1])
    axes[0].set_ylim(-1.2 * max(max_real, max_imag), 1.2 * max(max_real, max_imag))
    axes[0].set_ylabel('ψ(x, t)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlim(x[0], x[-1])
    axes[1].set_ylim(0, 1.2 * max_prob)
    axes[1].set_xlabel('Position x', fontsize=12)
    axes[1].set_ylabel('Probability Density |ψ|²', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        time_text.set_text('')
        return line_real, line_imag, line_prob, time_text

    def animate(frame):
        wf = snapshots[frame]
        t = times[frame]

        line_real.set_data(x, wf.psi.real)
        line_imag.set_data(x, wf.psi.imag)
        line_prob.set_data(x, np.abs(wf.psi)**2)

        # Update fill - remove old collections and add new one
        for coll in axes[1].collections:
            coll.remove()
        axes[1].fill_between(x, 0, np.abs(wf.psi)**2, alpha=0.4, color='green')

        time_text.set_text(f'Time t = {t:.3f}')

        return line_real, line_imag, line_prob, time_text

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(snapshots), interval=interval,
                        blit=False, repeat=True)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg',
                 fps=1000//interval)
        print("Animation saved!")

    return anim
