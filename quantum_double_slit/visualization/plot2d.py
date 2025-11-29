"""
2D wave function visualization tools.

Provides functions for plotting and animating 2D quantum states,
including the double-slit interference pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from quantum_double_slit.core.wavefunction import WaveFunction2D


def plot_wavefunction_2d(wf: WaveFunction2D, mode: str = 'probability',
                         title: str = "2D Wave Function",
                         figsize: tuple = (10, 8), cmap: str = 'viridis') -> plt.Figure:
    """
    Plot a 2D wave function.

    Args:
        wf: WaveFunction2D to plot
        mode: What to plot - 'probability', 'real', 'imag', 'phase', or 'magnitude'
        title: Plot title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    X, Y = np.meshgrid(wf.x, wf.y, indexing='ij')

    if mode == 'probability':
        data = np.abs(wf.psi)**2
        label = 'Probability Density |ψ(x,y)|²'
        cmap = 'hot'
    elif mode == 'magnitude':
        data = np.abs(wf.psi)
        label = 'Magnitude |ψ(x,y)|'
    elif mode == 'real':
        data = wf.psi.real
        label = 'Re(ψ(x,y))'
        cmap = 'RdBu'
    elif mode == 'imag':
        data = wf.psi.imag
        label = 'Im(ψ(x,y))'
        cmap = 'RdBu'
    elif mode == 'phase':
        data = np.angle(wf.psi)
        label = 'Phase arg(ψ(x,y))'
        cmap = 'hsv'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=11)

    plt.tight_layout()
    return fig


def plot_probability_density(wf: WaveFunction2D, title: str = "Probability Density",
                            figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Plot 2D probability density with side projections.

    Args:
        wf: WaveFunction2D to plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                         hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main plot: 2D probability density
    X, Y = np.meshgrid(wf.x, wf.y, indexing='ij')
    prob_density = np.abs(wf.psi)**2

    im = ax_main.pcolormesh(X, Y, prob_density, cmap='hot', shading='auto')
    ax_main.set_xlabel('x', fontsize=12)
    ax_main.set_ylabel('y', fontsize=12)
    ax_main.set_aspect('equal')

    # Top plot: Projection onto x-axis (integrate over y)
    prob_x = np.sum(prob_density, axis=1) * wf.dy
    ax_top.plot(wf.x, prob_x, 'b-', linewidth=2)
    ax_top.fill_between(wf.x, 0, prob_x, alpha=0.3)
    ax_top.set_ylabel('P(x)', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3)

    # Right plot: Projection onto y-axis (integrate over x)
    prob_y = np.sum(prob_density, axis=0) * wf.dx
    ax_right.plot(prob_y, wf.y, 'r-', linewidth=2)
    ax_right.fill_betweenx(wf.y, 0, prob_y, alpha=0.3, color='red')
    ax_right.set_xlabel('P(y)', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_right, pad=0.1)
    cbar.set_label('|ψ(x,y)|²', fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    return fig


def plot_interference_pattern(wf: WaveFunction2D, screen_y: float,
                              title: str = "Interference Pattern at Screen",
                              figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot the interference pattern at a detection screen.

    Args:
        wf: WaveFunction2D to analyze
        screen_y: y-coordinate of the detection screen
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Find closest y-index to screen position
    y_idx = np.argmin(np.abs(wf.y - screen_y))
    y_actual = wf.y[y_idx]

    # Extract probability density at screen
    screen_prob = np.abs(wf.psi[:, y_idx])**2

    ax.plot(wf.x, screen_prob, 'b-', linewidth=2.5)
    ax.fill_between(wf.x, 0, screen_prob, alpha=0.4, color='blue')

    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Detection Probability', fontsize=12)
    ax.set_title(f'{title} (y = {y_actual:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def animate_evolution_2d(snapshots: List[WaveFunction2D], times: List[float],
                         interval: int = 100, mode: str = 'probability',
                         title: str = "2D Wave Function Evolution",
                         save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animation of 2D wave function time evolution.

    Args:
        snapshots: List of WaveFunction2D at different times
        times: List of corresponding times
        interval: Time between frames in milliseconds
        mode: What to display - 'probability', 'magnitude', 'real', 'imag', or 'phase'
        title: Animation title
        save_path: If provided, save animation to this path

    Returns:
        FuncAnimation object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    X, Y = np.meshgrid(snapshots[0].x, snapshots[0].y, indexing='ij')

    # Determine colormap and data range
    if mode == 'probability':
        cmap = 'hot'
        vmin, vmax = 0, max(np.max(np.abs(wf.psi)**2) for wf in snapshots)
        label = '|ψ|²'
    elif mode == 'magnitude':
        cmap = 'viridis'
        vmin, vmax = 0, max(np.max(np.abs(wf.psi)) for wf in snapshots)
        label = '|ψ|'
    elif mode in ['real', 'imag']:
        cmap = 'RdBu'
        max_val = max(np.max(np.abs(wf.psi.real if mode == 'real' else wf.psi.imag))
                     for wf in snapshots)
        vmin, vmax = -max_val, max_val
        label = f'{mode.capitalize()}(ψ)'
    elif mode == 'phase':
        cmap = 'hsv'
        vmin, vmax = -np.pi, np.pi
        label = 'arg(ψ)'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize plot
    def get_data(wf):
        if mode == 'probability':
            return np.abs(wf.psi)**2
        elif mode == 'magnitude':
            return np.abs(wf.psi)
        elif mode == 'real':
            return wf.psi.real
        elif mode == 'imag':
            return wf.psi.imag
        elif mode == 'phase':
            return np.angle(wf.psi)

    im = ax.pcolormesh(X, Y, get_data(snapshots[0]), cmap=cmap,
                      vmin=vmin, vmax=vmax, shading='auto')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=11)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       color='white' if mode == 'probability' else 'black',
                       bbox=dict(boxstyle='round', facecolor='black' if mode == 'probability' else 'white',
                                alpha=0.7))

    def animate(frame):
        wf = snapshots[frame]
        t = times[frame]

        data = get_data(wf)
        im.set_array(data.ravel())
        time_text.set_text(f'Time: t = {t:.3f}')

        return im, time_text

    anim = FuncAnimation(fig, animate, frames=len(snapshots),
                        interval=interval, blit=False, repeat=True)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg',
                 fps=1000//interval)
        print("Animation saved!")

    return anim
