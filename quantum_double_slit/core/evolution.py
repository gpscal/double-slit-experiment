"""
Time evolution solvers for the Schrödinger equation.

Implements the split-operator method for efficient and accurate
propagation of quantum wave functions.
"""

import numpy as np
from typing import Callable, Optional, List
from .wavefunction import WaveFunction1D, WaveFunction2D


class SplitOperatorSolver1D:
    """
    Solves the 1D time-dependent Schrödinger equation using the split-operator method.

    The Hamiltonian H = K + V is split into kinetic (K) and potential (V) parts.
    Time evolution uses the approximation:
        U(dt) ≈ exp(-i dt V/2ℏ) exp(-i dt K/ℏ) exp(-i dt V/2ℏ)

    The kinetic operator is diagonal in momentum space, so we use FFT for efficiency.
    """

    def __init__(self, wavefunction: WaveFunction1D,
                 potential: Callable[[np.ndarray], np.ndarray],
                 dt: float = 0.01):
        """
        Initialize the solver.

        Args:
            wavefunction: Initial wave function
            potential: Function V(x) that returns potential energy
            dt: Time step size
        """
        self.wf = wavefunction.copy()
        self.V = potential(self.wf.x)
        self.dt = dt

        # Pre-compute time evolution operators
        self._setup_operators()

    def _setup_operators(self) -> None:
        """Pre-compute the time evolution operators for efficiency."""
        # Potential operator (applied in position space)
        # exp(-i V(x) dt / (2ℏ))
        self.exp_V_half = np.exp(-1j * self.V * self.dt / (2 * self.wf.hbar))

        # Kinetic operator (applied in momentum space)
        # exp(-i ℏk²/(2m) dt / ℏ) = exp(-i k²/(2m) dt)
        k = self.wf.k
        self.exp_K = np.exp(-1j * self.wf.hbar * k**2 * self.dt / (2 * self.wf.m))

    def step(self) -> None:
        """
        Advance the wave function by one time step using split-operator method.

        Algorithm:
            1. Apply V/2: ψ → exp(-i V dt/2ℏ) ψ
            2. FFT to momentum space
            3. Apply K: ψ̃ → exp(-i K dt/ℏ) ψ̃
            4. Inverse FFT to position space
            5. Apply V/2: ψ → exp(-i V dt/2ℏ) ψ
        """
        # Step 1: Apply half-step potential
        self.wf.psi *= self.exp_V_half

        # Step 2: Transform to momentum space
        psi_k = np.fft.fft(self.wf.psi)

        # Step 3: Apply kinetic operator
        psi_k *= self.exp_K

        # Step 4: Transform back to position space
        self.wf.psi = np.fft.ifft(psi_k)

        # Step 5: Apply second half-step potential
        self.wf.psi *= self.exp_V_half

    def evolve(self, t_final: float, n_snapshots: int = 100) -> List[WaveFunction1D]:
        """
        Evolve the wave function to final time and collect snapshots.

        Args:
            t_final: Final time
            n_snapshots: Number of snapshots to save

        Returns:
            List of WaveFunction1D snapshots at regular intervals
        """
        n_steps = int(t_final / self.dt)
        snapshot_interval = max(1, n_steps // n_snapshots)

        snapshots = [self.wf.copy()]
        times = [0.0]

        for step in range(n_steps):
            self.step()

            if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
                snapshots.append(self.wf.copy())
                times.append((step + 1) * self.dt)

        return snapshots, times

    def get_current_state(self) -> WaveFunction1D:
        """Return the current wave function state."""
        return self.wf.copy()


class SplitOperatorSolver2D:
    """
    Solves the 2D time-dependent Schrödinger equation using the split-operator method.

    Extends the 1D method to 2D using 2D FFTs for the kinetic operator.
    """

    def __init__(self, wavefunction: WaveFunction2D,
                 potential: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 dt: float = 0.01):
        """
        Initialize the 2D solver.

        Args:
            wavefunction: Initial 2D wave function
            potential: Function V(x, y) that returns 2D potential energy
            dt: Time step size
        """
        self.wf = wavefunction.copy()

        # Compute potential on 2D grid
        X, Y = np.meshgrid(self.wf.x, self.wf.y, indexing='ij')
        self.V = potential(X, Y)
        self.dt = dt

        # Pre-compute time evolution operators
        self._setup_operators()

    def _setup_operators(self) -> None:
        """Pre-compute the time evolution operators for efficiency."""
        # Potential operator (applied in position space)
        self.exp_V_half = np.exp(-1j * self.V * self.dt / (2 * self.wf.hbar))

        # Kinetic operator (applied in momentum space)
        # K = ℏ²(kx² + ky²)/(2m)
        KX, KY = np.meshgrid(self.wf.kx, self.wf.ky, indexing='ij')
        K = self.wf.hbar**2 * (KX**2 + KY**2) / (2 * self.wf.m)
        self.exp_K = np.exp(-1j * K * self.dt / self.wf.hbar)

    def step(self) -> None:
        """
        Advance the 2D wave function by one time step.

        Uses 2D FFT for kinetic operator application.
        """
        # Step 1: Apply half-step potential
        self.wf.psi *= self.exp_V_half

        # Step 2: Transform to momentum space (2D FFT)
        psi_k = np.fft.fft2(self.wf.psi)

        # Step 3: Apply kinetic operator
        psi_k *= self.exp_K

        # Step 4: Transform back to position space
        self.wf.psi = np.fft.ifft2(psi_k)

        # Step 5: Apply second half-step potential
        self.wf.psi *= self.exp_V_half

    def evolve(self, t_final: float, n_snapshots: int = 100) -> tuple:
        """
        Evolve the 2D wave function to final time.

        Args:
            t_final: Final time
            n_snapshots: Number of snapshots to save

        Returns:
            snapshots: List of WaveFunction2D snapshots
            times: List of corresponding times
        """
        n_steps = int(t_final / self.dt)
        snapshot_interval = max(1, n_steps // n_snapshots)

        snapshots = [self.wf.copy()]
        times = [0.0]

        for step in range(n_steps):
            self.step()

            if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
                snapshots.append(self.wf.copy())
                times.append((step + 1) * self.dt)

        return snapshots, times

    def get_current_state(self) -> WaveFunction2D:
        """Return the current wave function state."""
        return self.wf.copy()


def free_particle_potential_1d(x: np.ndarray) -> np.ndarray:
    """Free particle: V(x) = 0 everywhere"""
    return np.zeros_like(x)


def harmonic_oscillator_1d(x: np.ndarray, omega: float = 1.0, m: float = 1.0) -> np.ndarray:
    """
    Harmonic oscillator potential: V(x) = (1/2) m ω² x²

    Args:
        x: Position array
        omega: Angular frequency
        m: Mass

    Returns:
        Potential energy array
    """
    return 0.5 * m * omega**2 * x**2


def barrier_potential_1d(x: np.ndarray, barrier_height: float = 10.0,
                        barrier_position: float = 0.0,
                        barrier_width: float = 0.5) -> np.ndarray:
    """
    Rectangular potential barrier.

    Args:
        x: Position array
        barrier_height: Height of barrier
        barrier_position: Center of barrier
        barrier_width: Width of barrier

    Returns:
        Potential energy array
    """
    V = np.zeros_like(x)
    mask = np.abs(x - barrier_position) < barrier_width / 2
    V[mask] = barrier_height
    return V


def double_slit_potential_2d(X: np.ndarray, Y: np.ndarray,
                             slit_separation: float = 2.0,
                             slit_width: float = 0.4,
                             wall_position: float = 0.0,
                             wall_thickness: float = 0.1,
                             wall_height: float = 1e10) -> np.ndarray:
    """
    Double-slit barrier potential.

    Creates an infinite barrier at y=wall_position with two slits.

    Args:
        X, Y: 2D position meshgrids
        slit_separation: Distance between slit centers
        slit_width: Width of each slit
        wall_position: y-coordinate of the barrier wall
        wall_thickness: Thickness of the wall in y-direction
        wall_height: Barrier height (use large value for "infinite" wall)

    Returns:
        2D potential energy array
    """
    V = np.zeros_like(X)

    # Create wall region
    wall_mask = np.abs(Y - wall_position) < wall_thickness / 2

    # Define slit positions
    slit1_center = slit_separation / 2
    slit2_center = -slit_separation / 2

    # Mask for the two slits (regions where barrier is NOT present)
    slit1_mask = np.abs(X - slit1_center) < slit_width / 2
    slit2_mask = np.abs(X - slit2_center) < slit_width / 2
    slits_mask = slit1_mask | slit2_mask

    # Apply barrier everywhere on wall except at slits
    barrier_mask = wall_mask & ~slits_mask
    V[barrier_mask] = wall_height

    return V
