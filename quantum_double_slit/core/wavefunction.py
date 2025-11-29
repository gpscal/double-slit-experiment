"""
Wave function representations for quantum states.

Provides classes for 1D and 2D wave functions with methods for
normalization, expectation values, and Fourier transforms.
"""

import numpy as np
from typing import Optional, Tuple


class WaveFunction1D:
    """
    Represents a 1D quantum wave function ψ(x, t).

    Attributes:
        x: Spatial grid points
        psi: Complex wave function values
        dx: Spatial grid spacing
        hbar: Reduced Planck constant (default: 1.0 in atomic units)
        m: Particle mass (default: 1.0 in atomic units)
    """

    def __init__(self, x: np.ndarray, psi: np.ndarray,
                 hbar: float = 1.0, m: float = 1.0):
        """
        Initialize wave function.

        Args:
            x: 1D array of spatial coordinates
            psi: 1D complex array of wave function values
            hbar: Reduced Planck constant
            m: Particle mass
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.psi = np.asarray(psi, dtype=np.complex128)
        self.hbar = hbar
        self.m = m

        if len(self.x) != len(self.psi):
            raise ValueError("x and psi must have same length")

        self.dx = self.x[1] - self.x[0]
        self.N = len(self.x)

        # Momentum space grid (for FFT)
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)

    @classmethod
    def gaussian(cls, x: np.ndarray, x0: float = 0.0, sigma: float = 1.0,
                 k0: float = 0.0, hbar: float = 1.0, m: float = 1.0) -> 'WaveFunction1D':
        """
        Create a Gaussian wave packet.

        ψ(x) = (2πσ²)^(-1/4) exp[-(x-x0)²/(4σ²)] exp[ik0(x-x0)]

        Args:
            x: Spatial grid
            x0: Center position
            sigma: Spatial width (standard deviation)
            k0: Initial momentum (k = p/ℏ)
            hbar: Reduced Planck constant
            m: Particle mass

        Returns:
            Normalized WaveFunction1D instance
        """
        # Normalization factor for Gaussian
        norm = (2 * np.pi * sigma**2) ** (-0.25)

        # Gaussian envelope with plane wave
        psi = norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * (x - x0))

        wf = cls(x, psi, hbar=hbar, m=m)
        wf.normalize()
        return wf

    def normalize(self) -> None:
        """Normalize the wave function: ∫|ψ|² dx = 1"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        if norm > 0:
            self.psi /= norm

    @property
    def norm(self) -> float:
        """Calculate norm: ∫|ψ|² dx"""
        return np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)

    @property
    def probability_density(self) -> np.ndarray:
        """Return probability density |ψ(x)|²"""
        return np.abs(self.psi)**2

    def to_momentum_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fourier transform to momentum space.

        Returns:
            k: Momentum grid
            psi_k: Wave function in momentum space
        """
        psi_k = np.fft.fft(self.psi) * self.dx / np.sqrt(2 * np.pi)
        return self.k, psi_k

    def expectation_position(self) -> float:
        """Calculate expectation value ⟨x⟩ = ∫ ψ* x ψ dx"""
        return np.sum(np.conj(self.psi) * self.x * self.psi).real * self.dx

    def expectation_momentum(self) -> float:
        """Calculate expectation value ⟨p⟩ = ℏ⟨k⟩"""
        k, psi_k = self.to_momentum_space()
        p_k = np.sum(np.conj(psi_k) * k * psi_k).real * (k[1] - k[0])
        return self.hbar * p_k

    def uncertainty_position(self) -> float:
        """Calculate position uncertainty Δx = √(⟨x²⟩ - ⟨x⟩²)"""
        x_mean = self.expectation_position()
        x2_mean = np.sum(np.conj(self.psi) * self.x**2 * self.psi).real * self.dx
        return np.sqrt(x2_mean - x_mean**2)

    def uncertainty_momentum(self) -> float:
        """Calculate momentum uncertainty Δp"""
        k, psi_k = self.to_momentum_space()
        dk = k[1] - k[0]

        k_mean = np.sum(np.conj(psi_k) * k * psi_k).real * dk
        k2_mean = np.sum(np.conj(psi_k) * k**2 * psi_k).real * dk

        delta_k = np.sqrt(k2_mean - k_mean**2)
        return self.hbar * delta_k

    def copy(self) -> 'WaveFunction1D':
        """Create a deep copy of this wave function"""
        return WaveFunction1D(self.x.copy(), self.psi.copy(),
                             self.hbar, self.m)


class WaveFunction2D:
    """
    Represents a 2D quantum wave function ψ(x, y, t).

    Attributes:
        x: x-coordinate grid points
        y: y-coordinate grid points
        psi: Complex wave function values (2D array)
        dx, dy: Spatial grid spacing
        hbar: Reduced Planck constant
        m: Particle mass
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, psi: np.ndarray,
                 hbar: float = 1.0, m: float = 1.0):
        """
        Initialize 2D wave function.

        Args:
            x: 1D array of x coordinates
            y: 1D array of y coordinates
            psi: 2D complex array of wave function values (shape: len(x) × len(y))
            hbar: Reduced Planck constant
            m: Particle mass
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.psi = np.asarray(psi, dtype=np.complex128)
        self.hbar = hbar
        self.m = m

        if self.psi.shape != (len(self.x), len(self.y)):
            raise ValueError(f"psi shape {self.psi.shape} must match (len(x), len(y)) = ({len(self.x)}, {len(self.y)})")

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Nx = len(self.x)
        self.Ny = len(self.y)

        # Momentum space grids
        self.kx = 2 * np.pi * np.fft.fftfreq(self.Nx, self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.Ny, self.dy)

    @classmethod
    def gaussian(cls, x: np.ndarray, y: np.ndarray,
                 x0: float = 0.0, y0: float = 0.0,
                 sigma_x: float = 1.0, sigma_y: float = 1.0,
                 kx0: float = 0.0, ky0: float = 0.0,
                 hbar: float = 1.0, m: float = 1.0) -> 'WaveFunction2D':
        """
        Create a 2D Gaussian wave packet.

        Args:
            x, y: Spatial grids
            x0, y0: Center position
            sigma_x, sigma_y: Spatial widths
            kx0, ky0: Initial momenta
            hbar: Reduced Planck constant
            m: Particle mass

        Returns:
            Normalized WaveFunction2D instance
        """
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Normalization
        norm = (2 * np.pi * sigma_x * sigma_y) ** (-0.5)

        # Gaussian envelope with plane wave
        psi = norm * np.exp(-((X - x0)**2 / (4 * sigma_x**2) +
                             (Y - y0)**2 / (4 * sigma_y**2)), dtype=np.complex128)
        psi *= np.exp(1j * (kx0 * (X - x0) + ky0 * (Y - y0)))

        wf = cls(x, y, psi, hbar=hbar, m=m)
        wf.normalize()
        return wf

    def normalize(self) -> None:
        """Normalize the wave function: ∫∫|ψ|² dxdy = 1"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)
        if norm > 0:
            self.psi /= norm

    @property
    def norm(self) -> float:
        """Calculate norm: ∫∫|ψ|² dxdy"""
        return np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)

    @property
    def probability_density(self) -> np.ndarray:
        """Return probability density |ψ(x,y)|²"""
        return np.abs(self.psi)**2

    def to_momentum_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D Fourier transform to momentum space.

        Returns:
            kx: x-momentum grid
            ky: y-momentum grid
            psi_k: Wave function in momentum space
        """
        psi_k = np.fft.fft2(self.psi) * self.dx * self.dy / (2 * np.pi)
        return self.kx, self.ky, psi_k

    def copy(self) -> 'WaveFunction2D':
        """Create a deep copy of this wave function"""
        return WaveFunction2D(self.x.copy(), self.y.copy(), self.psi.copy(),
                             self.hbar, self.m)
