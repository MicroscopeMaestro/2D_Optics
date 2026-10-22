"""
Ray Representation and State Vector.
"""
from typing import Tuple, List, Dict, Any, Union, Optional
import math
import numpy as np


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=float)


def _norm3(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _default_orthogonal(v: np.ndarray) -> np.ndarray:
    ortho = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    s = _cross3(v, ortho)
    norm = _norm3(s)
    return s / norm if norm > 0 else np.array([1.0, 0.0, 0.0])


class Ray:
    def __init__(self, pos: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
                 dir: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 1.0),
                 wavelength: float = 0.5876,
                 intensity: float = 1.0,
                 jones_vector: Optional[Union[List[complex], np.ndarray]] = None,
                 phase: float = 0.0):
        self.pos = pos if isinstance(pos, np.ndarray) and pos.dtype == float else np.asarray(pos, dtype=float)
        dx, dy, dz = float(dir[0]), float(dir[1]), float(dir[2])
        norm_sq = dx * dx + dy * dy + dz * dz
        if abs(norm_sq - 1.0) < 1e-12:
            self.dir = dir if isinstance(dir, np.ndarray) and dir.dtype == float else np.array([dx, dy, dz], dtype=float)
        elif norm_sq > 0:
            norm = math.sqrt(norm_sq)
            self.dir = np.array([dx / norm, dy / norm, dz / norm], dtype=float)
        else:
            self.dir = np.array([0.0, 0.0, 1.0], dtype=float)
        self.wavelength = float(wavelength)
        self.intensity = float(intensity)
        self.path_length: float = 0.0
        self.phase: float = float(phase)
        self.tir: bool = False
        self.vignetted: bool = False
        self.interference: bool = False
        self.history: List[Dict[str, Any]] = []

        # Polarization state & local basis vectors
        if jones_vector is not None:
            self.jones_vector = np.asarray(jones_vector, dtype=complex)
        else:
            # Default linearly polarized state normalized to sqrt(intensity)
            self.jones_vector = np.array([math.sqrt(self.intensity) + 0j, 0.0 + 0j], dtype=complex)

        vx, vy, vz = float(self.dir[0]), float(self.dir[1]), float(self.dir[2])
        if abs(vx) < 0.9:
            sx, sy, sz = 0.0, vz, -vy
        else:
            sx, sy, sz = -vz, 0.0, vx
        s_norm = math.sqrt(sx * sx + sy * sy + sz * sz)
        if s_norm > 0:
            self.s_basis = np.array([sx / s_norm, sy / s_norm, sz / s_norm], dtype=float)
        else:
            self.s_basis = np.array([1.0, 0.0, 0.0], dtype=float)

        px = self.s_basis[1] * vz - self.s_basis[2] * vy
        py = self.s_basis[2] * vx - self.s_basis[0] * vz
        pz = self.s_basis[0] * vy - self.s_basis[1] * vx
        p_norm = math.sqrt(px * px + py * py + pz * pz)
        if p_norm > 0:
            self.p_basis = np.array([px / p_norm, py / p_norm, pz / p_norm], dtype=float)
        else:
            self.p_basis = np.array([0.0, 1.0, 0.0], dtype=float)

    def update_intensity(self) -> float:
        """Update intensity from Jones vector norm squared: I = |Es|^2 + |Ep|^2."""
        jv = self.jones_vector
        self.intensity = float(jv[0].real*jv[0].real + jv[0].imag*jv[0].imag + jv[1].real*jv[1].real + jv[1].imag*jv[1].imag)
        return self.intensity

    def copy(self) -> 'Ray':
        r = Ray.__new__(Ray)
        r.pos = self.pos.copy()
        r.dir = self.dir.copy()
        r.wavelength = self.wavelength
        r.intensity = self.intensity
        r.path_length = self.path_length
        r.phase = self.phase
        r.tir = self.tir
        r.vignetted = self.vignetted
        r.interference = self.interference
        r.history = [h.copy() for h in self.history]
        r.jones_vector = self.jones_vector.copy()
        r.s_basis = self.s_basis.copy()
        r.p_basis = self.p_basis.copy()
        for k, v in self.__dict__.items():
            if k not in r.__dict__:
                setattr(r, k, v)
        return r


class RayBundle:
    """
    Vectorized ray bundle leveraging (N, 3) NumPy arrays for ray positions,
    directions, wavelengths, intensity, phase, and state flags.
    """
    def __init__(
        self,
        pos: Union[np.ndarray, List[Any]],
        dir: Union[np.ndarray, List[Any]],
        wavelength: Union[float, np.ndarray, List[float]] = 0.5876,
        intensity: Union[float, np.ndarray, List[float]] = 1.0,
        phase: Union[float, np.ndarray, List[float]] = 0.0,
        vignetted: Optional[np.ndarray] = None,
        tir: Optional[np.ndarray] = None,
        path_length: Optional[np.ndarray] = None,
        launch_pos: Optional[np.ndarray] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        p = np.asarray(pos, dtype=float)
        if p.ndim == 1 and p.size == 3:
            p = p.reshape(1, 3)
        elif p.ndim == 0 or p.size == 0:
            p = np.empty((0, 3), dtype=float)
        self.pos = p

        n = len(self.pos)
        d = np.asarray(dir, dtype=float)
        if d.ndim == 1 and d.size == 3:
            d = d.reshape(1, 3)
        elif d.ndim == 0 or d.size == 0:
            d = np.empty((0, 3), dtype=float)

        if len(d) > 0:
            d_norms = np.linalg.norm(d, axis=1, keepdims=True)
            d_norms[d_norms == 0.0] = 1.0
            self.dir = d / d_norms
        else:
            self.dir = d

        if np.isscalar(wavelength):
            self.wavelength = np.full(n, float(wavelength), dtype=float)
        else:
            wl = np.asarray(wavelength, dtype=float)
            self.wavelength = wl if wl.shape == (n,) else np.full(n, 0.5876, dtype=float)

        if np.isscalar(intensity):
            self.intensity = np.full(n, float(intensity), dtype=float)
        else:
            inte = np.asarray(intensity, dtype=float)
            self.intensity = inte if inte.shape == (n,) else np.full(n, 1.0, dtype=float)

        if np.isscalar(phase):
            self.phase = np.full(n, float(phase), dtype=float)
        else:
            ph = np.asarray(phase, dtype=float)
            self.phase = ph if ph.shape == (n,) else np.full(n, 0.0, dtype=float)

        self.vignetted = np.asarray(vignetted, dtype=bool) if vignetted is not None else np.zeros(n, dtype=bool)
        self.tir = np.asarray(tir, dtype=bool) if tir is not None else np.zeros(n, dtype=bool)
        self.path_length = np.asarray(path_length, dtype=float) if path_length is not None else np.zeros(n, dtype=float)
        self.launch_pos = np.asarray(launch_pos, dtype=float) if launch_pos is not None else self.pos.copy()
        self.history: List[Dict[str, Any]] = history if history is not None else []

    def __len__(self) -> int:
        return len(self.pos)

    def copy(self) -> 'RayBundle':
        b = RayBundle.__new__(RayBundle)
        b.pos = self.pos.copy()
        b.dir = self.dir.copy()
        b.wavelength = self.wavelength.copy()
        b.intensity = self.intensity.copy()
        b.phase = self.phase.copy()
        b.vignetted = self.vignetted.copy()
        b.tir = self.tir.copy()
        b.path_length = self.path_length.copy()
        b.launch_pos = self.launch_pos.copy()
        b.history = [dict(h) for h in self.history] if self.history else []
        return b

    @classmethod
    def from_rays(cls, rays: List[Ray]) -> 'RayBundle':
        if not rays:
            return cls(np.empty((0, 3)), np.empty((0, 3)))
        pos = np.array([r.pos for r in rays], dtype=float)
        direction = np.array([r.dir for r in rays], dtype=float)
        wavelength = np.array([r.wavelength for r in rays], dtype=float)
        intensity = np.array([r.intensity for r in rays], dtype=float)
        phase = np.array([r.phase for r in rays], dtype=float)
        vignetted = np.array([getattr(r, "vignetted", False) for r in rays], dtype=bool)
        tir = np.array([getattr(r, "tir", False) for r in rays], dtype=bool)
        path_length = np.array([getattr(r, "path_length", 0.0) for r in rays], dtype=float)
        launch_pos = np.array([getattr(r, "launch_pos", r.pos) for r in rays], dtype=float)
        bundle = cls(
            pos=pos,
            dir=direction,
            wavelength=wavelength,
            intensity=intensity,
            phase=phase,
            vignetted=vignetted,
            tir=tir,
            path_length=path_length,
            launch_pos=launch_pos,
        )
        return bundle

    def to_rays(self) -> List[Ray]:
        N = len(self)
        rays: List[Ray] = []
        has_history = bool(self.history)
        for i in range(N):
            r = Ray(
                pos=self.pos[i].copy(),
                dir=self.dir[i].copy(),
                wavelength=float(self.wavelength[i]),
                intensity=float(self.intensity[i]),
                phase=float(self.phase[i]),
            )
            r.vignetted = bool(self.vignetted[i])
            r.tir = bool(self.tir[i])
            r.path_length = float(self.path_length[i])
            r.launch_pos = self.launch_pos[i].copy()
            if has_history:
                ray_hist = []
                for step in self.history:
                    h_item = {}
                    for k, v in step.items():
                        if isinstance(v, np.ndarray) and len(v) == N:
                            h_item[k] = v[i].copy() if hasattr(v[i], "copy") else v[i]
                        else:
                            h_item[k] = v
                    ray_hist.append(h_item)
                r.history = ray_hist
            rays.append(r)
        return rays

