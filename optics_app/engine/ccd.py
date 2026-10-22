"""
2D CCD Sensor Intensity Accumulation and Pixel Matrix Histogramming.
"""
from typing import Optional, Union, List
import numpy as np

class CCDSensor:
    def __init__(self, width_mm: float = 10.0, height_mm: float = 10.0, pixel_cols: int = 100, pixel_rows: int = 100):
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.pixel_cols = int(pixel_cols)
        self.pixel_rows = int(pixel_rows)
        self.pixel_pitch_x = self.width_mm / self.pixel_cols if self.pixel_cols > 0 else 0.0
        self.pixel_pitch_y = self.height_mm / self.pixel_rows if self.pixel_rows > 0 else 0.0
        self.matrix = np.zeros((self.pixel_rows, self.pixel_cols), dtype=float)
        self.total_rays_deposited = 0
        self.out_of_bounds_rays = 0

    def clear(self) -> None:
        self.matrix.fill(0.0)
        self.total_rays_deposited = 0
        self.out_of_bounds_rays = 0

    def deposit_ray(self, x_mm: float, y_mm: float, intensity: float = 1.0) -> None:
        col = int(np.floor((x_mm + self.width_mm / 2.0) / self.pixel_pitch_x))
        row = int(np.floor((y_mm + self.height_mm / 2.0) / self.pixel_pitch_y))

        if 0 <= col < self.pixel_cols and 0 <= row < self.pixel_rows:
            self.matrix[row, col] += intensity
            self.total_rays_deposited += 1
        else:
            self.out_of_bounds_rays += 1

    def deposit_rays(self, x_array: Union[List[float], np.ndarray], y_array: Union[List[float], np.ndarray], intensities: Optional[Union[List[float], np.ndarray]] = None) -> None:
        if intensities is None:
            intensities = np.ones_like(x_array)
        for x, y, I in zip(x_array, y_array, intensities):
            self.deposit_ray(float(x), float(y), float(I))

    def get_heatmap(self) -> np.ndarray:
        return self.matrix.copy()

    def get_total_intensity(self) -> float:
        return float(np.sum(self.matrix))

class CCDMatrix(CCDSensor):
    pass
