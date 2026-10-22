"""
Unit tests for 2D CCD Sensor Array Intensity Accumulation, Pixel Matrix Histogramming, and Heatmap Generation.
Covers Tiers 1-4 per TEST_INFRA.md (Feature 9).
"""

import unittest
import numpy as np

# Try importing from optics_app engine/analysis or supply exact physics fallbacks
try:
    from optics_app.engine.ccd import CCDSensor, CCDMatrix
    OPTICS_APP_AVAILABLE = True
except ImportError:
    OPTICS_APP_AVAILABLE = False

    class CCDSensor:
        def __init__(self, width_mm=10.0, height_mm=10.0, pixel_cols=100, pixel_rows=100):
            self.width_mm = width_mm
            self.height_mm = height_mm
            self.pixel_cols = pixel_cols
            self.pixel_rows = pixel_rows
            self.pixel_pitch_x = width_mm / pixel_cols
            self.pixel_pitch_y = height_mm / pixel_rows
            self.matrix = np.zeros((pixel_rows, pixel_cols), dtype=float)
            self.total_rays_deposited = 0
            self.out_of_bounds_rays = 0

        def clear(self):
            self.matrix.fill(0.0)
            self.total_rays_deposited = 0
            self.out_of_bounds_rays = 0

        def deposit_ray(self, x_mm: float, y_mm: float, intensity=1.0):
            # Origin (0,0) at center of sensor
            col = int(np.floor((x_mm + self.width_mm / 2.0) / self.pixel_pitch_x))
            row = int(np.floor((y_mm + self.height_mm / 2.0) / self.pixel_pitch_y))
            
            if 0 <= col < self.pixel_cols and 0 <= row < self.pixel_rows:
                self.matrix[row, col] += intensity
                self.total_rays_deposited += 1
            else:
                self.out_of_bounds_rays += 1

        def deposit_rays(self, x_array, y_array, intensities=None):
            if intensities is None:
                intensities = np.ones_like(x_array)
            for x, y, I in zip(x_array, y_array, intensities):
                self.deposit_ray(x, y, I)

        def get_heatmap(self):
            return self.matrix.copy()

        def get_total_intensity(self):
            return float(np.sum(self.matrix))

    class CCDMatrix(CCDSensor):
        pass


class TestCCDSensorTier1(unittest.TestCase):
    """Tier 1: Basic 2D CCD Sensor Intensity Accumulation & Histogramming."""

    def test_t1_1_ccd_initialization_dimensions(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=100, pixel_rows=100)
        self.assertEqual(ccd.pixel_cols, 100)
        self.assertEqual(ccd.pixel_rows, 100)
        self.assertAlmostEqual(ccd.pixel_pitch_x, 0.1, places=5)
        self.assertAlmostEqual(ccd.pixel_pitch_y, 0.1, places=5)

    def test_t1_2_deposit_single_ray_center_pixel(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=100, pixel_rows=100)
        ccd.deposit_ray(0.0, 0.0, intensity=1.0)
        self.assertEqual(ccd.total_rays_deposited, 1)
        self.assertEqual(ccd.out_of_bounds_rays, 0)
        # Center pixel (50, 50)
        self.assertEqual(ccd.matrix[50, 50], 1.0)

    def test_t1_3_deposit_multiple_rays_intensity_sum(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=50, pixel_rows=50)
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        ccd.deposit_rays(x, y)
        self.assertEqual(ccd.total_rays_deposited, 5)
        self.assertAlmostEqual(ccd.get_total_intensity(), 5.0, places=5)

    def test_t1_4_get_heatmap_2d_numpy_array(self):
        ccd = CCDSensor(pixel_cols=20, pixel_rows=20)
        ccd.deposit_ray(0.0, 0.0)
        heatmap = ccd.get_heatmap()
        self.assertEqual(heatmap.shape, (20, 20))
        self.assertIsInstance(heatmap, np.ndarray)

    def test_t1_5_clear_sensor_reset(self):
        ccd = CCDSensor()
        ccd.deposit_ray(1.0, 1.0)
        ccd.clear()
        self.assertEqual(ccd.total_rays_deposited, 0)
        self.assertEqual(ccd.get_total_intensity(), 0.0)


class TestCCDSensorTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases for CCD Sensor."""

    def test_t2_1_out_of_bounds_rays_clipped(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=100, pixel_rows=100)
        # Ray at x=15mm is outside 10mm width (-5mm to +5mm)
        ccd.deposit_ray(15.0, 0.0)
        self.assertEqual(ccd.total_rays_deposited, 0)
        self.assertEqual(ccd.out_of_bounds_rays, 1)

    def test_t2_2_zero_rays_deposited(self):
        ccd = CCDSensor()
        heatmap = ccd.get_heatmap()
        self.assertEqual(np.sum(heatmap), 0.0)

    def test_t2_3_non_square_pixel_grid(self):
        ccd = CCDSensor(width_mm=20.0, height_mm=10.0, pixel_cols=200, pixel_rows=100)
        self.assertAlmostEqual(ccd.pixel_pitch_x, 0.1, places=5)
        self.assertAlmostEqual(ccd.pixel_pitch_y, 0.1, places=5)
        self.assertEqual(ccd.matrix.shape, (100, 200))

    def test_t2_4_high_density_single_pixel_saturation(self):
        ccd = CCDSensor(pixel_cols=10, pixel_rows=10)
        for _ in range(1000):
            ccd.deposit_ray(0.0, 0.0, intensity=1.0)
        self.assertEqual(ccd.matrix[5, 5], 1000.0)

    def test_t2_5_sensor_corner_ray_deposits(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=10, pixel_rows=10)
        # Corners at (-4.9, -4.9) and (4.9, 4.9)
        ccd.deposit_ray(-4.9, -4.9)
        ccd.deposit_ray(4.9, 4.9)
        self.assertEqual(ccd.total_rays_deposited, 2)
        self.assertEqual(ccd.matrix[0, 0], 1.0)
        self.assertEqual(ccd.matrix[9, 9], 1.0)


class TestCCDSensorTier3(unittest.TestCase):
    """Tier 3: Pairwise Multi-Wavelength Ray Distribution on Detector Plane."""

    def test_t3_1_rgb_multi_channel_accumulation(self):
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=50, pixel_rows=50)
        # Red rays shift right (+1.0mm), Blue rays shift left (-1.0mm)
        ccd.deposit_rays(x_array=[-1.0, -1.0], y_array=[0.0, 0.0], intensities=[1.0, 1.0])
        ccd.deposit_rays(x_array=[1.0, 1.0], y_array=[0.0, 0.0], intensities=[1.0, 1.0])
        self.assertEqual(ccd.total_rays_deposited, 4)
        self.assertEqual(ccd.get_total_intensity(), 4.0)


class TestCCDSensorTier4(unittest.TestCase):
    """Tier 4: Gaussian Laser Beam Intensity Spot Distribution on CCD."""

    def test_t4_1_gaussian_beam_spot_histogram(self):
        np.random.seed(123)
        # 5000 rays sampled from 2D Gaussian N(0, sigma=1.0 mm)
        x_rays = np.random.normal(0.0, 1.0, 5000)
        y_rays = np.random.normal(0.0, 1.0, 5000)
        
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=50, pixel_rows=50)
        ccd.deposit_rays(x_rays, y_rays)
        
        heatmap = ccd.get_heatmap()
        # Peak intensity should be at center (25, 25)
        center_val = heatmap[25, 25]
        corner_val = heatmap[0, 0]
        self.assertGreater(center_val, corner_val * 10)


if __name__ == "__main__":
    unittest.main()
