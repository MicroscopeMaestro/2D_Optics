"""
Test suite for 2D YZ-Plane Refractive 1064 nm Spectrometer Layout.
Located in root tests directory for continuous integration and regression testing.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teamwork_yz_spectrometer.test_spectrometer_yz import TestYZSpectrometer

if __name__ == '__main__':
    unittest.main()
