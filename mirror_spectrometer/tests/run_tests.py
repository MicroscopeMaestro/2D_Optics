"""
Standalone CLI Test Runner for All-Mirror Spectrometer Layout.
Runs unittest/pytest with structured summary reporting across Tiers 1-4.
Usage:
    python mirror_spectrometer/tests/run_tests.py [path/to/layout.json]
    python mirror_spectrometer/tests/run_tests.py --fixture
"""

import sys
import os
import unittest
from pathlib import Path

# Add repo root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Handle CLI arguments
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "--fixture":
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "reference_all_mirror_spectrometer.json"
        os.environ["ALL_MIRROR_LAYOUT_PATH"] = str(fixture_path)
    elif Path(arg).is_file():
        os.environ["ALL_MIRROR_LAYOUT_PATH"] = str(Path(arg).resolve())

from mirror_spectrometer.tests.test_all_mirror_spectrometer import (
    TestTier1FeatureCoverage,
    TestTier2BoundaryAndCornerCases,
    TestTier3CrossFeatureAndClearances,
    TestTier4RealWorldWorkloadAndContainment,
    find_target_layout_path
)


def run_all_mirror_tests() -> int:
    print("=" * 80)
    print("  ALL-MIRROR SPECTROMETER E2E VERIFICATION TEST SUITE (TIERS 1 - 4)")
    print("=" * 80)

    try:
        layout_path = find_target_layout_path()
        print(f"[INFO] Target Layout Under Test: {layout_path}")
    except Exception as e:
        print(f"[ERROR] Layout discovery failed: {e}")
        return 1

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        ("Tier 1: Feature Coverage & COTS Compliance", TestTier1FeatureCoverage),
        ("Tier 2: Boundary & Corner Cases", TestTier2BoundaryAndCornerCases),
        ("Tier 3: Cross-Feature & Geometric Clearances", TestTier3CrossFeatureAndClearances),
        ("Tier 4: Real-World Workload & 100% Spot Containment", TestTier4RealWorldWorkloadAndContainment),
    ]

    for tier_name, test_cls in test_classes:
        sub_suite = loader.loadTestsFromTestCase(test_cls)
        print(f"[LOADED] {tier_name}: {sub_suite.countTestCases()} test cases")
        suite.addTests(sub_suite)

    print("-" * 80)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 80)

    if result.wasSuccessful():
        print(f"[PASS] ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"[FAIL] TEST RUN FAILED: {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_mirror_tests())
