import sys
import os
import time

print("Starting M3 Stress Validation Runner...", flush=True)
t0 = time.time()

import unittest
from tests.test_m3_challenger1_stress import (
    TestM3EmpiricalRapidOptimization,
    TestM3DynamicLDEInteractions,
    TestM3VariableTogglingAndModelSync,
    TestM3ZeroAndAllVariableOptimization,
    TestM3GUIStateConsistencyAndWavelengths,
    TestM3ExtremeBoundaryConditions,
)

print(f"Import time: {time.time() - t0:.2f}s", flush=True)

suite = unittest.TestSuite()
loader = unittest.TestLoader()

suite.addTests(loader.loadTestsFromTestCase(TestM3EmpiricalRapidOptimization))
suite.addTests(loader.loadTestsFromTestCase(TestM3DynamicLDEInteractions))
suite.addTests(loader.loadTestsFromTestCase(TestM3VariableTogglingAndModelSync))
suite.addTests(loader.loadTestsFromTestCase(TestM3ZeroAndAllVariableOptimization))
suite.addTests(loader.loadTestsFromTestCase(TestM3GUIStateConsistencyAndWavelengths))
suite.addTests(loader.loadTestsFromTestCase(TestM3ExtremeBoundaryConditions))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print(f"Total time: {time.time() - t0:.2f}s", flush=True)
print(f"Was successful: {result.wasSuccessful()}", flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
