import pytest
import sys

if __name__ == "__main__":
    res = pytest.main(["tests/", "-v", "--tb=short"])
    sys.exit(res)
