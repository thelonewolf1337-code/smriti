"""Minimal test runner for environments without pytest.

Normally just run `pytest`. This runner exists so the suite can be
verified with zero installed dependencies: python tests/run_tests.py
"""

import inspect
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pytest  # noqa: F401
except ImportError:  # provide the tiny subset the suite uses
    shim = types.ModuleType("pytest")
    shim.fixture = lambda *a, **k: (lambda f: f)
    sys.modules["pytest"] = shim

import test_brain  # noqa: E402
import test_improvements  # noqa: E402
import test_smriti  # noqa: E402

from smriti import MemoryEngine  # noqa: E402


def main() -> int:
    tests = []
    for mod in (test_smriti, test_brain, test_improvements):
        tests += [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    passed, failed = 0, 0
    for name, fn in tests:
        eng = MemoryEngine(":memory:")
        try:
            kwargs = {"eng": eng} if "eng" in inspect.signature(fn).parameters else {}
            fn(**kwargs)
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
        finally:
            eng.close()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
