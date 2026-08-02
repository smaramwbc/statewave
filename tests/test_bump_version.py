"""Regression tests for the version bump script."""

import subprocess
import sys
import unittest
from pathlib import Path

from scripts import bump_version


class BumpVersionHelpTests(unittest.TestCase):
    def test_help_includes_complete_module_docstring(self):
        script = Path(bump_version.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(bump_version.__doc__.strip(), result.stdout)


if __name__ == "__main__":
    unittest.main()
