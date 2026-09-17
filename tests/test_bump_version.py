"""Regression tests for the version bump script."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(bump_version.__doc__.strip(), result.stdout)


class BumpVersionCheckTests(unittest.TestCase):
    def test_check_passes_on_the_tree_as_committed(self):
        # The drift guard is only a signal while it is green on a clean
        # checkout; a target whose anchor rots makes it permanently red.
        self.assertEqual(bump_version.cmd_check(), 0)


class BumpVersionAtomicityTests(unittest.TestCase):
    def _sandbox(self, readme: str) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        (root / "pyproject.toml").write_text(
            '[project]\nname = "sandbox"\nversion = "1.0.0"\n', encoding="utf-8"
        )
        (root / "README.md").write_text(readme, encoding="utf-8")
        return root

    @staticmethod
    def _targets(root: Path) -> list[bump_version.Target]:
        return [
            bump_version.Target(
                path=root / "README.md",
                pattern=r"badge v(?P<version>\S+)\.",
                template="badge v{version}.",
            ),
            bump_version.Target(
                path=root / "README.md",
                pattern=r"prose \(v(?P<version>[^)]+)\)\.",
                template="prose (v{version}).",
            ),
        ]

    def _patched(self, root: Path, targets: list[bump_version.Target]):
        return mock.patch.multiple(
            bump_version,
            ROOT=root,
            PYPROJECT=root / "pyproject.toml",
            all_targets=lambda: targets,
        )

    def test_bump_updates_every_anchor_including_two_in_one_file(self):
        root = self._sandbox("badge v1.0.0.\n\nprose (v1.0.0).\n")

        with self._patched(root, self._targets(root)):
            self.assertEqual(bump_version.cmd_bump("1.0.1"), 0)

        self.assertIn('version = "1.0.1"', (root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            (root / "README.md").read_text(encoding="utf-8"),
            "badge v1.0.1.\n\nprose (v1.0.1).\n",
        )

    def test_missing_anchor_aborts_the_bump_without_writing_anything(self):
        root = self._sandbox("badge v1.0.0.\n")  # second anchor deliberately absent
        before = {
            path: path.read_text(encoding="utf-8") for path in sorted(root.iterdir())
        }

        with self._patched(root, self._targets(root)):
            self.assertEqual(bump_version.cmd_bump("1.0.1"), 1)

        for path, text in before.items():
            self.assertEqual(path.read_text(encoding="utf-8"), text, path.name)


if __name__ == "__main__":
    unittest.main()
