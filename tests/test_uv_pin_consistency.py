"""CI and the image must install from uv.lock with the same uv release.

Both paths run `uv export --locked` against the same lockfile, so a mismatch is
usually invisible: the exported pins come out identical and every check stays
green. It is still a pin nobody maintains. Dependabot's `docker` ecosystem
bumps the `ghcr.io/astral-sh/uv:<version>` stage in the Dockerfile (it parses
`FROM`, including a stage alias), and its `github-actions` ecosystem bumps the
`astral-sh/setup-uv` SHA. Neither updates a `with:` input, so CI's `version:`
string is the one pin no updater owns, and it drifts a release further behind
with every Dockerfile bump until the two installers genuinely disagree.

That is what happened on the first bump: #519 moved the image to 0.12.24 while
CI stayed on 0.11.12, and all nine checks passed. This test is what makes the
next one fail loudly instead.

`[tool.uv] required-version` in pyproject.toml would also collapse the two
pins into one, but it is the wrong tool here: Dependabot's own updater runs its
own uv build, and uv exits non-zero when the running version does not satisfy
the field, so an exact pin would break the weekly lockfile refresh. A range
would satisfy the field while still allowing the two pins to differ, which is
the bug this guards against.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# `FROM ghcr.io/astral-sh/uv:<version>` — optionally `AS <stage>`.
_UV_IMAGE_RE = re.compile(
    r"^\s*FROM\s+ghcr\.io/astral-sh/uv:(?P<version>[^\s]+)", re.MULTILINE
)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _dockerfile_uv_version() -> str:
    matches = _UV_IMAGE_RE.findall(DOCKERFILE.read_text(encoding="utf-8"))
    assert matches, (
        "no `FROM ghcr.io/astral-sh/uv:<version>` line in the Dockerfile. If the "
        "image stage was renamed or replaced, update this test with it — do not "
        "delete it, or CI's uv pin goes back to being unwatched."
    )
    assert len(set(matches)) == 1, (
        f"the Dockerfile references more than one uv version: {sorted(set(matches))}"
    )
    return matches[0]


def _ci_setup_uv_version() -> str:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    pinned: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps") or []:
            uses = step.get("uses") or ""
            if uses.startswith("astral-sh/setup-uv@"):
                # An absent `version:` means setup-uv resolves whatever it
                # considers latest, which is exactly the floating pin this
                # whole arrangement exists to avoid.
                version = (step.get("with") or {}).get("version")
                assert version, (
                    "the astral-sh/setup-uv step declares no `version:`, so CI "
                    "would install whichever uv is latest that day and the "
                    "lockfile gate would stop being reproducible."
                )
                pinned.append(str(version))
    assert pinned, "no astral-sh/setup-uv step found in .github/workflows/ci.yml"
    assert len(set(pinned)) == 1, (
        f"ci.yml pins more than one uv version: {sorted(set(pinned))}"
    )
    return pinned[0]


def test_ci_and_image_pin_the_same_uv_release() -> None:
    dockerfile_version = _dockerfile_uv_version()
    ci_version = _ci_setup_uv_version()
    assert dockerfile_version == ci_version, (
        f"uv pin drift: the Dockerfile builds with uv {dockerfile_version} but "
        f"CI installs uv {ci_version}.\n\n"
        "Both run `uv export --locked` against the same uv.lock, so the exported "
        "pins may well be identical today and nothing else will fail. Align them "
        "anyway: set `version:` in the astral-sh/setup-uv step of "
        ".github/workflows/ci.yml to match the `FROM ghcr.io/astral-sh/uv:` stage "
        "in the Dockerfile.\n\n"
        "Dependabot bumps the Dockerfile stage but never a workflow `with:` "
        "input, so this drift is the expected outcome of an un-followed uv bump."
    )


def test_both_uv_pins_are_exact_versions() -> None:
    # A tag like `latest` or `0.12` would build with whatever that resolves to
    # on the day, which defeats pinning the installer to the lockfile.
    for label, version in (
        ("Dockerfile", _dockerfile_uv_version()),
        ("ci.yml", _ci_setup_uv_version()),
    ):
        assert _SEMVER_RE.match(version), (
            f"{label} pins uv as {version!r}, which is not an exact X.Y.Z "
            "version. A floating tag means the installer behind the lockfile "
            "gate can change without any commit."
        )
