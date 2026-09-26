"""Verify hash-pinned locks still satisfy direct runtime/dev requirements."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
LOCK_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;\\]+)")


def direct_requirements(path: Path, seen: set[Path] | None = None) -> list[Requirement]:
    """Read direct pins, following relative requirement includes."""
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)
    seen.add(path)

    requirements: list[Requirement] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            include = line.split(maxsplit=1)[1]
            requirements.extend(direct_requirements(path.parent / include, seen))
            continue
        if line.startswith("-"):
            raise ValueError(
                f"unsupported requirement directive in {path.name}:{number}"
            )
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            requirements.append(requirement)
    return requirements


def lock_versions(path: Path) -> dict[str, Version]:
    versions: dict[str, Version] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LOCK_PIN.match(line)
        if match is None:
            continue
        name, raw_version = match.groups()
        try:
            versions[canonicalize_name(name)] = Version(raw_version)
        except InvalidVersion as error:
            raise ValueError(
                f"invalid pinned version in {path.name}:{number}"
            ) from error
    return versions


def check_pair(source: Path, lock: Path) -> list[str]:
    """Return direct requirements missing from or excluded by a lock."""
    pins = lock_versions(lock)
    errors: list[str] = []
    for requirement in direct_requirements(source):
        name = canonicalize_name(requirement.name)
        version = pins.get(name)
        if version is None:
            errors.append(
                f"{lock.name} does not pin direct requirement {requirement.name}"
            )
        elif requirement.specifier and not requirement.specifier.contains(
            version, prereleases=True
        ):
            errors.append(
                f"{lock.name} pins {requirement.name}=={version}, which does not satisfy "
                f"{requirement.specifier} from {source.name}"
            )
    return errors


def check_all(root: Path = ROOT) -> list[str]:
    pairs = (
        (root / "requirements.txt", root / "requirements.lock"),
        (root / "requirements-dev.txt", root / "requirements-dev.lock"),
    )
    errors: list[str] = []
    for source, lock in pairs:
        try:
            errors.extend(check_pair(source, lock))
        except (OSError, ValueError) as error:
            errors.append(f"cannot verify {source.name} against {lock.name}: {error}")
    return errors


def main() -> int:
    errors = check_all()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print("Hash-pinned runtime and development locks match direct requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
