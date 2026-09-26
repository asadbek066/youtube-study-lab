from pathlib import Path

from tools.check_requirement_locks import check_pair


def test_lock_check_follows_requirement_includes_and_accepts_matching_pins(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "requirements.txt"
    runtime.write_text("openai==3.19.2\n", encoding="utf-8")
    development = tmp_path / "requirements-dev.txt"
    development.write_text("-r requirements.txt\nruff==0.16.9\n", encoding="utf-8")
    lock = tmp_path / "requirements-dev.lock"
    lock.write_text("openai==3.19.2\nruff==0.16.9\n", encoding="utf-8")

    assert check_pair(development, lock) == []


def test_lock_check_reports_a_stale_direct_dependency_pin(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("openai==3.19.2\n", encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    lock.write_text("openai==3.5.0\n", encoding="utf-8")

    errors = check_pair(requirements, lock)

    assert len(errors) == 1
    assert "openai==3.5.0" in errors[0]
    assert "requirements.txt" in errors[0]
