import os
import shutil

import pytest

from harness import integrity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_kit_and_harness_manifests_currently_match():
    # sanity check that the checked-in manifests match the checked-in files
    # (this is what scripts/*.py rely on before a real run) -- if this fails,
    # someone edited kit/ or a HARNESS_CRITICAL_FILES entry without
    # regenerating the manifest.
    assert integrity.verify_kit() is True
    assert integrity.verify_harness() is True


def test_tampered_kit_file_is_detected(tmp_path):
    victim = os.path.join(ROOT, "kit", "evaluate.py")
    backup = tmp_path / "evaluate.py.bak"
    shutil.copy(victim, backup)
    try:
        with open(victim, "a", encoding="utf-8") as fh:
            fh.write("\n# tampered\n")
        with pytest.raises(RuntimeError, match="TAMPER"):
            integrity.verify_kit()
    finally:
        shutil.copy(backup, victim)
    assert integrity.verify_kit() is True  # restored


def test_tampered_harness_file_is_detected(tmp_path):
    victim = os.path.join(ROOT, "harness", "guards.py")
    backup = tmp_path / "guards.py.bak"
    shutil.copy(victim, backup)
    try:
        with open(victim, "a", encoding="utf-8") as fh:
            fh.write("\n# tampered\n")
        with pytest.raises(RuntimeError, match="TAMPER"):
            integrity.verify_harness()
    finally:
        shutil.copy(backup, victim)
    assert integrity.verify_harness() is True  # restored


def test_missing_kit_file_is_detected(tmp_path):
    victim = os.path.join(ROOT, "kit", "submit.py")
    backup = tmp_path / "submit.py.bak"
    shutil.move(victim, backup)
    try:
        with pytest.raises(RuntimeError, match="TAMPER"):
            integrity.verify_kit()
    finally:
        shutil.move(str(backup), victim)
    assert integrity.verify_kit() is True  # restored
