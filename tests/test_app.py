"""The demo's warning for numbers that are not in the input."""

from app.app import resolve_model_dir, unsupported_numbers_note

import pytest

SOURCE = "We included 4 studies (171 patients; 95% CI 1.2 to 2.3, 10,056 women)."


def test_all_numbers_supported():
    note = unsupported_numbers_note(SOURCE, "Four studies with 171 patients and 10056 women.")
    assert note.startswith("✅")


def test_lists_invented_numbers_once_in_order():
    note = unsupported_numbers_note(SOURCE, "27 studies, 78 patients, 27 again, 171 in total.")
    assert note.startswith("⚠️")
    assert "`27`, `78`" in note and "171" not in note


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No model found"):
        resolve_model_dir(tmp_path)


def test_model_dir_accepts_best_subfolder(tmp_path):
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "model.safetensors").write_bytes(b"x")
    assert resolve_model_dir(tmp_path) == tmp_path / "best"
