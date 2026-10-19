"""Sharing packed storage cannot substitute different or partially bound bytes."""

import os

import pytest

from mx_gemmini_support.merlin_catalog_builder import _write_packed_data


def test_reused_storage_is_checked_against_fresh_quantization(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    original = previous / "weight_codes.bin"
    original.write_bytes(bytes(range(256)))
    selected = tmp_path / original.name
    _write_packed_data(selected, bytes(range(256)), previous)
    assert os.path.samefile(original, selected)
    assert selected.read_bytes() == bytes(range(256))


@pytest.mark.parametrize("case", ["missing", "different", "truncated", "symlink"])
def test_reuse_refuses_incorrect_storage_before_creating_destination(tmp_path, case):
    previous = tmp_path / "previous"
    previous.mkdir()
    original = previous / "weight_codes.bin"
    if case == "symlink":
        target = previous / "target"
        target.write_bytes(b"expected")
        original.symlink_to(target)
    elif case != "missing":
        original.write_bytes(b"diffbits" if case == "different" else b"short")
    selected = tmp_path / original.name
    with pytest.raises(ValueError, match="freshly computed bytes"):
        _write_packed_data(selected, b"expected", previous)
    assert not selected.exists()


def test_binding_rechecks_bytes_after_hardlink(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    original = previous / "weight_codes.bin"
    original.write_bytes(b"expected")
    link = os.link

    def mutate(source, destination):
        link(source, destination)
        original.write_bytes(b"mutation")

    monkeypatch.setattr(os, "link", mutate)
    with pytest.raises(ValueError, match="changed during binding"):
        _write_packed_data(tmp_path / original.name, b"expected", previous)


def test_reuse_does_not_overwrite_existing_destination(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "weight_codes.bin").write_bytes(b"expected")
    selected = tmp_path / "weight_codes.bin"
    selected.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _write_packed_data(selected, b"expected", previous)
    assert selected.read_bytes() == b"existing"
