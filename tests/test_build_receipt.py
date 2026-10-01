"""A declared Scala census must match the current selected source tree."""

import hashlib

import pytest

from mx_gemmini_support.build_receipt import _source_census


def _receipt_census(root):
    recorded = {}
    for path in sorted(root.rglob("*.scala")):
        recorded[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = hashlib.sha256("".join(
        f"{name} {digest}\n" for name, digest in recorded.items()
    ).encode()).hexdigest()
    return {"sha256_by_relative_path": recorded, "scala_files": len(recorded),
            "manifest_sha256": manifest}


def test_source_census_refuses_changed_and_extra_scala_files(tmp_path):
    source = tmp_path / "src/main/scala/gemmini/Config.scala"
    source.parent.mkdir(parents=True)
    source.write_text("object Config {}\n")
    mx_source = tmp_path / "mxgen/src/main/scala/mxgen/Mx.scala"
    mx_source.parent.mkdir(parents=True)
    mx_source.write_text("object Mx {}\n")
    census = _receipt_census(tmp_path)
    assert _source_census(tmp_path, census) == census["manifest_sha256"]
    mx_source.write_text("object Mx { val changed = true }\n")
    with pytest.raises(ValueError, match="Scala source differs"):
        _source_census(tmp_path, census)
    mx_source.write_text("object Mx {}\n")
    (source.parent / "Added.scala").write_text("object Added {}\n")
    with pytest.raises(ValueError, match="omits or adds"):
        _source_census(tmp_path, census)
