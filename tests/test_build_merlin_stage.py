"""Small stage-probe checks that do not need a bare-metal toolchain."""

from __future__ import annotations

import numpy as np
import pytest

from tools.build_merlin_stage import _can_hash_first_output


def test_first_output_digest_matches_merlin_supported_dtype(tmp_path) -> None:
    golden = tmp_path / "golden.npy"
    np.save(golden, np.zeros((2,), dtype=np.float32))
    assert _can_hash_first_output(tmp_path)
    np.save(golden, np.zeros((2,), dtype=np.bool_))
    assert not _can_hash_first_output(tmp_path)
    golden.unlink()
    with pytest.raises(ValueError, match="golden.npy"):
        _can_hash_first_output(tmp_path)
