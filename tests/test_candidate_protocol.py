"""Boundary checks for the latest source-scoped scale-load encoding."""

from pathlib import Path

import pytest

from mx_gemmini_support.candidate_protocol import scale_load_2d_operands


CONTRACTS = Path(__file__).resolve().parents[1] / "mx_gemmini_support/contracts"
CANDIDATE = (CONTRACTS / "software-spec-2029218-candidate.yaml").read_bytes()
OLDER = (CONTRACTS / "software-spec.yaml").read_bytes()


def test_candidate_scale_load_fields():
    rs1, rs2 = scale_load_2d_operands(
        CANDIDATE, source_address=0x1000, row_pitch=64,
        bytes_per_row=32, operand="weight", destination_offset=4096,
        row_count=2, gated=True,
    )
    assert rs1 == 0x1000 | (64 << 40)
    assert rs2 == 32 | (1 << 32) | (4096 << 33) | (2 << 46) | (1 << 54)


@pytest.mark.parametrize("change,reason", [
    ({"bytes_per_row": 12}, "8-byte multiple"),
    ({"source_address": 0x1004}, "source address"),
    ({"row_pitch": 12}, "row pitch"),
    ({"row_count": 256}, "row count"),
    ({"destination_offset": 4088, "bytes_per_row": 16}, "destination half"),
    ({"source_address": (1 << 40) - 8, "bytes_per_row": 16}, "source rows"),
])
def test_candidate_scale_load_rejects_truncation_and_wrap(change, reason):
    fields = dict(source_address=0x1000, row_pitch=0, bytes_per_row=32,
                  operand="activation", destination_offset=0, row_count=1)
    fields.update(change)
    with pytest.raises(ValueError, match=reason):
        scale_load_2d_operands(CANDIDATE, **fields)


def test_candidate_encoder_rejects_older_protocol():
    with pytest.raises(ValueError, match="exact 2029218"):
        scale_load_2d_operands(
            OLDER, source_address=0x1000, row_pitch=0,
            bytes_per_row=32, operand="activation", destination_offset=0,
        )
