"""Source-scoped encoders for the unreviewed 2029218 MX command candidate."""

from __future__ import annotations

import yaml

from .contract import compile_contract

RTL_COMMIT = "2029218197f771ce71416f859d975bea47b7aabc"
MXGEN_COMMIT = "56ef1c6810924e1cb0af07add09156b0e2f53576"
_HALF_BYTES = 4096
_FIELD_CONTRACT = {
    "upload_funct": 27,
    "source_address_bits": "rs1[39:0]",
    "source_row_pitch_bits": "rs1[63:40]",
    "row_bytes_bits": "rs2[31:0]",
    "operand_select_bit": "rs2[32]",
    "operand_select_values": {"activation": 0, "weight": 1},
    "destination_byte_offset_bits": "rs2[45:33]",
    "row_count_bits": "rs2[53:46]",
    "gated_load_bit": "rs2[54]",
    "source_address_alignment_bytes": 8,
    "source_row_pitch_alignment_bytes": 8,
    "destination_alignment_bytes": 8,
    "length_multiple_bytes": 8,
    "row_count_zero_means_one": True,
    "row_pitch_zero_means_contiguous": True,
    "active_buffer_window_bytes_per_operand": _HALF_BYTES,
}


def _selected_scale_fields(spec_bytes: bytes) -> dict:
    contract = compile_contract(spec_bytes)
    if (contract["rtl_commit"] != RTL_COMMIT or
            contract["mxgen_commit"] != MXGEN_COMMIT or
            contract["rtl_config"] != "GemminiMxFPConfigs.standaloneMxFPConfig" or
            contract["rtl_config_class"] != "GemminiMxFPStandaloneConfig"):
        raise ValueError("2-D MX scale loads require the exact 2029218 standalone contract")
    source = yaml.safe_load(spec_bytes)
    fields = source["transfer_contracts"]["scale_load"]["semantics"]
    if any(fields.get(key) != value for key, value in _FIELD_CONTRACT.items()):
        raise ValueError("2-D MX scale-load fields differ from the reviewed encoder assumptions")
    return fields


def scale_load_2d_operands(
    spec_bytes: bytes, *, source_address: int, row_pitch: int,
    bytes_per_row: int, operand: str, destination_offset: int,
    row_count: int = 1, gated: bool = False,
) -> tuple[int, int]:
    """Encode funct-27 rs1/rs2 without accepting RTL truncation or half wrap."""
    _selected_scale_fields(spec_bytes)
    values = (source_address, row_pitch, bytes_per_row, destination_offset, row_count)
    if any(type(value) is not int for value in values) or type(gated) is not bool:
        raise TypeError("MX scale-load fields must be integers and gated must be bool")
    if operand not in ("activation", "weight"):
        raise ValueError("scale operand must be activation or weight")
    if not 0 <= source_address < 1 << 40 or source_address % 8:
        raise ValueError("scale source address must be aligned and fit 40 bits")
    if not 0 <= row_pitch < 1 << 24 or row_pitch % 8:
        raise ValueError("scale row pitch must be zero or aligned and fit 24 bits")
    if not 8 <= bytes_per_row <= _HALF_BYTES or bytes_per_row % 8:
        raise ValueError("scale row bytes must fit one half and be an 8-byte multiple")
    if not 1 <= row_count <= 255:
        raise ValueError("scale row count must be between 1 and 255")
    if not 0 <= destination_offset < 2 * _HALF_BYTES or destination_offset % 8:
        raise ValueError("scale destination must be aligned and fit 13 bits")
    half_end = (destination_offset // _HALF_BYTES + 1) * _HALF_BYTES
    if destination_offset + row_count * bytes_per_row > half_end:
        raise ValueError("scale rows cross the selected 4 KiB destination half")
    effective_pitch = row_pitch or bytes_per_row
    if source_address + (row_count - 1) * effective_pitch + bytes_per_row > 1 << 40:
        raise ValueError("scale source rows exceed the 40-bit address field")
    rs1 = source_address | (row_pitch << 40)
    rs2 = (bytes_per_row | (int(operand == "weight") << 32) |
           (destination_offset << 33) | (row_count << 46) | (int(gated) << 54))
    return rs1, rs2
