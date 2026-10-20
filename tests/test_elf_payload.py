"""ELF payload identity must come from bounded sections and unambiguous symbols."""

import hashlib
import struct

import pytest

from mx_gemmini_support.elf_payload import RV64PayloadImage


def fixture(
    path, *, size=8, duplicate=False, machine=243, nobits=False, data_offset=0x100
):
    data = bytearray(0x300)
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        ident,
        2,
        machine,
        1,
        0,
        0,
        0x200,
        0,
        64,
        0,
        0,
        64,
        4,
        0,
    )
    data[0x100:0x108] = b"payload!"
    data[0x140:0x147] = b"\0input\0"
    struct.pack_into("<IBBHQQ", data, 0x180 + 24, 1, 1, 0, 1, 0x80001000, size)
    if duplicate:
        struct.pack_into("<IBBHQQ", data, 0x180 + 48, 1, 1, 0, 1, 0x80001000, size)
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        0x200 + 64,
        0,
        8 if nobits else 1,
        2,
        0x80001000,
        data_offset,
        8,
        0,
        0,
        1,
        0,
    )
    struct.pack_into("<IIQQQQIIQQ", data, 0x200 + 128, 0, 3, 0, 0, 0x140, 7, 0, 0, 1, 0)
    struct.pack_into(
        "<IIQQQQIIQQ", data, 0x200 + 192, 0, 2, 0, 0, 0x180, 72, 2, 0, 8, 24
    )
    path.write_bytes(data)
    return path


def test_actual_payload_bytes_and_digest(tmp_path):
    path = fixture(tmp_path / "fixture.elf")
    with RV64PayloadImage(path) as image:
        assert image.read("input", 8) == b"payload!"
        assert image.digest("input", 8) == hashlib.sha256(b"payload!").hexdigest()
    assert image.image.closed and image.file.closed


def test_unsized_assembly_labels_require_a_bounded_external_extent(tmp_path):
    with RV64PayloadImage(fixture(tmp_path / "fixture.elf", size=0)) as image:
        assert image.read("input", 8) == b"payload!"
        for count in (0, -1, 9, True):
            with pytest.raises(ValueError):
                image.read("input", count)
        with pytest.raises(ValueError):
            image.read("unknown", 1)


@pytest.mark.parametrize(
    "options",
    [{"duplicate": True}, {"nobits": True}, {"size": 4}, {"data_offset": 0x1000}],
)
def test_ambiguous_truncated_or_non_file_backed_payload_refuses(tmp_path, options):
    with RV64PayloadImage(fixture(tmp_path / "fixture.elf", **options)) as image:
        with pytest.raises(ValueError):
            image.read("input", 8)


def test_wrong_machine_or_truncated_file_refuses_and_closes(tmp_path):
    for path in (
        fixture(tmp_path / "wrong.elf", machine=62),
        tmp_path / "truncated.elf",
    ):
        if path.name == "truncated.elf":
            path.write_bytes(b"\x7fELF")
        image = RV64PayloadImage(path)
        with pytest.raises(ValueError):
            with image:
                pass
        assert image.file.closed and image.image.closed


def write_payload_elf(path, payloads):
    """Build a small RV64 data image for corruption checks, without a toolchain."""
    data = bytearray(0x100)
    labels = {}
    strings = bytearray(b"\0")
    symbols = []
    for name, payload in payloads.items():
        data.extend(b"\0" * ((-len(data)) % 8))
        offset = len(data)
        data.extend(payload)
        labels[name] = offset
        name_offset = len(strings)
        strings.extend(name.encode() + b"\0")
        symbols.append((name_offset, 1, 0, 1, 0x80000000 + offset, len(payload)))
    payload_size = len(data) - 0x100
    data.extend(b"\0" * ((-len(data)) % 8))
    string_offset = len(data)
    data.extend(strings)
    data.extend(b"\0" * ((-len(data)) % 8))
    symbol_offset = len(data)
    data.extend(b"\0" * 24)
    for symbol in symbols:
        data.extend(struct.pack("<IBBHQQ", *symbol))
    data.extend(b"\0" * ((-len(data)) % 8))
    section_offset = len(data)
    data.extend(b"\0" * 256)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        2,
        243,
        1,
        0,
        0,
        section_offset,
        0,
        64,
        0,
        0,
        64,
        4,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 64,
        0,
        1,
        2,
        0x80000100,
        0x100,
        payload_size,
        0,
        0,
        8,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 128,
        0,
        3,
        0,
        0,
        string_offset,
        len(strings),
        0,
        0,
        1,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 192,
        0,
        2,
        0,
        0,
        symbol_offset,
        24 * (len(symbols) + 1),
        2,
        0,
        8,
        24,
    )
    path.write_bytes(data)
    return labels
