"""Read file-backed RV64 ELF payloads without trusting generated C headers."""

from __future__ import annotations

import hashlib
import mmap
import struct
from collections import defaultdict
from pathlib import Path


class RV64PayloadImage:
    """Locate named static payloads through a checked ELF64 little-endian symbol table.

    Assembly data labels can have size zero. The caller must then supply a
    source-bound byte extent, which must fit the symbol's file-backed section.
    This reader checks payload identity; it does not validate executable code.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self.image = None
        self.symbols = defaultdict(list)

    def __enter__(self):
        self.file = self.path.open("rb")
        try:
            self.image = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._parse()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.image is not None:
            self.image.close()
        if self.file is not None:
            self.file.close()

    def _bounds(self, offset, length):
        if offset < 0 or length < 0 or offset + length > len(self.image):
            raise ValueError("ELF payload or table exceeds its actual file extent")

    def _parse(self):
        self._bounds(0, 64)
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", self.image)
        if (
            header[0][:7] != b"\x7fELF\x02\x01\x01"
            or header[1] != 2
            or header[2] != 243
            or header[3] != 1
            or header[8] != 64
            or header[11] != 64
            or not header[12]
        ):
            raise ValueError(
                "payload audit requires an RV64 ELF64 little-endian executable"
            )
        self._bounds(header[6], header[11] * header[12])
        self.sections = [
            struct.unpack_from("<IIQQQQIIQQ", self.image, header[6] + i * 64)
            for i in range(header[12])
        ]
        for table in self.sections:
            if table[1] != 2:
                continue
            if table[9] != 24 or table[5] % 24 or table[6] >= len(self.sections):
                raise ValueError(
                    "ELF symbol table has an unsupported entry or string table"
                )
            strings = self.sections[table[6]]
            if strings[1] != 3:
                raise ValueError("ELF symbol names are not a string table")
            self._bounds(table[4], table[5])
            self._bounds(strings[4], strings[5])
            for offset in range(table[4], table[4] + table[5], 24):
                name, info, _, section, address, size = struct.unpack_from(
                    "<IBBHQQ", self.image, offset
                )
                if (
                    info & 15 not in (0, 1)
                    or not 0 < section < len(self.sections)
                    or not name
                ):
                    continue
                if name >= strings[5]:
                    raise ValueError("ELF symbol name exceeds its string table")
                end = self.image.find(b"\0", strings[4] + name, strings[4] + strings[5])
                if end < 0:
                    raise ValueError("ELF symbol name has no terminator")
                text = self.image[strings[4] + name : end].decode("utf-8")
                self.symbols[text].append((section, address, size))
        if not self.symbols:
            raise ValueError("ELF has no usable static payload symbol table")

    def span(self, symbol, length):
        found = self.symbols.get(symbol, ())
        if len(found) != 1 or type(length) is not int or length <= 0:
            raise ValueError(
                "ELF payload needs one unambiguous symbol and a positive byte extent"
            )
        index, address, size = found[0]
        section = self.sections[index]
        relative = address - section[3]
        if (
            section[1] == 8
            or not section[2] & 2
            or relative < 0
            or relative + length > section[5]
            or size
            and length > size
        ):
            raise ValueError(
                "ELF payload exceeds its allocated file-backed symbol section"
            )
        offset = section[4] + relative
        self._bounds(offset, length)
        return offset, length

    def read(self, symbol, length):
        offset, length = self.span(symbol, length)
        return self.image[offset : offset + length]

    def digest(self, symbol, length):
        offset, length = self.span(symbol, length)
        with memoryview(self.image)[offset : offset + length] as view:
            return hashlib.sha256(view).hexdigest()
