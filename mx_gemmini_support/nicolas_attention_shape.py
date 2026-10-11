"""Scratchpad and buffer geometry for Nicolas's dense MX attention family."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct


@dataclass(frozen=True)
class DenseAttentionShape:
    sq: int
    sk: int
    d: int

    @classmethod
    def from_header(cls, header: str) -> "DenseAttentionShape":
        values = []
        for name in ("SQ", "SK", "D"):
            matches = re.findall(rf"^#define ATTN_{name}\s+(\d+)\s*$", header,
                                 re.MULTILINE)
            if len(matches) != 1:
                raise ValueError(f"dense attention header needs one ATTN_{name}")
            values.append(int(matches[0]))
        return cls(*values)

    @classmethod
    def from_plan(cls, plan: dict) -> "DenseAttentionShape":
        dims = plan.get("shape_sq_sk_d")
        if (not isinstance(dims, list) or len(dims) != 3 or
                any(type(value) is not int for value in dims)):
            raise ValueError("dense attention binding shape is malformed")
        return cls(*dims)

    def validate(self, profile: dict) -> None:
        if (any(type(value) is not int for value in (self.sq, self.sk, self.d)) or
                self.sq <= 0 or self.sk <= 0 or self.d <= 0 or
                self.sq % 16 or self.sk % 32 or self.d % 32 or
                self.requant_blocks > 2048 or self.end_row > self.max_row or
                profile.get("transport") != "rocket_rocc" or
                profile.get("geometry", {}).get("mesh_rows") != 16 or
                profile.get("geometry", {}).get("mesh_columns") != 16 or
                not profile.get("resources", {}).get("vpu") or
                not profile.get("resources", {}).get("spad_requant") or
                self.sum_row + self.sq >
                profile.get("resources", {}).get("scratchpad_bytes", 0) // 16):
            raise ValueError("dense attention shape exceeds DIM16 MX+VPU residency")

    @property
    def kt_row(self) -> int:
        return self.sq * self.d // 16

    @property
    def score_row(self) -> int:
        return self.kt_row + self.d * self.sk // 16

    @property
    def probability_row(self) -> int:
        return self.score_row + self.sq * self.sk * 2 // 16

    @property
    def v_row(self) -> int:
        return self.probability_row + self.sq * self.sk // 16

    @property
    def output_row(self) -> int:
        return self.v_row + self.sk * self.d // 16

    @property
    def end_row(self) -> int:
        return self.output_row + self.sq * self.d * 2 // 16

    @property
    def max_row(self) -> int:
        return 0x3000

    @property
    def sum_row(self) -> int:
        return self.max_row + self.sq

    @property
    def vpu_rows(self) -> int:
        return self.sq * self.sk // 8

    @property
    def reduction_length(self) -> int:
        return self.sk // 8

    @property
    def requant_blocks(self) -> int:
        return self.sq * self.sk // 32

    @property
    def scale_immediate_bf16(self) -> int:
        # The source computes vpu_f_to_bf16(1.0f / sqrtf((float)D)).
        root = struct.unpack("<f", struct.pack("<f", math.sqrt(self.d)))[0]
        value = struct.unpack("<I", struct.pack("<f", 1.0 / root))[0]
        upper, lower = value >> 16, value & 0xffff
        return upper + int(lower > 0x8000 or (lower == 0x8000 and upper & 1))

    def buffer_bytes(self) -> dict[str, int]:
        return {
            "Q_SCALES": self.sq * self.d // 32,
            "KT_SCALES": self.d * self.sk // 32,
            "V_SCALES": self.sk * self.d // 32,
            "Q_IN": self.sq * self.d,
            "KT_IN": self.d * self.sk,
            "V_IN": self.sk * self.d,
            "scale_sink": 2048,
            "S_hw": self.sq * self.sk * 2,
            "p_scales": self.sq * self.sk // 32,
            "O_hw": self.sq * self.d * 2,
            "P_hw": self.sq * self.sk * 2,
            "p_codes_hw": self.sq * self.sk,
        }
