"""Physical MX row reuse and async completion safety."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest

from mx_gemmini_support.command_ir import Command, Fence, Operand
from mx_gemmini_support.physical_memory_plan import (
    Access, MemoryEvent, MemoryValue, PhysicalMemoryError,
    _single_fp8_accumulator_rows, plan_physical_memory,
    source_bound_capacities, source_bound_fp6_lut_caches,
    source_bound_single_fp8_accumulator,
)
from mx_gemmini_support.target_profile import _revision, export_profiles_v3, profile_sha256


def _profile(dim: int = 16) -> dict:
    return {"schema": "mx_gemmini.target_profile.v3",
            "geometry": {"mesh_rows": dim, "mesh_columns": dim},
            "resources": {"scratchpad_bytes": dim * 16,
                          "scratchpad_row_bits": dim * 8,
                          "accumulator_bytes": 512,
                          "accumulator_row_bits": 1024,
                          "accumulator_rows": 4,
                          "scale_mem_config": {"size_bytes": 256,
                                               "subbank_line_bytes": 16}}}


def _bindings(profile: dict) -> dict[str, str]:
    return {"profile_sha256": profile_sha256(profile),
            "plan_sha256": "1" * 64, "driver_sha256": "2" * 64}


def _opaque(id: str, bindings: dict, *, reads=(), writes=(),
            queue: str = "gemmini") -> MemoryEvent:
    source = "source-bound test dispatch"
    effects = {"source": source,
               "reads": [asdict(x) for x in reads],
               "writes": [asdict(x) for x in writes]}
    digest = hashlib.sha256(json.dumps(effects, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    cert = {**bindings, "effects_sha256": digest}
    return MemoryEvent(id, "opaque", source, reads, writes,
                       queue=queue, certificate=cert)


def _fence(id: str, queue: str = "gemmini") -> MemoryEvent:
    return MemoryEvent(id, "fence", "selected completion", command=Fence(),
                       completes=(queue,))


class PhysicalMemoryPlanTest(unittest.TestCase):
    def test_overlapping_fixed_rows_refuse_while_both_async_writes_live(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("a", "scratchpad", 4, "A", fixed_row=0),
                  MemoryValue("b", "scratchpad", 4, "B", fixed_row=2))
        events = (_opaque("dispatch", bindings,
                          writes=(Access("a", 0, 4), Access("b", 0, 4))),
                  _fence("complete"))
        with self.assertRaisesRegex(PhysicalMemoryError, "overlaps live physical rows"):
            plan_physical_memory(profile, values, events, bindings=bindings)

    def test_last_use_reuses_rows_after_certified_fence(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("a", "scratchpad", 4, "first stage", fixed_row=0),
                  MemoryValue("b", "scratchpad", 4, "second stage"))
        events = (_opaque("first", bindings, writes=(Access("a", 0, 4),)),
                  _fence("first_done"),
                  _opaque("second", bindings, writes=(Access("b", 0, 4),)),
                  _fence("second_done"))
        receipt = plan_physical_memory(profile, values, events, bindings=bindings)
        self.assertEqual(receipt["placement"]["a"]["row"], 0)
        self.assertEqual(receipt["placement"]["b"]["row"], 0)
        self.assertEqual(receipt["peaks"]["scratchpad"]["peak_live_rows"], 4)

    def test_view_alias_extends_host_storage_without_new_allocation(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("producer", "host", 32, "materialization"),
                  MemoryValue("view", "host", 32, "captured reshape", alias_of="producer"),
                  MemoryValue("sink", "host", 32, "consumer"))
        events = (MemoryEvent("produce", "host", "producer",
                              writes=(Access("producer", 0, 32),)),
                  MemoryEvent("reshape", "view", "captured view",
                              writes=(Access("view", 0, 32),)),
                  MemoryEvent("consume", "host", "consumer",
                              reads=(Access("view", 0, 32),),
                              writes=(Access("sink", 0, 32),)))
        receipt = plan_physical_memory(profile, values, events, bindings=bindings)
        self.assertEqual(receipt["placement"]["view"]["alias_of"], "producer")
        self.assertEqual(receipt["placement"]["producer"]["last_use_completed_at"], 2)
        self.assertEqual(receipt["peaks"]["scratchpad"]["active_values"], 0)

    def test_unknown_lut_capacity_and_fixed_overflow_refuse(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        for value, pattern in (
            (MemoryValue("too_large", "scratchpad", 4, "bad row", fixed_row=14),
             "exceed source capacity"),
            (MemoryValue("lut", "lut", 1, "uncertified LUT", fixed_row=0),
             "LUT/unknown physical capacity"),
        ):
            with self.subTest(value=value.id), self.assertRaisesRegex(
                    PhysicalMemoryError, pattern):
                plan_physical_memory(profile, (value,),
                                     (_opaque("write", bindings,
                                              writes=(Access(value.id, 0, value.rows),)),
                                      _fence("done")), bindings=bindings)

    def test_fence_releases_only_its_named_queue(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("on_chip", "scratchpad", 2, "async DMA", fixed_row=0),)
        events = (_opaque("write", bindings, writes=(Access("on_chip", 0, 2),),
                          queue="dma"),
                  _fence("wrong_completion", queue="compute"))
        with self.assertRaisesRegex(PhysicalMemoryError, "unknown queue compute"):
            plan_physical_memory(profile, values, events, bindings=bindings)

    def test_host_read_before_async_completion_refuses(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("dma_result", "host", 16, "DMA readout"),
                  MemoryValue("consumer", "host", 16, "host output"))
        events = (_opaque("dma", bindings, writes=(Access("dma_result", 0, 16),)),
                  MemoryEvent("early_read", "host", "host consumer",
                              reads=(Access("dma_result", 0, 16),),
                              writes=(Access("consumer", 0, 16),)),
                  _fence("too_late"))
        with self.assertRaisesRegex(PhysicalMemoryError, "before its async producer completes"):
            plan_physical_memory(profile, values, events, bindings=bindings)

    def test_four_source_legal_row_geometries_have_capacity(self) -> None:
        for dim in (4, 8, 16, 32):
            with self.subTest(dim=dim):
                profile = _profile(dim)
                self.assertEqual(source_bound_capacities(profile)["scratchpad"], 16)
                self.assertIsNone(source_bound_capacities(profile)["lut"])

    def test_unknown_command_and_changed_opaque_effects_refuse(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        host = MemoryValue("host", "host", 8, "external input", external=True)
        unknown = MemoryEvent("unknown", "command", "unmodeled opcode",
                              command=Command(99, Operand(immediate=0),
                                              Operand(immediate=0)), queue="gemmini")
        with self.assertRaisesRegex(PhysicalMemoryError, "no checked physical effect model"):
            plan_physical_memory(profile, (host,), (unknown,), bindings=bindings)
        on_chip = MemoryValue("spad", "scratchpad", 2, "declared write", fixed_row=0)
        certified = _opaque("matrix", bindings, writes=(Access("spad", 0, 2),))
        changed = MemoryEvent("matrix", "opaque", certified.source,
                              writes=(Access("spad", 0, 1),), queue="gemmini",
                              certificate=certified.certificate)
        with self.assertRaisesRegex(PhysicalMemoryError, "lacks exact certified effects"):
            plan_physical_memory(profile, (on_chip,), (changed, _fence("done")),
                                 bindings=bindings)

    def test_binary_vpu_mul_requires_both_encoded_source_rows(self) -> None:
        profile = _profile()
        bindings = _bindings(profile)
        values = (MemoryValue("lhs", "scratchpad", 2, "lhs", fixed_row=0,
                              external=True),
                  MemoryValue("rhs", "scratchpad", 2, "rhs", fixed_row=4,
                              external=True),
                  MemoryValue("out", "scratchpad", 2, "result", fixed_row=8))
        rs1 = (4 << 14) | (8 << 28) | (2 << 42)
        mul = Command(33, Operand(immediate=rs1), Operand(immediate=2 | (1 << 5)))
        good = MemoryEvent("mul", "command", "encoded VPU mul",
                           reads=(Access("lhs", 0, 2), Access("rhs", 0, 2)),
                           writes=(Access("out", 0, 2),),
                           command=mul, queue="gemmini")
        receipt = plan_physical_memory(profile, values, (good, _fence("done")),
                                       bindings=bindings)
        self.assertEqual(receipt["placement"]["out"]["row"], 8)
        wrong = MemoryEvent("mul", "command", "changed source 2",
                            reads=(Access("lhs", 0, 2), Access("rhs", 1, 1)),
                            writes=(Access("out", 0, 2),),
                            command=mul, queue="gemmini")
        with self.assertRaisesRegex(PhysicalMemoryError, "binary mul effect differs"):
            plan_physical_memory(profile, values, (wrong, _fence("done")),
                                 bindings=bindings)

    def test_single_fp8_packed_execute_and_store_bounds(self) -> None:
        self.assertEqual(_single_fp8_accumulator_rows(1, 22), 96)
        self.assertEqual(_single_fp8_accumulator_rows(2, 22), 192)
        self.assertEqual(_single_fp8_accumulator_rows(1, 64), 256)
        self.assertEqual(_single_fp8_accumulator_rows(1, 65), 272)
        self.assertEqual(_single_fp8_accumulator_rows(2, 86, dim=4), 192)
        self.assertEqual(_single_fp8_accumulator_rows(1, 43, dim=8), 96)
        self.assertEqual(_single_fp8_accumulator_rows(1, 11, dim=32), 96)
        with self.assertRaisesRegex(PhysicalMemoryError, "source-qualified mesh DIM"):
            _single_fp8_accumulator_rows(1, 11, dim=64)
        with self.assertRaisesRegex(PhysicalMemoryError, "positive tile counts"):
            _single_fp8_accumulator_rows(0, 22)

    def test_source_pinned_packed_extent_across_coupled_geometries(self) -> None:
        rtl = Path(__file__).resolve().parents[2] / "gemmini"
        if not (rtl / "src/main/scala/gemmini/LoopMatmul.scala").is_file():
            self.skipTest("selected Gemmini sibling is unavailable")
        import subprocess
        revision = subprocess.check_output(
            ["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip()
        if revision not in {
                "12912323f1aa66d7b28687f01faf0a33af455f6f",
                "abd61f98e11855fb25d5da0623724239d83e78cc"}:
            self.skipTest("selected Gemmini source revision is unavailable")
        cases = (
            ("MxDim4Fp4E3M2VpuGemminiRocketConfig", 2, 86, 192),
            ("MxDim8Fp4E3M2E4M3VpuGemminiRocketConfig", 1, 43, 96),
            ("MxFp4E3M2E4M3VpuGemminiRocketConfig", 1, 22, 96),
            ("MxDim32E4M3VpuWideMemGemminiRocketConfig", 1, 11, 96),
        )
        profiles = export_profiles_v3(rtl, names={case[0] for case in cases})
        for name, tiles_i, tiles_j, expected in cases:
            with self.subTest(name=name):
                proof = source_bound_single_fp8_accumulator(
                    profiles[name], tiles_i=tiles_i, tiles_j=tiles_j,
                    rtl_root=rtl)
                self.assertEqual(proof["rows"], expected)
                self.assertLessEqual(expected, proof["capacity_rows"])

    def test_selected_acc32_source_pin_and_capacity(self) -> None:
        rtl = Path(__file__).resolve().parents[2] / "gemmini"
        if not (rtl / "src/main/scala/gemmini/LoopMatmul.scala").is_file():
            self.skipTest("selected Gemmini sibling is unavailable")
        profile = _profile()
        profile["name"] = "MxFp4E3M2E4M3VpuAcc32GemminiRocketConfig"
        profile["resources"].update(accumulator_bytes=32768,
                                    accumulator_row_bits=1024,
                                    accumulator_rows=256)
        profile["source"] = {"gemmini_revision": _revision(rtl)}
        proof = source_bound_single_fp8_accumulator(
            profile, tiles_i=1, tiles_j=22, rtl_root=rtl)
        self.assertEqual((proof["rows"], proof["capacity_rows"]), (96, 256))
        with self.assertRaisesRegex(PhysicalMemoryError, "needs 272 accumulator rows"):
            source_bound_single_fp8_accumulator(
                profile, tiles_i=1, tiles_j=65, rtl_root=rtl)
        profile["source"]["gemmini_revision"] = "0" * 40
        with self.assertRaisesRegex(PhysicalMemoryError, "certified source profile"):
            source_bound_single_fp8_accumulator(
                profile, tiles_i=1, tiles_j=22, rtl_root=rtl)

    def test_selected_fp6_direct_lut_cache_lifetime_and_bounds(self) -> None:
        rtl = Path(__file__).resolve().parents[2] / "gemmini"
        if _revision(rtl) != "12912323f1aa66d7b28687f01faf0a33af455f6f":
            self.skipTest("FP6 LUT cache source proof is qualified only on Gemmini 12912323")
        profile = export_profiles_v3(
            rtl, names={"MxFp4E3M2E4M3VpuGemminiRocketConfig"})[
                "MxFp4E3M2E4M3VpuGemminiRocketConfig"]
        proof = source_bound_fp6_lut_caches(profile, rtl)
        self.assertEqual(proof["capacity_codebooks"], {
            "lut_weight": 64, "lut_activation": 64, "lut_output": 64})
        bindings = _bindings(profile)
        values = (MemoryValue("B", "lut_weight", 44, "direct sel0", fixed_row=0),
                  MemoryValue("A", "lut_activation", 4, "direct sel1", fixed_row=0),
                  MemoryValue("C", "lut_output", 4, "direct sel2", fixed_row=0))
        events = (_opaque("load", bindings,
                          writes=(Access("B", 0, 44), Access("A", 0, 4),
                                  Access("C", 0, 4))),
                  _opaque("compute", bindings,
                          reads=(Access("B", 0, 44), Access("A", 0, 4),
                                 Access("C", 0, 4))),
                  _fence("complete"))
        with self.assertRaisesRegex(PhysicalMemoryError, "LUT/unknown physical capacity"):
            plan_physical_memory(profile, values, events, bindings=bindings)
        receipt = plan_physical_memory(profile, values, events, bindings=bindings,
                                       rtl_root=rtl)
        self.assertEqual(receipt["peaks"]["lut_weight"]["peak_live_rows"], 44)
        self.assertEqual(receipt["placement"]["B"]["last_use_completed_at"], 2)
        for bad in (MemoryValue("too_many", "lut_weight", 65, "overflow", fixed_row=0),
                    MemoryValue("past_end", "lut_weight", 1, "address", fixed_row=64)):
            with self.subTest(bad=bad.id), self.assertRaisesRegex(
                    PhysicalMemoryError, "exceed source capacity"):
                plan_physical_memory(profile, (bad,),
                                     (_opaque("write", bindings,
                                              writes=(Access(bad.id, 0, bad.rows),)),
                                      _fence("done")),
                                     bindings=bindings, rtl_root=rtl)
        overlap = (MemoryValue("B0", "lut_weight", 44, "first", fixed_row=0),
                   MemoryValue("B1", "lut_weight", 44, "second", fixed_row=0))
        with self.assertRaisesRegex(PhysicalMemoryError, "overlaps live physical rows"):
            plan_physical_memory(profile, overlap,
                                 (_opaque("both", bindings,
                                          writes=(Access("B0", 0, 44),
                                                  Access("B1", 0, 44))),
                                  _fence("done")), bindings=bindings, rtl_root=rtl)
        profile["source"]["gemmini_revision"] = "0" * 40
        with self.assertRaisesRegex(PhysicalMemoryError, "source-bound DIM16 profile"):
            source_bound_fp6_lut_caches(profile, rtl)


if __name__ == "__main__":
    unittest.main()
