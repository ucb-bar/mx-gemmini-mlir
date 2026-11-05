"""A model-stage adapter sends its live lhs pointer to a pinned user C rule."""

from __future__ import annotations

import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from mx_gemmini_support.dense_kernel import emit_dense_f32_kernel
from mx_gemmini_support.model_activation_callback import (
    bind_callbacks_from_model_plan, bind_model_activation_callback,
)
from mx_gemmini_support.merlin_catalog_plan import plan_merlin_catalog
from mx_gemmini_support.runtime_issuer import RuntimeIssuer
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
NPU_HEADER = ROOT.parent / "npu-exploration/compiler/model_activation_abi.h"
USER_C = r'''
#include "model_activation_abi.h"
#include <string.h>
int npu_mx_quantize_activation_v1(const npu_mx_activation_quant_args_v1 *a) {
  if (!a || !a->input || !a->codes || !a->scales ||
      strcmp(a->region_id, "dense_0") || strcmp(a->format, "fp8_e4m3") ||
      strcmp(a->projection, "direct") || a->code_bits != 8 ||
      a->block_size != 32 || a->logical_m != 7 || a->logical_k != 64 ||
      a->physical_m != 16 || a->physical_k != 64 ||
      a->input_row_stride != 64 || a->codes_capacity_bytes != 1024 ||
      a->scales_capacity_bytes != 32 || a->lut || a->lut_capacity_bytes ||
      a->settings_json_bytes != 10 ||
      memcmp(a->settings_json, "{\"mode\":1}", 10)) return -1;
  const uint8_t code = a->input[0] > 0.0f ? 2 : 1;
  for (unsigned i = 0; i < 1024; ++i) a->codes[i] = code;
  for (unsigned i = 0; i < 32; ++i) a->scales[i] = 127;
  return 0;
}
'''
ISSUER_C = r'''
#include <stdint.h>
const uint8_t fixed_codes[1024] = {0};
const uint8_t fixed_scales[32] = {0};
uint8_t observed_first_code;
void mx_issue_encoded(const void *a, const void *as, const void *b,
                      const void *bs, const void *out, const void *scratch) {
  (void)as; (void)b; (void)bs; (void)scratch;
  observed_first_code = ((const uint8_t *)a)[0];
  uint16_t *words = (uint16_t *)out;
  for (unsigned i = 0; i < 16 * 16; ++i) words[i] = 0x3f80;
}
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelActivationCallbackTests(unittest.TestCase):
    def test_npu_plan_binding_refuses_changed_recipe_or_unsupported_mode(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            source = Path(scratch) / "user.c"
            source.write_text(USER_C)
            runtime = {"kind": "host_c_v1", "symbol": "npu_mx_quantize_activation_v1",
                       "source_sha256": _sha(source),
                       "headers_sha256": {"model_activation_abi.h": _sha(NPU_HEADER)}}
            plan = {
                "schema": "npu_exploration.model_quantization_plan.v2",
                "status": "planning_only_no_executable",
                "source_mlir_sha256": "a" * 64, "profile_sha256": "b" * 64,
                "quantizer": {"kind": "external_wire", "settings": {"mode": 1},
                              "activation_runtime": runtime},
                "sites": [{"region_id": "dense_0", "status": "structural_candidate_not_executable",
                           "shape_mnk": [7, 16, 64],
                           "requested": {"placement": "mesh", "activation_format": "fp8_e4m3",
                                         "weight_format": "fp8_e4m3",
                                         "activation_projection": "direct",
                                         "weight_projection": "direct", "output_format": "bf16"}}],
            }
            admission = {
                "schema": "mx_gemmini.model_quantization_admission.v1",
                "status": "structural_admission_only_no_executable",
                "run_recipe_digest_status": "recomputed_from_selected_recipe",
                "source_mlir_sha256": plan["source_mlir_sha256"],
                "profile_sha256": plan["profile_sha256"],
                "canonical_plan_sha256": hashlib.sha256((json.dumps(
                    plan, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
            }
            selected = bind_callbacks_from_model_plan(
                plan, admission, regions=frozenset({"dense_0"}),
                source_path=source, abi_header_path=NPU_HEADER)
            self.assertEqual(selected["dense_0"].settings_json, '{"mode":1}')
            changed = json.loads(json.dumps(plan))
            changed["quantizer"]["settings"]["mode"] = 2
            with self.assertRaisesRegex(ValueError, "differs from MX admission"):
                bind_callbacks_from_model_plan(
                    changed, admission, regions=frozenset({"dense_0"}),
                    source_path=source, abi_header_path=NPU_HEADER)
            changed = json.loads(json.dumps(plan))
            changed["sites"][0]["requested"]["activation_format"] = "fp6_e3m2"
            admission["canonical_plan_sha256"] = hashlib.sha256((json.dumps(
                changed, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "no direct rank-2"):
                bind_callbacks_from_model_plan(
                    changed, admission, regions=frozenset({"dense_0"}),
                    source_path=source, abi_header_path=NPU_HEADER)

    def test_real_model_plan_splits_selected_source_region(self):
        archive = ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593"
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            source = Path(scratch) / "model.mlir"
            source.write_bytes(gzip.decompress((archive / "small_llama.model2mlir.mlir.gz").read_bytes()))
            old = os.environ.get("MERLIN_TARGET_PATH")
            os.environ["MERLIN_TARGET_PATH"] = str(ROOT)
            try:
                baseline = plan_merlin_catalog(source, profile)
                selected = plan_merlin_catalog(
                    source, profile, activation_callback_regions=frozenset({"matmul_0"}))
            finally:
                if old is None:
                    os.environ.pop("MERLIN_TARGET_PATH", None)
                else:
                    os.environ["MERLIN_TARGET_PATH"] = old
            by_region = {row["region"]: row["symbol"] for row in selected["bindings"]}
            self.assertEqual(selected["selected_operations"], baseline["selected_operations"])
            self.assertEqual(selected["unique_kernels"], baseline["unique_kernels"] + 1)
            self.assertEqual(len([entry for entry in selected["kernels"]
                                  if entry.get("activation_callback_region") == "matmul_0"]), 1)
            self.assertNotEqual(by_region["matmul_0"], by_region["matmul_1"])

    def test_live_lhs_reaches_callback_and_issuer_with_exact_buffers(self):
        self.assertTrue(NPU_HEADER.is_file(), "managed npu sibling supplies the ABI header")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            folder = Path(scratch)
            user_file = folder / "user.c"
            user_file.write_text(USER_C)
            recipe = {"kind": "host_c_v1", "symbol": "npu_mx_quantize_activation_v1",
                      "source_sha256": _sha(user_file),
                      "headers_sha256": {"model_activation_abi.h": _sha(NPU_HEADER)}}
            callback = bind_model_activation_callback(
                recipe, {"mode": 1}, region_id="dense_0",
                source_path=user_file, abi_header_path=NPU_HEADER)
            issuer = RuntimeIssuer("", (16, 16, 64), "FP8", "p", "q", {
                "activation": 1024, "activation_scales": 32,
                "weight": 1024, "weight_scales": 32,
                "output_bf16": 512, "scratch_output_scales": 2048})
            generated = emit_dense_f32_kernel(
                issuer, (7, 16, 64), symbol="live_kernel",
                prepacked_weight_symbols=("fixed_codes", "fixed_scales"),
                shared_workspace_symbol="shared_workspace",
                activation_callback=callback)
            self.assertEqual(generated.activation_callback_sha256, _sha(user_file))
            self.assertIn(".input = lhs", generated.source)
            self.assertNotIn("mx_pack_activation_f32(", generated.source)
            (folder / "kernel.c").write_text(generated.source)
            (folder / "issuer.c").write_text(ISSUER_C)
            (folder / "workspace.c").write_text(
                "unsigned char shared_workspace[8192] __attribute__((aligned(64)));\n")
            (folder / NPU_HEADER.name).write_bytes(NPU_HEADER.read_bytes())
            library = folder / "model.so"
            subprocess.run([
                "cc", "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
                "-I", str(folder), "-I", str(ROOT / "mx_gemmini_support"),
                str(folder / "kernel.c"), str(folder / "issuer.c"),
                str(folder / "workspace.c"), str(user_file),
                "-o", str(library),
            ], check=True, capture_output=True, text=True)
            loaded = ctypes.CDLL(str(library))
            loaded.live_kernel.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
            lhs = (ctypes.c_float * (7 * 64))()
            rhs = (ctypes.c_float * (64 * 16))()
            out = (ctypes.c_float * (7 * 16))()
            lhs[0] = -1.0
            loaded.live_kernel(lhs, rhs, out)
            self.assertEqual(ctypes.c_uint8.in_dll(loaded, "observed_first_code").value, 1)
            lhs[0] = 1.0
            loaded.live_kernel(lhs, rhs, out)
            self.assertEqual(ctypes.c_uint8.in_dll(loaded, "observed_first_code").value, 2)
            self.assertEqual(out[0], 1.0)

            user_file.write_text(USER_C + "\n/* changed */\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                bind_model_activation_callback(recipe, {"mode": 1}, region_id="dense_0",
                                               source_path=user_file,
                                               abi_header_path=NPU_HEADER)
            user_file.write_text(USER_C)
            changed_header = folder / "changed_abi.h"
            changed_header.write_bytes(NPU_HEADER.read_bytes() + b"\n/* changed */\n")
            with self.assertRaisesRegex(ValueError, "header model_activation_abi.h differs"):
                bind_model_activation_callback(recipe, {"mode": 1}, region_id="dense_0",
                                               source_path=user_file,
                                               abi_header_path=changed_header)


if __name__ == "__main__":
    unittest.main()
