"""A native gpt-oss B tile and a user FP4 callback reach one live GEMM stage."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from mx_gemmini_support.gptoss_native_tile import bind_native_expert_tile
from mx_gemmini_support.native_expert_gemm import emit_native_expert_gemm_stage
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
ABI = r'''
#ifndef TEST_MODEL_ACTIVATION_ABI_H
#define TEST_MODEL_ACTIVATION_ABI_H
#include <stdint.h>
typedef struct {
  const float *input;
  uint32_t logical_m, logical_k, input_row_stride, physical_m, physical_k;
  const char *region_id, *format, *projection;
  uint32_t code_bits, block_size, lut_group;
  uint8_t *codes;
  uint64_t codes_capacity_bytes;
  uint8_t *scales;
  uint64_t scales_capacity_bytes;
  uint32_t *lut;
  uint64_t lut_capacity_bytes;
  const uint8_t *settings_json;
  uint64_t settings_json_bytes;
} npu_mx_activation_quant_args_v1;
int npu_mx_quantize_activation_v1(const npu_mx_activation_quant_args_v1 *);
#endif
'''
CALLBACK = r'''
#include <string.h>
#include "model_activation_abi.h"
int npu_mx_quantize_activation_v1(const npu_mx_activation_quant_args_v1 *a) {
  if (!a || !a->input || !a->codes || !a->scales ||
      !a->region_id || strcmp(a->region_id, "gptoss.test.expert0") ||
      !a->format || strcmp(a->format, "fp4_e2m1") ||
      !a->projection || strcmp(a->projection, "direct") ||
      a->logical_m != 2 || a->logical_k != 64 ||
      a->physical_m != 32 || a->physical_k != 64 ||
      a->code_bits != 4 || a->block_size != 32 ||
      a->codes_capacity_bytes != 1024 || a->scales_capacity_bytes != 64 ||
      a->lut || a->lut_capacity_bytes) return 9;
  memset(a->codes, 0, a->codes_capacity_bytes);
  memset(a->scales, 127, a->scales_capacity_bytes);
  a->codes[0] = a->input[0] > 0.0f ? 2 : 1;
  return 0;
}
'''
HARNESS = r'''
#include <stdint.h>
#include <stdio.h>
extern const uint8_t gptoss_native_b_codes[];
extern const uint8_t gptoss_native_b_scales[];
void mx_gptoss_expert_tile(const float *, const float *, uint16_t *);
static uint8_t observed[2];
static unsigned calls;
void mx_gptoss_expert_issue(const void *a, const void *as, const void *b,
                            const void *bs, const void *out, const void *scratch) {
  (void)as; (void)scratch;
  if (b != gptoss_native_b_codes || bs != gptoss_native_b_scales) __builtin_trap();
  observed[calls++] = ((const uint8_t *)a)[0];
  ((uint16_t *)out)[0] = 0x3f80;
}
int main(void) {
  float lhs[2 * 64] = {0};
  uint16_t out[2 * 32] = {0};
  mx_gptoss_expert_tile(lhs, 0, out);
  lhs[0] = 1.0f;
  mx_gptoss_expert_tile(lhs, 0, out);
  if (calls != 2 || observed[0] != 1 || observed[1] != 2 || out[0] != 0x3f80)
    return 1;
  puts("PASS live native B callback stage");
  return 0;
}
'''


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_slice(path: Path, name: str, values: np.ndarray) -> bytes:
    payload = values.tobytes()
    header = json.dumps({name: {"dtype": "U8", "shape": list(values.shape),
                                "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return payload


class NativeExpertGemmTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT.parent / ".tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "extract"
        self.fixture.mkdir()
        self.tile = self.root / "tile"
        self.tile.mkdir()
        self.profile = load_profile(PROFILE)
        prefix = "model.layers.0.mlp.experts.gate_up_proj_"
        blocks = np.arange(32 * 2 * 16, dtype=np.uint16).astype(np.uint8).reshape(1, 32, 2, 16)
        scales = np.arange(64, dtype=np.uint8).reshape(1, 32, 2)
        rows = []
        for kind, values in (("blocks", blocks), ("scales", scales)):
            name = prefix + kind
            filename = kind + ".safetensors"
            payload = _write_slice(self.fixture / filename, name, values)
            rows.append({"name": name, "axis0_index": 0, "dtype": "U8",
                         "file": filename, "raw_sha256": _sha(payload),
                         "selected_shape": list(values.shape),
                         "source_shape": [2, *values.shape[1:]],
                         "source_byte_range": [100, 99 + len(payload)],
                         "shard": "model-00000-of-00002.safetensors",
                         "source_shard_bytes": 10000})
        extractor = {"schema": "mx_checkpoint_ranges.v1",
                     "model": "openai/gpt-oss-20b", "revision": "a" * 40,
                     "tensors": rows}
        self.extractor_path = self.fixture / "receipt.json"
        self.extractor_path.write_text(json.dumps(extractor, sort_keys=True) + "\n")
        original_a = bytes(2 * 64 * 2)
        codes, b_scales, manifest = bind_native_expert_tile(
            extractor, self.extractor_path.read_bytes(), self.fixture,
            projection="gate_up", expert_index=0, output_start=0,
            output_count=32, block_start=0, block_count=2,
            activation_bf16=original_a, activation_m=2,
            expected_revision="a" * 40, profile=self.profile)
        (self.tile / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (self.tile / "a_bf16.bin").write_bytes(original_a)
        (self.tile / "b_codes.bin").write_bytes(codes)
        (self.tile / "b_scales.bin").write_bytes(b_scales)
        self.abi = self.root / "model_activation_abi.h"
        self.abi.write_text(ABI)
        self.callback = self.root / "quant.c"
        self.callback.write_text(CALLBACK)
        self.site = "gptoss.test.expert0"
        native = {"checkpoint_revision": "a" * 40,
                  "codes_tensor": rows[0]["name"], "scales_tensor": rows[1]["name"],
                  "codes_sha256": rows[0]["raw_sha256"],
                  "scales_sha256": rows[1]["raw_sha256"]}
        self.run = {"schema": "npu_exploration.model_run.v2",
                    "name": "synthetic_native_expert_gemm",
                    "quantizer": {"kind": "external_wire", "provider": "test.opaque",
                                  "implementation_sha256": "b" * 64,
                                  "settings": {"rule": "test_only"},
                                  "activation_runtime": {
                                      "kind": "host_c_v1",
                                      "symbol": "npu_mx_quantize_activation_v1",
                                      "source_sha256": _sha(self.callback.read_bytes()),
                                      "headers_sha256": {"model_activation_abi.h": _sha(self.abi.read_bytes())}}},
                    "matrix_default": {"placement": "mesh",
                                       "activation_format": "fp4_e2m1", "weight_format": "fp4_e2m1",
                                       "activation_projection": "direct", "weight_projection": "direct",
                                       "activation_mode": "dynamic_per_call",
                                       "weight_source": "preserve_native", "output_format": "bf16",
                                       "native_weight": native}}
        self.run_path = self.root / "run.json"
        self.run_path.write_text(json.dumps(self.run, sort_keys=True) + "\n")

    def emit(self, name="stage"):
        return emit_native_expert_gemm_stage(
            tile_dir=self.tile, fixture_dir=self.fixture,
            extractor_receipt_path=self.extractor_path,
            run_recipe_path=self.run_path, profile=self.profile,
            site_id=self.site, callback_source_path=self.callback,
            abi_header_path=self.abi, out_dir=self.root / name)

    def test_live_fp4_a_and_checkpoint_b_reach_direct_gemm(self):
        receipt = self.emit()
        self.assertEqual(receipt["status"],
                         "generated_source_only_no_model_capture_elf_or_spike_parity")
        self.assertEqual(receipt["physical_shape_mnk"], [32, 32, 64])
        self.assertEqual(receipt["b_codes_sha256"], _sha((self.tile / "b_codes.bin").read_bytes()))
        stage = self.root / "stage"
        (stage / "harness.c").write_text(HARNESS)
        binary = stage / "host_test"
        subprocess.run(["cc", "-std=c11", "-I", str(stage),
                        str(stage / "model_entry.c"), str(stage / "native_weight.c"),
                        str(stage / "user_activation_quantizer.c"),
                        str(stage / "harness.c"), "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("PASS live native B callback stage", result.stdout)

    def test_tampered_b_and_recipe_are_refused(self):
        bad = bytearray((self.tile / "b_codes.bin").read_bytes())
        bad[0] ^= 1
        (self.tile / "b_codes.bin").write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "differs from original checkpoint"):
            self.emit()
        (self.tile / "b_codes.bin").write_bytes(bytes([bad[0] ^ 1, *bad[1:]]))
        changed = copy.deepcopy(self.run)
        changed["matrix_default"]["native_weight"]["codes_sha256"] = "0" * 64
        self.run_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "exact checkpoint FP4 B"):
            self.emit()

    def test_callback_or_profile_drift_is_refused(self):
        self.callback.write_text(CALLBACK + "\n/* changed */\n")
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.emit()
        self.callback.write_text(CALLBACK)
        changed = copy.deepcopy(self.profile)
        changed["resources"]["scratchpad_bytes"] += 64
        with self.assertRaisesRegex(ValueError, "source-bound FP4 profile"):
            emit_native_expert_gemm_stage(
                tile_dir=self.tile, fixture_dir=self.fixture,
                extractor_receipt_path=self.extractor_path,
                run_recipe_path=self.run_path, profile=changed,
                site_id=self.site, callback_source_path=self.callback,
                abi_header_path=self.abi, out_dir=self.root / "wrong-profile")


if __name__ == "__main__":
    unittest.main()
