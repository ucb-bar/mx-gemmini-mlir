"""Synthetic, source-bound admission of one native gpt-oss expert tile."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mx_gemmini_support.gptoss_checkpoint import restore_checkpoint_layout, PackedExpertWeight
from mx_gemmini_support.gptoss_native_tile import bind_native_expert_tile


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_slice(path: Path, name: str, values: np.ndarray) -> bytes:
    payload = values.tobytes()
    header = json.dumps({name: {"dtype": "U8", "shape": list(values.shape),
                                "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return payload


class NativeTileTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[2] / ".tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = "model.layers.0.mlp.experts.gate_up_proj_"
        self.blocks = np.arange(32 * 2 * 16, dtype=np.uint16).astype(np.uint8).reshape(1, 32, 2, 16)
        self.scales = np.arange(64, dtype=np.uint8).reshape(1, 32, 2)
        rows = []
        for kind, values in (("blocks", self.blocks), ("scales", self.scales)):
            name = self.prefix + kind
            filename = kind + ".safetensors"
            payload = _write_slice(self.root / filename, name, values)
            rows.append({"name": name, "axis0_index": 0, "dtype": "U8",
                         "file": filename, "raw_sha256": _sha(payload),
                         "selected_shape": list(values.shape),
                         "source_shape": [2, *values.shape[1:]],
                         "source_byte_range": [100, 99 + len(payload)],
                         "shard": "model-00000-of-00002.safetensors",
                         "source_shard_bytes": 10000})
        self.receipt = {"schema": "mx_checkpoint_ranges.v1",
                        "model": "openai/gpt-oss-20b", "revision": "a" * 40,
                        "tensors": rows}
        self.activation = bytes(2 * 64 * 2)

    def bind(self):
        raw = (json.dumps(self.receipt, sort_keys=True) + "\n").encode()
        return bind_native_expert_tile(self.receipt, raw, self.root,
                                       projection="gate_up", expert_index=0,
                                       output_start=0, output_count=32,
                                       block_start=0, block_count=2,
                                       activation_bf16=self.activation, activation_m=2,
                                       expected_revision="a" * 40)

    def test_lossless_tile_and_manifest(self):
        codes, scales, manifest = self.bind()
        unpacked = restore_checkpoint_layout(PackedExpertWeight(64, 32, codes, scales))
        self.assertTrue(np.array_equal(unpacked[0], self.blocks[0]))
        self.assertTrue(np.array_equal(unpacked[1], self.scales[0]))
        self.assertEqual(manifest["shape_mnk"], [2, 32, 64])
        self.assertEqual(manifest["activation"]["sha256"], _sha(self.activation))
        self.assertEqual(manifest["b_codes_sha256"], _sha(codes))
        self.assertIn("No direct BF16", manifest["execution_refusal"])

    def test_v3_profile_admits_direct_fp4_tile(self):
        profile = {
            "schema": "mx_gemmini.target_profile.v3",
            "legal_compute": [{
                "activation_format": "fp4_e2m1", "weight_format": "fp4_e2m1",
                "activation_projection": "direct", "weight_projection": "direct",
                "pe_mode": 0,
            }],
        }
        raw = (json.dumps(self.receipt, sort_keys=True) + "\n").encode()
        codes, scales, manifest = bind_native_expert_tile(
            self.receipt, raw, self.root, projection="gate_up", expert_index=0,
            output_start=0, output_count=32, block_start=0, block_count=2,
            activation_bf16=self.activation, activation_m=2,
            expected_revision="a" * 40, profile=profile)
        self.assertEqual(manifest["hardware_mode"], profile["legal_compute"][0])
        self.assertEqual(manifest["b_codes_sha256"], _sha(codes))
        self.assertEqual(manifest["b_scales_sha256"], _sha(scales))

    def test_rejects_modified_extracted_payload(self):
        path = self.root / "blocks.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "differs from selected source"):
            self.bind()

    def test_rejects_forged_source_range(self):
        self.receipt["tensors"][0]["source_byte_range"] = [100, 101]
        with self.assertRaisesRegex(ValueError, "invalid inclusive source range"):
            self.bind()

    def test_rejects_wrong_activation_geometry(self):
        self.activation = b"\x00" * 8
        with self.assertRaisesRegex(ValueError, "M x selected K"):
            self.bind()

    def test_rejects_unpinned_revision(self):
        self.receipt["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "differs from pinned"):
            self.bind()

    def test_rejects_fixture_path_escape(self):
        self.receipt["tensors"][0]["file"] = "../blocks.safetensors"
        with self.assertRaisesRegex(ValueError, "no extracted file"):
            self.bind()

    def test_cli_writes_bound_operands(self):
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(self.receipt) + "\n")
        a_path = self.root / "input_a.bin"
        a_path.write_bytes(self.activation)
        output = self.root / "out"
        command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                                    "tools/bind_gptoss_native_tile.py"),
                   "--fixture-dir", str(self.root), "--extractor-receipt", str(receipt_path),
                   "--expected-revision", "a" * 40, "--activation-bf16", str(a_path),
                   "--activation-m", "2", "--projection", "gate_up", "--expert-index", "0",
                   "--output-start", "0", "--output-count", "32", "--block-start", "0",
                   "--block-count", "2", "--output-dir", str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual((output / "a_bf16.bin").read_bytes(), self.activation)
        self.assertEqual(_sha((output / "b_codes.bin").read_bytes()),
                         manifest["b_codes_sha256"])


if __name__ == "__main__":
    unittest.main()
