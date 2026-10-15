"""Check the compact, public part of the same-input evidence bundle."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
identity = json.loads((HERE / "input_identity.json").read_text())
recipe = json.loads((HERE / "all_fp8_overrides.json").read_text())
build = json.loads((HERE / "compiler_build_summary.json").read_text())
with np.load(HERE / "input_ids.npz", allow_pickle=False) as data:
    ids = data["in0"].reshape(-1)
assert ids.shape == (160,)
assert hashlib.sha256(ids.tobytes()).hexdigest() == identity["token_ids_sha256"]
assert identity["source_and_compiler_token_ids_exact"] is True
assert identity["source_and_compiler_fp32_weights_exact"] is True
assert identity["matched_fp32_weight_tensors"] == 200
assert recipe["source_mlir_sha256"] == identity["compiler_mlir_sha256"]
assert build["capture_model_sha256"] == identity["compiler_mlir_sha256"]
assert recipe["profile_sha256"] == build["profile_sha256"]
assert len(recipe["regions"]) == 155
assert build["selected_mx_contractions"] == 200
assert build["all_mx_operand_pairs_e4m3"] is True
print("160 token IDs, 200 checkpoint tensors, MLIR capture, and FP8 recipe agree")
