"""Capture the 64×2,048×64 causal attention graph through model2MLIR."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
FRONTEND_SHA256 = "56099369c7eb23486170f0d092651fcd7c7b1e608fe8f14ed2f4bee13a5b4fb9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = args.model2mlir_root.resolve()
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=source, text=True).strip()
        if revision != MODEL2MLIR_REVISION:
            raise ValueError("causal frontend needs the pinned model2MLIR revision")
    sys.path.insert(0, str(source))
    import m2m
    from m2m.coverage import opaque_report
    import torch

    if Path(m2m.__file__).resolve().parents[1] != source:
        raise ValueError("model2MLIR imported from another source tree")

    class Attention(torch.nn.Module):
        def forward(self, q, kt, mask, v):
            return torch.bmm(torch.softmax(torch.bmm(q, kt) * 0.125 + mask,
                                           dim=-1), v)

    q = torch.zeros((1, 64, 64), dtype=torch.float32)
    kt = torch.zeros((1, 64, 2048), dtype=torch.float32)
    mask = torch.zeros((1, 64, 2048), dtype=torch.float32)
    mask[:, :, -64:] = torch.triu(torch.full((64, 64), float("-inf")), diagonal=1)
    v = torch.zeros((1, 2048, 64), dtype=torch.float32)
    capture = m2m.convert(Attention().eval(), (q, kt, mask, v),
                          backend="fx_importer")
    if (not capture.ok or capture.path_taken != "fx_importer" or
            opaque_report(capture.mlir_text)):
        raise ValueError("model2MLIR causal capture is opaque or failed")
    digest = hashlib.sha256(capture.mlir_text.encode()).hexdigest()
    if digest != FRONTEND_SHA256:
        raise ValueError(f"model2MLIR causal frontend changed: {digest}")
    args.out.write_text(capture.mlir_text)
    print(f"causal attention model2MLIR capture: {args.out}")


if __name__ == "__main__":
    main()
