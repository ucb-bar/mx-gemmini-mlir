"""Generate the reference implementation’s TinyLlama source blob with one captured input and weights.

The pinned source generator expects a directory of per-layer ``.npz`` files.
This adapter combines checkpoint tensors from a weight capture with the
activations of the exact token sequence used by a model2MLIR bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-capture", type=Path, required=True)
    parser.add_argument("--activation-capture", type=Path, required=True)
    parser.add_argument("--source-generator", type=Path, required=True,
                        help="pinned npu-exploration gen_llama_model.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="_m160")
    args = parser.parse_args()
    weights = args.weights_capture.resolve(strict=True)
    activations = args.activation_capture.resolve(strict=True)
    generator = args.source_generator.resolve(strict=True)
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("--out-dir must be fresh")
    if not all((weights / f"layer{i}.npz").is_file() for i in range(22)):
        parser.error("weight capture needs all 22 layer files")
    if not (weights / "model.npz").is_file() or not (activations / "model.npz").is_file():
        parser.error("both captures need model.npz")

    out.mkdir(parents=True)
    sys.path.insert(0, str(generator.parent))
    import gen_llama_model as gen  # noqa: PLC0415

    if Path(gen.__file__).resolve() != generator:
        raise RuntimeError("loaded a different Nicolas source generator")

    def merged(name: str) -> dict[str, np.ndarray]:
        with np.load(weights / name, allow_pickle=False) as captured:
            data = {key: captured[key] for key in captured.files}
        with np.load(activations / name, allow_pickle=False) as captured:
            data.update({key: captured[key] for key in captured.files})
        return data

    gen.CAPTURE = weights
    gen.DATA = out
    gen.load_layer = lambda layer: merged(f"layer{layer}.npz")
    gen.load_model = lambda: merged("model.npz")
    sys.argv = [str(generator), "--tag", args.tag]
    raise SystemExit(gen.main())


if __name__ == "__main__":
    main()
