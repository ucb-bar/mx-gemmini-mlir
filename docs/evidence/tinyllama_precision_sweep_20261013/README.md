# TinyLlama dense precision sensitivity

This diagnostic uses the captured pretrained 22-layer TinyLlama prefill
stage, its synthetic input, and its unquantized PyTorch output. The source
MLIR SHA-256 is
`8b7e6e709fd12352773b5a7b264916b3f98b9a6b901f329de04446619ef09031`;
the selected MX+VPU profile SHA-256 is
`e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d`.
All four runs quantize 44 attention matmuls and one rotary BMM to FP8.
The reference top token is 279.

| Dense policy | FP4 / FP8 sites | Logit cosine | Mean absolute error | Top token | Reference rank | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Default](fp4_control_result.json) | 155 / 0 | 0.498 | 2.167 | 800 | 40 | 1 |
| [Down projections](fp8_down_result.json) | 133 / 22 | 0.671 | 1.578 | 5481 | 943 | 3 |
| [All MLP projections](fp8_mlp_result.json) | 89 / 66 | 0.516 | 1.950 | 29889 | 26 | 2 |
| [All dense FP8](fp8_all_result.json) | 0 / 155 | 0.922 | 0.751 | 362 | 2 | 9 |

The [down-projection](fp8_down_overrides.json) and
[all-MLP](fp8_mlp_overrides.json) manifests bind exact source regions to FP8.
The [all-FP8 control](fp8_all_overrides.json) promotes every ordinary dense
site while retaining the same 45 FP8 attention and rotary contractions.
Each result records the captured weights and inputs, model2MLIR revision,
baseline and proxy output digests, and the override-file digest. The
[comparison receipt](comparison.json) validates those identities and the
saved local `baseline.npy` and `proxy.npy` arrays before computing the
cosine and error columns as float64. The arrays are not published with this
record.

Reproduce the four results with
`tools.diagnose_tinyllama_operand_quant` on the same captured prefill stage:
use `--layers 22 --capacity 160`, `--expected-fp8 44`,
`--expected-fp8-bmm 1`, and `--stage-kind prefill`. Set `--expected-fp4` to the table's
FP4 count and pass the corresponding checked-in manifest through
`--precision-overrides` and the selected profile through `--profile` for
the promoted runs. Then call `tools.compare_tinyllama_precision` with four
`--run name=/path/to/diagnostic-output` arguments to regenerate the
comparison receipt. The [model guide](../../compile_e2e_models.md#7-test-a-source-bound-fp8-promotion)
shows the manifest and stage-build commands.

The down-projection promotion reduces mean absolute logit error by 27.2%
against the default operand proxy. Promoting all 155 dense sites reduces it
by 65.3% and raises cosine to 0.922. The reference token is then rank 2 and
nine of its top ten tokens remain in the proxy top ten. This implicates FP4
operand quantization as a major source of drift for this input, but the
nonmonotone MLP-only result shows that selecting sites by layer type alone is
unreliable. None of the tested policies selects the reference top token.

The full captured prefill graph also compiled to a source-bound MX catalog
with the down-projection manifest: all 200 contractions remained on MX,
with 133 FP4 and 67 FP8 bindings, eight distinct kernels, and eight shared
issuers. The relocatable RV64 object is 501,000 bytes with SHA-256
`4f842b7fbc8584ef2ec15ff14be24850c3da1afe24cda3d93ba70de7fe124d0d`.
This checks compiler coverage and object emission for the selected policy;
the promoted whole-stage ELF has not yet been linked or run on Spike. The
all-FP8 control has only been evaluated in the operand proxy.

The diagnostic uses model2MLIR's captured model and the MX operand
fake-quantization kernel, followed by PyTorch matrix arithmetic. It does
not model Gemmini MX mesh accumulation or VPU arithmetic and is not a
Spike result or a model-quality acceptance test. A policy must be checked
on task inputs with an exact target reference and a declared quality
threshold before replacing the compiler default.
