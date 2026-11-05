# Captured FP6 stage annotation

`mx_gemmini.execution_policy.v3` records a recipe-selected FP6 E3M2 matrix
site at the original captured source. `matrix_formats[region] = "mxfp6"`
implies LUT projection on both operands. `matrix_lut_groups[region]` records
the user's `G` selection, and `run_recipe_sha256` binds the complete model-run
recipe that selected the quantizer and placement. Every matrix and VPU source
site still appears in the policy; admission checks the exact source/profile
hashes, a source-eligible dense site, and a legal FP6 E3M2/LUT compute tuple.

The npu bridge emits this annotation only after validating the whole model-run
plan and its selected VPU rewrite. The captured-site runner verifies the
annotation before compiling its selected Spike guest. The annotation does not
bind LUT values to Merlin's whole-stage catalog. `build_merlin_stage.py`
explicitly refuses a v3 policy containing FP6 until that catalog lowering is
implemented and qualified. Physical LUT allocation is likewise unqualified.

Older v1/v2 policies retain their FP8/FP4 behavior and schemas.
