# Guarded host library-call specialization

This evidence covers host exponential calls, not a softmax kernel or a complete
model. The selected emitted wrapper preserves the original exponential routine
and retains the original public entry point as fallback. A full TinyLlama forward
with this option is running separately; no new whole-model result is claimed.

The measured host normalization cost motivated inspection of the actual linked
`expf`. Its wrapper first computes `__ieee754_expf`, then checks finiteness and
range, with errno handling for exceptional finite inputs. The reviewed library's
two finite thresholds strictly contain [-80,80]. Within that interval the public
wrapper returns the unchanged internal result. Its finite checks add no errno
or floating effects for these inputs. For negative infinity the reviewed
implementation returns positive zero without changing errno or flags.

The selected library is SHA256
`38965f2df6e94d3389596fcf414caf37040294fc0f43f3afa3e4369acb837934`.
Its complete regular archive, actual wrapper/body/finite-check objects, relocations,
constants and disassemblies are retained. The provider's equivalence review is
specific to these bytes and bindings. Finite sample checks alone do not prove
equivalence for another library or establish an approximation permission.

Merlin's explicit `guarded_f32` host policy generates integer bit guards. A typed
provider contract specifies a finite magnitude interval, optional exact points,
two endpoint symbols and the reviewed archive digest. The ordinary build checks
the selected archive and automatically records both endpoints' defining link
suppliers. Other inputs call the original endpoint. The default native policy
still emits no object or guard flags. MX's `--host-math-guard guard.json` exposes
this existing generic build seam; no model or provenance ID selects behavior.

For this experiment, the interval is [-80,80], and the only exact point is
`exp(-infinity)=+0`. NaNs and inputs outside the interval/point retain the
original library call. This preserves all-masked softmax NaN behavior: its
subtraction produces NaN, which takes fallback.

Each timing loop performs 32,768 calls, with matching output stores and hashes.
Complete output comparison happens after the timer. Values/errno/flags are also
compared separately for all 65,536 BF16 bit patterns and 65,536 xorshift-generated
F32 words in each of the five supported RISC-V rounding modes: 655,360 paired
comparisons per native image. These include nonfinite encodings and signed zeros.

| Distribution | Original cycles | Selected emitted wrapper | Change |
|---|---:|---:|---:|
| Finite values from −15.99609375 through −0 | 3,837,325 | 3,214,733 | −16.2% |
| Half the finite sequence, half negative infinity | 2,623,013 | 1,869,349 | −28.7% |
| All negative infinity | 1,409,053 | 524,317 | −62.8% |
| All −128, original fallback | 2,883,613 | 3,178,525 | **+10.2%** |

All three repetitions have identical counts. The fallback-heavy regression is
retained, so this remains an explicit experiment rather than a general default.
Actual model benefit and its fallback distribution remain separate questions.
The selected wrapper is 40 text bytes in this section ELF; it introduces no heap
allocation. No whole-program peak-stack or host-memory-traffic claim is made.
Host tracking is off, as in the matched full-model runs. Native counters report
zero accelerator commands and zero accelerator transfer bytes for these sections.

Two earlier inline probes are retained as ablations. A magnitude guard alone
saved 20.5% for finite inputs but regressed all negative infinity by 9.3%.
Adding the exact point removed that regression. The table uses the actual emitted
out-of-line wrapper, including its call and guard overhead; the faster inline
prototype figures are not substituted for production-wrapper measurements.

`executions.tar.xz` contains three actual executed ELFs, their complete native
logs and structural admission receipts, source and producer snapshots, commands,
environment, and the selected library. The independent artifact check verifies
every member, reopens ELF symbols and their function bytes, re-extracts reviewed
library objects from the retained archive, checks finite-check code identity and
the wrapper's retained range constants, and checks all native comparison records.
The control-flow equivalence review remains a provider obligation; the verifier
does not turn sample tests into a universal theorem. Startup and transitive
toolchain headers remain external, so this is not a hermetic build closure.

```sh
python docs/evidence/guarded_host_math_20261019/verify.py
```

Python and NumPy are needed for the digest-pinned ELF reader reused from the
adjacent broadcast evidence. No compiler code is imported by the artifact check.

To reproduce with an unused task output directory:

```sh
export PYTHONPATH="$PWD/../merlin/src:$PWD/../model2mlir:$PWD"
export MERLIN_READELF=/path/to/llvm-install/bin/llvm-readelf
python -m tools.qualify_guarded_host_math_spike \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --guard docs/evidence/guarded_host_math_20261019/guard.json \
  --riscv-root /path/to/riscv-tools \
  --extlib docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so \
  --startup-dir /path/to/gemmini-rocc-tests/riscv-tests/benchmarks/common \
  --out-dir "$TMPDIR/guarded-math-repeat" --structural-elf-admission
```

The pinned archive must match. Changing its digest requires a fresh provider
review and numerical/effect qualification, not editing this witness to force a
build through. Complete logits and the unchanged existing model quality gate
remain required for any full-forward result.
