"""Run the model-derived TinyLlama MX issuer twice with distinct live pointers.

This checks the encoded-pointer ABI on the reference implementation’s Spike, not a whole model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.runtime_issuer import emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from tools.compile_mx import _require_gitlink


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_projection_a042643_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


DRIVER = r'''#include <stdint.h>
#include <stdio.h>
extern const uint8_t activation[], activation_scales[], weight[], weight_scales[], golden_bf16[];
void mx_issue_encoded(const void *, const void *, const void *, const void *, const void *, const void *);
static uint8_t a2[32768] __attribute__((aligned(64)));
static uint8_t as2[1024] __attribute__((aligned(64)));
static uint8_t b2[65536] __attribute__((aligned(64)));
static uint8_t bs2[2048] __attribute__((aligned(64)));
static uint16_t c1[512] __attribute__((aligned(64)));
static uint16_t c2[512] __attribute__((aligned(64)));
static uint8_t scratch1[2048] __attribute__((aligned(64)));
static uint8_t scratch2[2048] __attribute__((aligned(64)));
int main(void) {
  for (unsigned i = 0; i < sizeof a2; ++i) a2[i] = activation[i];
  for (unsigned i = 0; i < sizeof as2; ++i) as2[i] = activation_scales[i];
  for (unsigned i = 0; i < sizeof b2; ++i) b2[i] = weight[i];
  for (unsigned i = 0; i < sizeof bs2; ++i) bs2[i] = weight_scales[i];
  mx_issue_encoded(activation, activation_scales, weight, weight_scales, c1, scratch1);
  mx_issue_encoded(a2, as2, b2, bs2, c2, scratch2);
  unsigned errors = 0;
  const uint16_t *golden = (const uint16_t *)golden_bf16;
  for (unsigned i = 0; i < 512; ++i)
    if (c1[i] != golden[i] || c2[i] != golden[i] || c1[i] != c2[i]) ++errors;
  printf("live MX issuer two addresses: %u BF16 mismatches\n", errors);
  return errors != 0;
}
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, out: Path, log: str) -> None:
    result = subprocess.run(command, cwd=out, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see {out / log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rtl, riscv, out = (getattr(args, key).resolve()
                       for key in ("rtl_root", "riscv_root", "out_dir"))
    if out.exists():
        parser.error("output directory already exists")
    software, extension = rtl / "software/gemmini-rocc-tests", rtl / "software/libgemmini"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(PROFILE, rtl_root=rtl)
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    program = lower_bound_source((ARCHIVE / "projection.payload_bound.mlir").read_text(),
                                 profile, manifest, resources)
    issuer = emit_runtime_issuer(program, manifest)
    if issuer.shape_mnk != (16, 32, 2048) or issuer.precision != "FP8":
        raise ValueError("archived TinyLlama projection shape or precision changed")
    write_standalone_sources(out, program, resources)
    # Reuse only its byte-bound data assembly; the standalone one-call driver
    # and receipt no longer describe this two-call executable.
    (out / "artifact_manifest.json").unlink()
    (out / "mx_issue.c").write_text(issuer.source)
    (out / "mx_driver.c").write_text(DRIVER)
    bench = software / "riscv-tests/benchmarks/common"
    cc = riscv / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc", "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"), "-I", str(software), "-I", str(bench)]
    sources = [out / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, source in enumerate(sources):
        obj = out / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             out=out, log=f"compile_{index}.log")
        objects.append(obj)
    elf = out / "live.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
          *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)],
         out=out, log="link.log")
    so = out / "libgemmini.so"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    _run(["g++", "-L", str(riscv / "lib"), f"-Wl,-rpath,{riscv / 'lib'}",
          "-shared", "-o", str(so), "-std=c++17", "-I", str(riscv / "include"),
          "-fPIC", "-O3", *(str(source) for source in extension_sources)],
         out=out, log="extension_build.log")
    spike = subprocess.run([str(riscv / "bin/spike"), f"--extlib={so}",
                            "--extension=gemmini", str(elf)], cwd=out, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "spike.log").write_text(spike.stdout)
    passed = (spike.returncode == 0 and
              "live MX issuer two addresses: 0 BF16 mismatches" in spike.stdout)
    receipt = {
        "schema": "mx_gemmini.runtime_issuer_spike_diagnostic.v1",
        "status": "two_live_encoded_pointer_calls_matched" if passed else "failed",
        "scope": "one archived TinyLlama projection, two distinct runtime buffer addresses",
        "compared_bf16_outputs": 1024,
        "profile_sha256": issuer.profile_sha256,
        "payload_manifest_sha256": issuer.payload_manifest_sha256,
        "issuer_sha256": _sha(out / "mx_issue.c"),
        "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(out / "spike.log"),
        "spike_exit_code": spike.returncode,
    }
    (out / "runtime_issuer_spike.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"live MX issuer mismatch; see {out / 'spike.log'}")


if __name__ == "__main__":
    main()
