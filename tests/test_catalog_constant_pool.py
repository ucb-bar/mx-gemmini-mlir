"""Native constant pooling closes all readers and preserves partial matrix outputs."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mx_gemmini_support.catalog_constant_pool import prepare_shared_catalog_constants
from mx_gemmini_support.dense_contraction import emit_dense_contraction
from mx_gemmini_support.target_profile import load_profile, profile_sha256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def native_catalogs(tmp_path):
    root = os.environ.get("MERLIN_TEST_RISCV_ROOT")
    readelf = os.environ.get("MERLIN_READELF")
    if not root or not readelf:
        pytest.skip("explicit RISC-V toolchain and readelf required")
    root, readelf = Path(root), Path(readelf)
    cc, ld = (root / "bin" / name for name in ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-ld"))
    profile, builds = load_profile(PROFILE), {}
    for group, m in [("alpha", 5), ("beta", 16)]:
        build = tmp_path / group
        directory = build / "device_catalog"
        directory.mkdir(parents=True)
        builds[group] = build
        symbol, issuer = "operation_" + group, "issue_" + group
        generated = emit_dense_contraction(
            profile, (m, 17, 35), precision="FP8", symbol=symbol, issuer_symbol=issuer,
            shared_workspace_symbol="mx_catalog_workspace",
            prepacked_weight_symbols=(symbol + "_fixed_codes", symbol + "_fixed_scales"), schedule="native")
        n, k = generated.issuer.shape_mnk[1:]
        flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding",
                 "-fno-fast-math", "-fno-common", "-fno-tree-loop-distribute-patterns", "-std=gnu11"]
        shutil.copyfile(ROOT / "mx_gemmini_support/runtime_quant.h", directory / "runtime_quant.h")
        sources = {symbol: generated.kernel.source, issuer: generated.issuer.source}
        components = []
        for name, text in sources.items():
            path = directory / (name + ".c")
            path.write_text(text)
            obj = path.with_suffix(".o")
            subprocess.run([cc, *flags, "-c", path, "-o", obj], check=True, capture_output=True)
            components.append(obj)
        record = {"symbol": symbol, "batched": False, "prepacked_checkpoint_weight": True,
                  "cache_immutable_weight": False, "issuer_mode": "merlin_mk_native_dram_loop",
                  "schedule_file": issuer + ".mk", "dimensions": {"m": m, "n": 17, "k": 35},
                  "precision": "mxfp8", "padded_shape_mnk": list(generated.issuer.shape_mnk)}
        assembly = [".section .rodata", ".balign 64"]
        for kind, raw in [("codes", bytes([0x38]) * (k * n)), ("scales", bytes([127]) * (k * n // 32))]:
            path = directory / (symbol + "_" + kind + ".bin")
            path.write_bytes(raw)
            record[kind + "_sha256"], record[kind + "_bytes"] = sha(path), len(raw)
            name = symbol + "_fixed_" + kind
            assembly.extend([".global " + name, name + ":", f'.incbin "{path}"', ".balign 64"])
        path = directory / "checkpoint_weights.S"
        path.write_text("\n".join(assembly) + "\n")
        subprocess.run([cc, "-march=rv64gc", "-mabi=lp64d", "-c", path, "-o", path.with_suffix(".o")],
                       check=True, capture_output=True)
        subprocess.run([ld, "-r", "-o", directory / "mx_catalog.o", *components, path.with_suffix(".o")],
                       check=True, capture_output=True)
        catalog = {"kernels": [record], "profile_sha256": profile_sha256(profile),
                   "native_tile_policy": "legacy", "virtual_zero_padding": False,
                   "compilation": {"compiler": str(cc), "flags": flags,
                       "object_sha256": sha(directory / "mx_catalog.o"),
                       "source_sha256": {name: sha(directory / (name + ".c")) for name in sources}}}
        (directory / "mx_catalog.json").write_text(json.dumps(catalog) + "\n")
    return builds, profile, root, readelf


def prepare(tmp_path, values):
    builds, profile, root, readelf = values
    return prepare_shared_catalog_constants(builds, profile=profile, output=tmp_path / "pool",
                                            riscv_root=root, readelf=readelf)


def test_native_owned_readers_and_tail_shapes_share_only_complete_packed_values(tmp_path, native_catalogs):
    code, obj, maps, record = prepare(tmp_path, native_catalogs)
    assert set(code) == {"alpha", "beta"} and obj.is_file()
    assert len(record["pool"]["blobs"]) == 2 and len(record["pool"]["bindings"]) == 4
    assert record["pool"]["original_bytes"] == 2 * record["pool"]["pooled_bytes"]
    assert set(maps["alpha"].values()) == set(maps["beta"].values())
    assert all(len(row["reader_proofs"]) == 2 for row in record["groups"].values())


def test_changed_derived_source_refuses_before_redirecting_uses(tmp_path, native_catalogs):
    builds = native_catalogs[0]
    path = builds["alpha"] / "device_catalog/operation_alpha.c"
    path.write_text(path.read_text() + "\n/* modified */\n")
    with pytest.raises(ValueError, match="owned reader source"):
        prepare(tmp_path, native_catalogs)


def test_changed_packed_bytes_refuse_even_when_size_is_unchanged(tmp_path, native_catalogs):
    path = native_catalogs[0]["alpha"] / "device_catalog/operation_alpha_codes.bin"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="packed input differs"):
        prepare(tmp_path, native_catalogs)


def test_no_redundant_storage_keeps_original_route(tmp_path, native_catalogs):
    builds, profile, root, readelf = native_catalogs
    with pytest.raises(ValueError, match="no storage reduction"):
        prepare(tmp_path, ({"alpha": builds["alpha"]}, profile, root, readelf))


def test_unowned_reader_header_cannot_interpose_address_observations(tmp_path, native_catalogs):
    header = native_catalogs[0]["alpha"] / "device_catalog/runtime_quant.h"
    header.write_text(header.read_text() + "\n/* changed ABI input */\n")
    with pytest.raises(ValueError, match="header differs from the owned ABI"):
        prepare(tmp_path, native_catalogs)


def test_saved_compiler_options_are_not_executed_as_proof_authority(tmp_path, native_catalogs):
    path = native_catalogs[0]["alpha"] / "device_catalog/mx_catalog.json"
    catalog = json.loads(path.read_text())
    catalog["compilation"]["flags"] = ["-include", "/nonexistent/unowned-header.h"]
    path.write_text(json.dumps(catalog) + "\n")
    # Exact fresh object equality under the owned options, rather than trusting
    # old command strings, establishes the original reader's machine code.
    assert prepare(tmp_path, native_catalogs)[3]["groups"]["alpha"]["reader_proofs"]


def test_extra_compiled_address_observer_refuses_complete_use_closure(tmp_path, native_catalogs):
    builds, profile, root, readelf = native_catalogs
    directory = builds["alpha"] / "device_catalog"
    path = directory / "observer.c"
    path.write_text("#include <stdint.h>\nextern const unsigned char operation_alpha_fixed_codes[];\n"
                    "uintptr_t observe(void){return (uintptr_t)operation_alpha_fixed_codes;}\n")
    cc, ld = (root / "bin" / name for name in ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-ld"))
    subprocess.run([cc, "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-c", path,
                    "-o", path.with_suffix(".o")], check=True, capture_output=True)
    subprocess.run([ld, "-r", "-o", directory / "mx_catalog.o", directory / "operation_alpha.o",
                    directory / "issue_alpha.o", path.with_suffix(".o"), directory / "checkpoint_weights.o"],
                   check=True, capture_output=True)
    catalog = json.loads((directory / "mx_catalog.json").read_text())
    catalog["compilation"]["object_sha256"] = sha(directory / "mx_catalog.o")
    (directory / "mx_catalog.json").write_text(json.dumps(catalog) + "\n")
    with pytest.raises(ValueError, match="unqualified address observer"):
        prepare(tmp_path, (builds, profile, root, readelf))


def test_model_object_cannot_observe_pooled_constant_addresses(tmp_path, native_catalogs):
    builds = native_catalogs[0]
    shutil.copyfile(builds["alpha"] / "device_catalog/operation_alpha.o", builds["alpha"] / "model.o")
    with pytest.raises(ValueError, match="unqualified address observer"):
        prepare(tmp_path, native_catalogs)


@pytest.mark.parametrize("foreign_observer", [False, True])
def test_session_namespace_rewrites_constant_imports_and_refuses_name_capture(
        tmp_path, native_catalogs, foreign_observer):
    from tools.link_merlin_session import _namespace_stage, _symbol_names

    builds, profile, root, readelf = native_catalogs
    code, pool, maps, record = prepare(tmp_path, native_catalogs)
    canonical = next(iter(maps["alpha"].values()))
    text = "void forward(void){}\nvoid _mlir_ciface_forward(void){}\n"
    if foreign_observer:
        text += f"extern const unsigned char {canonical}[];\nconst void *foreign(void){{return {canonical};}}\n"
    source = tmp_path / "model.c"
    source.write_text(text)
    cc, nm, objcopy = (root / "bin" / name for name in
                       ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-nm", "riscv64-unknown-elf-objcopy"))
    subprocess.run([cc, "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-c", source,
                    "-o", source.with_suffix(".o")], check=True, capture_output=True)
    kwargs = dict(index=0, name="alpha", objects=[source.with_suffix(".o"), code["alpha"]], previous=set(),
                  nm=nm, objcopy=objcopy, output=tmp_path, external_symbols=maps["alpha"])
    if foreign_observer:
        with pytest.raises(ValueError, match="conflicts with stage definitions"):
            _namespace_stage(**kwargs)
    else:
        objects, symbols = _namespace_stage(**kwargs)
        names = set(_symbol_names(nm, objects[-1]))
        assert set(maps["alpha"].values()) <= names
        assert not set(maps["alpha"]) & names


def test_two_real_admitted_images_preserve_three_changed_partial_tile_calls(tmp_path, native_catalogs):
    startup, extension = os.environ.get("MERLIN_TEST_STARTUP"), os.environ.get("MERLIN_TEST_EXTLIB")
    if not startup or not extension:
        pytest.skip("explicit startup and selected simulator extension required")
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service

    builds, profile, root, readelf = native_catalogs
    code, pool, maps, record = prepare(tmp_path, native_catalogs)
    cc, objcopy = (root / "bin" / name for name in ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-objcopy"))
    pooled = []
    for group, source in code.items():
        mapping = tmp_path / (group + ".symbols")
        mapping.write_text("".join(f"{old} {new}\n" for old, new in maps[group].items()))
        destination = tmp_path / (group + ".o")
        subprocess.run([objcopy, "--redefine-syms", mapping, source, destination], check=True, capture_output=True)
        pooled.append(destination)
    driver = tmp_path / "driver.c"
    driver.write_text('''#include <stdint.h>
#include <stdio.h>
void operation_alpha(const float*,const float*,float*);
void operation_beta(const float*,const float*,float*);
unsigned char mx_catalog_workspace[65536] __attribute__((aligned(64)));
static float input[560] __attribute__((aligned(64)));
static struct {uint32_t pre[8];float data[85];uint32_t post[8];} a;
static struct {uint32_t pre[8];float data[272];uint32_t post[8];} b;
int main(void){unsigned errors=0;
  for(unsigned run=0;run<3;++run){
    float value=run==0?1.0f:run==1?-1.0f:0.5f;
    for(unsigned i=0;i<560;++i)input[i]=value;
    for(unsigned i=0;i<85;++i)a.data[i]=0.0f;
    for(unsigned i=0;i<272;++i)b.data[i]=0.0f;
    for(unsigned i=0;i<8;++i)a.pre[i]=a.post[i]=b.pre[i]=b.post[i]=0xa55aa55a;
    operation_alpha(input,input,a.data); operation_beta(input,input,b.data);
    for(unsigned i=0;i<85;++i)errors+=a.data[i]!=35.0f*value;
    for(unsigned i=0;i<272;++i)errors+=b.data[i]!=35.0f*value;
    for(unsigned i=0;i<8;++i)errors+=a.pre[i]!=0xa55aa55a || a.post[i]!=0xa55aa55a ||
      b.pre[i]!=0xa55aa55a || b.post[i]!=0xa55aa55a;
    printf("VALUES alpha run=%u ",run);
    for(unsigned i=0;i<85;++i){union{float f;uint32_t u;} v;v.f=a.data[i];printf("%08x",v.u);}
    printf("\\nVALUES beta run=%u ",run);
    for(unsigned i=0;i<272;++i){union{float f;uint32_t u;} v;v.f=b.data[i];printf("%08x",v.u);}
    printf("\\n");
    printf("POOL run=%u errors=%u\\n",run,errors);
  }
  printf("DONE errors=%u\\n",errors);return errors!=0;
}
''')
    startup = Path(startup)
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-mabi=lp64d",
             "-I", str(startup), "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    common = []
    for index, source in enumerate([driver, ROOT / "mx_gemmini_support/runtime_quant.c",
                                    *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S"))]):
        obj = tmp_path / f"common{index}.o"
        subprocess.run([cc, *flags, "-c", source, "-o", obj], check=True, capture_output=True)
        common.append(obj)
    route = DeviceRouting(device="mx_gemmini", package_dir=ROOT, operand_dtype="f32", accum_dtype="bf16",
                          linked_elf_admission=service("mx_gemmini"))
    for label, objects in [("original", [b / "device_catalog/mx_catalog.o" for b in builds.values()]),
                            ("pooled", [*pooled, pool])]:
        directory = tmp_path / label
        directory.mkdir()
        elf = directory / "section.elf"
        subprocess.run([cc, "-nostdlib", "-nostartfiles", "-static", "-T", startup / "test.ld",
                        *objects, *common, "-lm", "-lgcc", "-o", elf], check=True, capture_output=True)
        admit_linked_device_elf(route, freeze_device_elf_admission(route), elf=elf,
                                linked_sha256=sha(elf), directory=directory)
        result = subprocess.run([root / "bin/spike", "--isa=rv64gc_zicntr", "--extlib=" + extension,
                                 "--extension=gemmini", elf], check=True, capture_output=True, text=True,
                                env=os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                                  "GEMMINI_PERF_SET": "mem.host_tracking=0",
                                                  "GEMMINI_PERF_OUT": str(directory / "perf.txt")}, timeout=60)
        (directory / "spike.log").write_text(result.stdout + result.stderr)
        assert result.stdout.splitlines().count("DONE errors=0") == 1
        assert [line for line in result.stdout.splitlines() if line.startswith("POOL ")] == [
            f"POOL run={index} errors=0" for index in range(3)]
        expected_words = ["420c0000", "c20c0000", "418c0000"]
        values = [line.split() for line in result.stdout.splitlines() if line.startswith("VALUES ")]
        assert len(values) == 6
        for index, parts in enumerate(values):
            stage, count = ("alpha", 85) if index % 2 == 0 else ("beta", 272)
            assert parts[:3] == ["VALUES", stage, f"run={index // 2}"]
            assert parts[3] == expected_words[index // 2] * count
        first, mesh, traffic = [line.split() for line in (directory / "perf.txt").read_text().splitlines()[:3]]
        assert (int(first[5]), int(mesh[1]), int(mesh[3]), int(mesh[7])) == (6, 48, 768, 0)
        assert int(traffic[1]) == 18432 and int(traffic[traffic.index("scales:") + 1]) == 576
