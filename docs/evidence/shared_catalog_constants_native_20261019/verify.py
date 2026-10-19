"""Verify actual native constant sharing, ownership receipts and full outputs."""

import hashlib
import io
import json
import struct
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pool_contents(raw):
    assert raw[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", raw, 40)[0]
    size, count, names = struct.unpack_from("<HHH", raw, 58)
    assert size == 64 and 0 < names < count
    rows = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + i * size) for i in range(count)]
    strings = raw[rows[names][4]:rows[names][4] + rows[names][5]]
    named = {strings[r[0]:strings.index(b"\0", r[0])].decode(): i for i, r in enumerate(rows)}
    index = named[".rodata"]
    data = rows[index]
    assert data[1] == 1 and data[2] == 2 and data[8] == 64
    assert not any(r[1] in (4, 9) and r[5] for r in rows)
    assert not any(r[2] & 2 and r[5] and i != index for i, r in enumerate(rows))
    table = rows[named[".symtab"]]
    string_row = rows[table[6]]
    symbols = raw[string_row[4]:string_row[4] + string_row[5]]
    exports = {}
    for pos in range(table[4], table[4] + table[5], 24):
        name, info, other, section, value, length = struct.unpack_from("<IBBHQQ", raw, pos)
        if info >> 4:
            assert info >> 4 == 1 and section == index and not other
            text = symbols[name:symbols.index(b"\0", name)].decode()
            assert text not in exports
            exports[text] = (value, length)
    return raw[data[4]:data[4] + data[5]], exports


def main():
    manifest = json.loads((HERE / "artifacts.json").read_bytes())
    raw = (HERE / "qualifiers.tar.xz").read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, value in files.items():
        assert {"bytes": len(value), "sha256": sha(value)} == manifest["members"][name]

    def load(name):
        return json.loads(files[name])

    proof = load("native/pool/receipt.json")
    plan = proof["pool"]
    assert plan["original_bytes"] == 4224 and plan["pooled_bytes"] == 2112
    assert len(plan["bindings"]) == 4 and len(plan["blobs"]) == 2
    for group, metadata in proof["groups"].items():
        assert len(metadata["reader_proofs"]) == 2
        assert metadata["original_object_sha256"] == sha(files[f"native/{group}/device_catalog/mx_catalog.o"])
        assert metadata["code_sha256"] == sha(files[f"native/pool/{group}/code.o"])
        for name, record in metadata["reader_proofs"].items():
            assert sha(files[f"native/{group}/device_catalog/{name}.c"]) == record["source_sha256"]
            assert sha(files[f"native/{group}/device_catalog/{name}.o"]) == record["object_sha256"]
            assert files[f"native/pool/{group}/{name}.o"] == files[f"native/{group}/device_catalog/{name}.o"]
        assert files[f"native/pool/{group}/original.o"] == files[f"native/{group}/device_catalog/mx_catalog.o"]
    for binding in plan["bindings"]:
        assert binding["access"] == "readonly_bytes_address_unobserved"
        assert binding["alignment"] == 64
        kind = "codes" if binding["symbol"].endswith("_codes") else "scales"
        name = binding["symbol"].replace("_fixed_", "_")
        content = files[f"native/{binding['group']}/device_catalog/{name}.bin"]
        assert len(content) == binding["bytes"] and sha(content) == binding["sha256"]
        assert set(content) == ({0x38} if kind == "codes" else {127})
    for path, digest in proof["producer_and_tool_pins"].items():
        name = PurePosixPath(path).name
        candidate = next((files[prefix + name] for prefix in ("frozen_target/", "frozen_core/")
                          if prefix + name in files), None)
        if name == "runtime_quant.h":
            candidate = files["native/pool/alpha/runtime_quant.h"]
        if candidate is not None:
            assert sha(candidate) == digest
    assert all(row["exit_code"] == 0 for row in proof["commands"])
    assert proof["pool_object_sha256"] == sha(files["native/pool/shared_constants.o"])
    assert files["native/pool/shared_constants.o"] == files["native/pool/verified_shared_constants.o"]
    payload, exports = pool_contents(files["native/pool/shared_constants.o"])
    expected_payload, cursor = bytearray(), 0
    assert set(exports) == {row["symbol"] for row in plan["blobs"]}
    for blob in plan["blobs"]:
        aligned = (cursor + blob["alignment"] - 1) // blob["alignment"] * blob["alignment"]
        expected_payload.extend(bytes(aligned - cursor))
        assert exports[blob["symbol"]] in [(aligned, 0), (aligned, blob["bytes"])]
        representative = next(b for b in plan["bindings"] if b["canonical_symbol"] == blob["symbol"])
        name = representative["symbol"].replace("_fixed_", "_")
        expected_payload.extend(files[f"native/{representative['group']}/device_catalog/{name}.bin"])
        cursor = len(expected_payload)
    expected_payload.extend(bytes((-len(expected_payload)) % plan["blobs"][-1]["alignment"]))
    assert payload == expected_payload and len(payload) == plan["pooled_bytes"]
    observed, receipts = {}, {}
    for role in ("original", "pooled"):
        prefix = "replay/" + role + "/"
        receipt = load(prefix + "invocation.json")
        receipts[role] = receipt
        elf = files[prefix + "section.elf"]
        assert elf == files[f"native/{role}/section.elf"]
        assert sha(elf) == receipt["elf_sha256"] == receipt["admission"]["elf_sha256"]
        assert receipt["admission"]["status"] == "accepted" and receipt["exit_code"] == 0
        report = prefix + "invocations/" + receipt["admission"]["report_path"].split("/invocations/", 1)[1]
        assert sha(files[report]) == receipt["admission"]["report_sha256"]
        assert receipt["scope"] == "complete_native_correctness_replay_no_section_timer"
        assert receipt["environment"]["GEMMINI_MODE"] == "both"
        assert receipt["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert receipt["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        assert receipt["command"][1] == "--isa=rv64gc_zicntr" and receipt["command"][3] == "--extension=gemmini"
        assert receipt["command"][2].endswith(
            "/docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so")
        console = files[prefix + "spike.log"]
        assert sha(console) == receipt["log_sha256"] and console.endswith(b"\n")
        lines = console.decode().splitlines()
        assert lines.count("DONE errors=0") == 1
        assert [line for line in lines if line.startswith("POOL ")] == [f"POOL run={i} errors=0" for i in range(3)]
        values = [line.split() for line in lines if line.startswith("VALUES ")]
        assert len(values) == 6
        complete = []
        for index, parts in enumerate(values):
            group, count = ("alpha", 85) if index % 2 == 0 else ("beta", 272)
            assert parts[:3] == ["VALUES", group, f"run={index // 2}"]
            encoded = bytes.fromhex(parts[3])
            expected = struct.pack(">f", 35 * [1, -1, .5][index // 2]) * count
            assert encoded == expected and len(encoded) == count * 4
            complete.append(encoded)
        observed[role] = complete
        perf = files[prefix + "perf.txt"]
        assert sha(perf) == receipt["perf_sha256"]
        first, mesh, traffic = [line.split() for line in perf.decode().splitlines()[:3]]
        assert (int(first[5]), int(mesh[1]), int(mesh[3]), int(mesh[7])) == (6, 48, 768, 0)
        assert int(traffic[1]) == 18432 and int(traffic[traffic.index("scales:") + 1]) == 576
    assert observed["original"] == observed["pooled"]
    for key in ("spike_sha256", "extension_sha256"):
        assert receipts["original"][key] == receipts["pooled"][key]
    assert receipts["pooled"]["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
    inventory = load("input_inventory_only.json")
    assert inventory["prepacked_matrix_count"] == 155 and inventory["matched_blob_count"] == 310
    assert inventory["duplicate_payload_bytes"] == 1066745856
    print("PASS: both admitted native images, all 1071 F32 words each, three changed inputs, guards and complete work counters.")
    print("Original/pooled images: 30,416/28,192 bytes; full-model link and performance remain separate pending experiments.")


if __name__ == "__main__":
    main()
