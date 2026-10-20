"""Verify the retained recurrent source smoke without importing either compiler."""

import argparse
import hashlib
import io
import json
import lzma
import re
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    raw = (HERE / "checks.tar.xz").read_bytes()
    assert (
        sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for entry in archive:
            path = PurePosixPath(entry.name)
            assert entry.isfile() and not path.is_absolute() and ".." not in path.parts
            assert entry.name not in files
            files[entry.name] = archive.extractfile(entry).read()
    originals = frozenset(files)
    assert len(originals) == manifest["stored_members"]
    for alias, original in manifest["aliases"].items():
        assert alias not in files and original in originals
        files[alias] = files[original]
    assert set(files) == set(manifest["members"])
    for name, contents in files.items():
        record = manifest["members"][name]
        assert len(contents) == record["bytes"] and sha(contents) == record["sha256"], (
            name
        )
    port = json.loads(files["port/receipt.json"])
    replay = json.loads(files["replay/receipt.json"])
    assert (
        port["status"] == "completed_requires_matched_comparison"
        and port["mode"] == "cached_smoke"
    )
    assert (
        port["independent_host_graph"] is True
        and port["full_model_speedup_qualified"] is False
    )
    assert replay["passed"] is True and replay["full_model_comparison"] is False
    assert replay["source_receipt_sha256"] == sha(files["port/receipt.json"])
    for name, expected in port["source_files_sha256"].items():
        assert sha(files["radiance/" + name]) == expected
    for name, expected in port["startup_sha256"].items():
        assert sha(files["startup/" + Path(name).name]) == expected
    assert sha(files["port/session.elf"]) == port["elf_sha256"]
    assert sha(files["port/driver.c"]) == port["port_source_sha256"]
    assert sha(files["port/radiance_rocket_llama.c"]) == port["base_port_sha256"]
    assert sha(files["replay/vpu_ref.h"]) == replay["vpu_reference_header_sha256"]
    for name, expected in port["files_sha256"].items():
        assert sha(files["port/" + name]) == expected
    program = np.frombuffer(files["port/phases.bin"], "<u4").reshape(-1, 32)
    assert len(program) == port["phase_count"] == 26 * 34
    assert (
        port["session_steps"] == replay["session_steps"] == 34
        and port["cache_capacity"] == 97
    )
    assert np.count_nonzero(program[:, 0] == 1) == 442
    assert (
        replay["matrix_reference_calls"]
        == int(program[program[:, 0] == 1, 14].sum())
        == 578
    )
    console = files["port/spike.log"].decode()
    assert console.splitlines().count("PORT_DONE") == 1
    assert "DEADLOCK" not in console and "guards=0" in console
    assert re.findall(r"^PORT_PHASES_COMPLETED (\d+)$", console, re.MULTILINE) == [
        str(len(program))
    ]
    reset = list(
        map(int, re.findall(r"^CACHED_RESET ([1-9][0-9]*)$", console, re.MULTILINE))
    )
    steps = [
        tuple(map(int, r))
        for r in re.findall(
            r"^CACHED_STEP (\d+) (\d+) ([1-9][0-9]*)$", console, re.MULTILINE
        )
    ]
    assert len(reset) == 1 and [(s, p) for s, p, _ in steps] == [
        (s, 64 + s) for s in range(34)
    ]
    total = reset[0] + sum(c for _, _, c in steps)
    assert total == port["cycles"] == replay["cycles"] == 70078807
    assert re.findall(
        r"^PORT_TOTAL ([1-9][0-9]*) guards=0$", console, re.MULTILINE
    ) == [str(total)]
    observed = {0: bytearray(), 1: bytearray()}
    for line in console.splitlines():
        if line.startswith("PORT_BYTES "):
            _, index, offset, count, *encoded = line.split()
            index, offset, count = int(index), int(offset), int(count)
            assert (
                index in observed
                and offset == len(observed[index])
                and len(encoded) == count
            )
            body = bytes.fromhex(" ".join(encoded))
            assert len(body) == count
            observed[index].extend(body)
    for i, size in enumerate([34 * 512 * 4, 2 * 2 * 2 * 97 * 64 * 4]):
        actual, expected = (
            files[f"port/observed{i}.bin"],
            files[f"replay/expected{i}.bin"],
        )
        assert actual == expected == bytes(observed[i]) and len(actual) == size
        assert np.isfinite(np.frombuffer(actual, "<f4")).all()
        assert (
            replay["results"][i]["word_mismatches"]
            == replay["results"][i]["byte_mismatches"]
            == 0
        )
    for i in range(578):
        assert all(
            f"replay/matrix_inputs/{i}/{name}.bin" in files
            for name in ["A", "B", "SA", "SB", "C"]
        )
    assert port["accelerator_work"]["passed"] is True
    assert port["environment"] == {
        "GEMMINI_MODE": "both",
        "GEMMINI_PERF_CONFIG": "e4m3_vpu",
        "GEMMINI_PERF_SET": "mem.host_tracking=0",
    }
    recovery = json.loads(files["checks/cached_recovery.json"])
    assert (
        recovery["passed"] is True
        and recovery["original_weight_object_shared_inode_restored"] is True
    )
    for name, cycles in [
        ("cached_restored_state_audit.json", 20262623973),
        ("smol_state_audit.json", 73577503143),
        ("smol_control_state_audit.json", 87964575985),
    ]:
        check = json.loads(files["checks/" + name])
        assert (
            check["state_checks_passed"] is True
            and check["actual_final_elf_reopened"] is True
        )
        assert (
            check["measured_cycles"] == cycles
            and check["independent_radiance_baseline"] is False
        )
    payload = json.loads(files["checks/forward_actual_payload.json"])["audits"][0]
    assert (
        payload["status"] == "linked_payloads_match"
        and payload["packed_matrices_checked"] == 155
    )
    assert json.loads(files["checks/forward_recomputed.json"])["passed"] is True
    gqa_port = json.loads(files["gqa8_port/receipt.json"])
    gqa_replay = json.loads(files["gqa8_replay/receipt.json"])
    assert gqa_port["defines"]["LL_HQ"] // gqa_port["defines"]["LL_HKV"] == 8
    assert gqa_port["session_steps"] == gqa_replay["session_steps"] == 34
    assert gqa_replay["source_receipt_sha256"] == sha(files["gqa8_port/receipt.json"])
    assert (
        gqa_replay["passed"] is True and gqa_port["accelerator_work"]["passed"] is True
    )
    assert sha(files["gqa8_port/session.elf"]) == gqa_port["elf_sha256"]
    for name, expected in gqa_port["files_sha256"].items():
        assert sha(files["gqa8_port/" + name]) == expected
    for i, size in enumerate([34 * 512 * 4, 2 * 2 * 2 * 97 * 64 * 4]):
        actual = files[f"gqa8_port/observed{i}.bin"]
        assert actual == files[f"gqa8_replay/expected{i}.bin"] and len(actual) == size
        assert np.isfinite(np.frombuffer(actual, "<f4")).all()
        assert gqa_replay["results"][i]["word_mismatches"] == 0
    preflight = json.loads(files["preflight/cached_port_receipt.json"])
    graph = json.loads(files["logical_graph/receipt.json"])
    assert graph["passed"] is True and graph["decoder_descriptors_interpreted"] is False
    assert graph["session_steps"] == 34 and graph["matrix_reference_calls"] == 578
    assert graph["source_receipt_sha256"] == sha(files["gqa8_port/receipt.json"])
    assert graph["producer_sha256"] == sha(files["logical_graph/producer.py"])
    for i in range(2):
        expected = files[f"logical_graph/expected{i}.bin"]
        assert expected == files[f"gqa8_port/observed{i}.bin"]
        assert graph["results"][i]["word_mismatches"] == 0
        assert graph["results"][i]["expected_sha256"] == sha(expected)
    for stage in ("prefill", "decode"):
        check = json.loads(files[f"checks/cached_{stage}_independent_weights.json"])
        assert (
            check["status"] == "all_encoded_checkpoint_matrices_equal"
            and check["matrices_checked"] == 155
        )
        assert check["source_receipt_sha256"] == sha(
            files["preflight/cached_port_receipt.json"]
        )
        assert check["source_elf_sha256"] == preflight["elf_sha256"]
        assert (
            check["compiler_elf_sha256"]
            == json.loads(files["checks/cached_restored_state_audit.json"])[
                "elf_sha256"
            ]
        )
    for name in (
        "failed_catalog_delta",
        "link_object_storage/rhs_reuse_shared",
        "link_object_storage/compact_rhs_guarded",
    ):
        stored = json.loads(files[name + "/reconstruction.json"])
        assert stored["reconstructed_full_hash_checked"] is True
        for part in ("prefix", "suffix"):
            raw = lzma.decompress(files[f"{name}/{part}.xz"])
            assert (
                len(raw) == stored[part + "_bytes"]
                and sha(raw) == stored[part + "_sha256"]
            )
    print(
        "PASS: both 2:1 and 8:1 GQA recurrent cases; each checks 34 steps, 17,408 logit words and 49,664 K/V words against the independent mesh/VPU replay."
    )
    print(
        "Fresh full-run artifact and state checks retained. This archive adds no full-session speedup claim."
    )
    print(
        "Logical Llama graph also matches exactly without interpreting decoder descriptors."
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
