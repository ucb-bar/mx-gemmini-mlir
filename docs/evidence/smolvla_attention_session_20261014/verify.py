"""Verify the archived SmolVLA attention session and late Spike cutpoints."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / 'smolvla_attention_cutpoint_20261014'


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verify_compiler_pair(late: dict) -> None:
    folder = HERE / 'compiled_pair_pipeline'
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['schema'] == 'mx_gemmini.smolvla_masked_attention_paired_pipelined.v1'
    assert receipt['scope'] == 'final flow step calls 14 and 15; head 0 only'
    assert receipt['is_stitched_stage_or_full_model'] is False
    assert receipt['compiler_schedule'] == 'pipelined'
    assert receipt['spike_mode'] == ('GEMMINI_MODE=both; '
                                     'GEMMINI_PERF_CONFIG=e4m3_vpu; '
                                     'GEMMINI_PERF_SET=mem.host_tracking=0')
    assert receipt['source_revision'] == late['source_revision']
    assert receipt['spike_extension_sha256'] == late['spike_extension_sha256']
    assert receipt['source_patch_sha256'] == sha(FIRST / 'attn_flash_smol.patch')
    assert receipt['schedule_plan_sha256'] == sha(
        FIRST / 'attention_pipeline.schedule.mlir')
    assert receipt['driver_sha256'] == sha(
        HERE.parents[2] / 'tools/fixtures/smol_attention_compiler_first_driver.c')
    for name, digest in receipt['artifact_sha256'].items():
        assert sha(folder / name) == digest, name
    assert [row['logical_keys'] for row in receipt['rows']] == [113, 163]
    for row in receipt['rows']:
        keys = row['logical_keys']
        call = 15 if keys == 113 else 14
        source = next(entry for entry in late['rows'] if entry['call'] == call)
        case = folder / f'keys{keys}'
        assert row['captured_step'] == 9 and row['captured_call'] == call
        assert row['cutpoint_sha256'] == source['cutpoint_sha256']
        assert row['header_sha256'] == source['header_sha256']
        assert row['expected_output_hash'] == source['source_output_hash']
        assert row['model2mlir_frontend_sha256'] == sha(
            FIRST / f'attention{keys}.model2mlir.mlir')
        assert row['typed_mlir_sha256'] == sha(case / 'attention.mx.mlir')
        assert row['issuer_source_sha256'] == sha(case / 'mx_issue.c')
        assert row['issuer_object_sha256'] == sha(case / 'mx_issue.o')
        assert row['physical_shape'] == [64, 128 if keys == 113 else 192, 64]
        assert row['command_count'] == (167 if keys == 113 else 232)
        assert row['fence_count'] == 0 and row['same_output'] is True
        log = gzip.decompress((case / 'spike.log.gz').read_bytes()).decode()
        assert 'attn_flash PASSED' in log
        assert 'gemmini perf [e4m3_vpu]:' in log
        assert 'paired full BF16 output match 4096 values' in log
        assert 'compiler first full BF16 output match 4096 values' in log
        assert f"compiler first fresh guest hash {row['expected_output_hash']}" in log
        assert f"check O hash vs Spike: match ({row['expected_output_hash']})" in log
        matches = re.findall(
            r'^paired (pipeline|compiler) ([0-3]) cycles (\d+) hash ([0-9a-f]{16})$',
            log, re.M)
        assert [(kind, int(index)) for kind, index, _, _ in matches] == [
            ('pipeline', 0), ('compiler', 1),
            ('pipeline', 2), ('compiler', 3)]
        assert all(digest == row['expected_output_hash']
                   for _, _, _, digest in matches)
        reference = [int(matches[index][2]) for index in (0, 2)]
        compiler = [int(matches[index][2]) for index in (1, 3)]
        assert reference == row['source_pipelined_cycles']
        assert compiler == row['compiler_pipelined_cycles']
        assert abs(sum(compiler) / sum(reference) - 1 -
                   row['compiler_over_source_fraction']) < 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step-dir', type=Path,
                        help='Optional regenerated step00.npz through step09.npz')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'session_capture_manifest.json').read_text())
    late = json.loads((HERE / 'late_replay_receipt.json').read_text())
    assert manifest['schema'] == 'mx_gemmini.smolvla_flow_attention_session_capture.v1'
    assert sha(HERE / 'capture_session.py') == manifest['capture_script_sha256']
    assert len(manifest['steps']) == 10
    assert late['schema'] == 'mx_gemmini.smolvla_late_source_attention.v1'
    assert late['perf_config'] == 'e4m3_vpu'
    assert late['perf_set'] == 'mem.host_tracking=0'
    assert late['full_model_cycle_comparison'] is False
    for step, row in enumerate(manifest['steps']):
        assert row['step'] == step and row['calls'] == 16
        assert row['heads_per_call'] == 15
        assert row['key_counts'] == [163, 113] * 8
        assert len(row['events']) == 48
        expected_events = []
        for call in range(16):
            keys = 163 if call % 2 == 0 else 113
            expected_events.extend([['qk', keys, call * 2],
                                    ['mask', keys, call],
                                    ['pv', keys, call * 2 + 1]])
        assert row['events'] == expected_events
        if args.step_dir:
            path = args.step_dir / f'step{step:02d}.npz'
            assert path.stat().st_size == row['npz_bytes']
            assert sha(path) == row['npz_sha256']
    assert [(row['step'], row['call'], row['keys'], row['head'])
            for row in late['rows']] == [(9, 14, 163, 0), (9, 15, 113, 0)]
    for row in late['rows']:
        call = row['call']
        folder = HERE / f'call{call:02d}'
        cutpoint = folder / 'cutpoint.npz'
        assert sha(cutpoint) == row['cutpoint_sha256']
        assert sha(folder / 'header.h') == row['header_sha256']
        archive = folder / 'source_spike.log.gz'
        assert sha(archive) == row['compressed_log_sha256']
        raw = gzip.decompress(archive.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == row['spike_log_sha256']
        text = raw.decode()
        assert 'attn_flash PASSED' in text
        assert 'pipelined  O: 0/4096 differ' in text
        assert 'gemmini perf [e4m3_vpu]:' in text
        equal = re.search(r'check O hash: serial ([0-9a-f]{16}), '
                          r'pipelined ([0-9a-f]{16}) \(equal\)', text)
        assert equal and equal.group(1) == equal.group(2) == row['source_output_hash']
        assert int(re.search(r'PERF pipelined\s+(\d+) cycles', text).group(1)) == row['pipelined_cycles']
        assert int(re.search(r'accuracy O head 0 \(50 rows\): rel_fro (\d+) ppm',
                             text).group(1)) == row['relative_fro_error_ppm']
        with np.load(cutpoint, allow_pickle=False) as data:
            assert data['q'].shape == (15, 50, 64)
            assert data['kt'].shape == (15, 64, row['keys'])
            assert data['p'].shape == (15, 50, row['keys'])
            assert data['v'].shape == (15, row['keys'], 64)
            assert data['mask'].shape == (1, 1, 50, row['keys'])
            if args.step_dir:
                with np.load(args.step_dir / 'step09.npz', allow_pickle=False) as step:
                    for name in ('q', 'kt', 'p', 'v', 'mask'):
                        assert np.array_equal(data[name], step[f'call{call:02d}_{name}'])
    verify_compiler_pair(late)
    print('SmolVLA attention capture, source, and compiler pairs verified')


if __name__ == '__main__':
    main()
