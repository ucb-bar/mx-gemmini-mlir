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


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


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
    print('SmolVLA attention capture and late source replays verified')


if __name__ == '__main__':
    main()
