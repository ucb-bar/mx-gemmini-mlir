"""Reject invalid deployment artifacts before creating a workload directory."""

import json
import struct
from pathlib import Path

import pytest

from tools.model_release import registry, sha
from tools.package_firesim_model import package


def executable(path: Path, *, entry=0x80000078, physical=0x80000000):
    data = bytearray(128)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<HHIQQQIHHHHHH', data, 16, 2, 243, 1, entry, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    struct.pack_into('<IIQQQQQQ', data, 64, 1, 5, 0, 0x80000000, physical, 128, 4096, 4096)
    path.write_bytes(data)
    return path


def receipt(path, elf):
    path.write_text(json.dumps({'schema': 'mx_gemmini.linked_merlin_session.v1',
                               'profile_sha256': registry()['profile']['canonical_sha256'],
                               'elf_sha256': sha(elf), 'elf_bytes': elf.stat().st_size, 'mem_bytes': 8192}))
    return path


@pytest.mark.parametrize('case', ['entry', 'nonidentity', 'changed_elf', 'changed_profile'])
def test_invalid_artifact_leaves_no_workload(tmp_path, case):
    elf = executable(tmp_path / 'session.elf', entry=0x90000000 if case == 'entry' else 0x80000078,
                     physical=0x80001000 if case == 'nonidentity' else 0x80000000)
    link = receipt(tmp_path / 'link.json', elf)
    if case == 'changed_elf':
        with elf.open('ab') as stream:
            stream.write(b'changed')
    if case == 'changed_profile':
        data = json.loads(link.read_text())
        data['profile_sha256'] = '0' * 64
        link.write_text(json.dumps(data))
    out = tmp_path / 'package'
    with pytest.raises(ValueError):
        package(elf, link, out, 'mx-model')
    assert not out.exists()


def test_workload_preserves_binary_and_reports_memory(tmp_path):
    elf = executable(tmp_path / 'session.elf')
    link = receipt(tmp_path / 'link.json', elf)
    out = tmp_path / 'package'
    result = package(elf, link, out, 'mx-model')
    assert sha(out / 'workloads/mx-model/session.elf') == sha(elf)
    assert result['recommended_dram_bytes'] == 8192
    assert result['status'] == 'packaged_not_fpga_executed'
    config = json.loads((out / 'workloads/mx-model.json').read_text())
    assert config['common_rootfs'] is None
    assert config['common_simulation_outputs'] == ['uartlog']


def test_completed_output_refuses_deadlocked_timing(tmp_path, monkeypatch):
    from tools import check_reproduced_model
    from tools.model_release import ROOT

    item = registry()['models']['tinyllama-forward']
    record = item['stages']['forward']
    probe = {'capture_model_sha256': record['capture_model_sha256'],
             'checkpoint_weights_sha256': record['weights_sha256'],
             'execution_policy': json.loads((ROOT / record['policy']['path']).read_text()),
             'spike': {'exit_code': 0}}
    (tmp_path / 'mx_stage_probe.json').write_text(json.dumps(probe))
    (tmp_path / 'spike.log').write_text('gemmini perf: MODEL DEADLOCK\nDONE\n')
    monkeypatch.setattr(check_reproduced_model, 'evidence', lambda name: (item, {}))
    out = tmp_path / 'result.json'
    with pytest.raises(ValueError, match='deadlock'):
        check_reproduced_model.check('tinyllama-forward', out, stage_build=tmp_path)
    assert not out.exists()
