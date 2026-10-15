"""Capture every attention call across a SmolVLA flow session."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model2mlir-root", required=True, type=Path)
parser.add_argument("--capture-root", required=True, type=Path)
parser.add_argument("--out-dir", required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.model2mlir_root.resolve() / "workloads/smolvla"))
from loader import (_MODEL_ID, _MODEL_REVISION, _load_pretrained_config,
                    SmolVLAFlowSession)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = args.capture_root.resolve()
OUT = args.out_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


class Tap(TorchDispatchMode):
    def __init__(self):
        self.rows = []
        self.masks = []
        self.events = []
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        pair = None
        if func == torch.ops.aten.bmm.default and len(args) >= 2:
            left, right = args[:2]
            shape = (tuple(left.shape), tuple(right.shape))
            if shape[0] == (15, 50, 64) and shape[1] in (
                    (15, 64, 113), (15, 64, 163)):
                pair = ('qk', left, right, shape[1][-1])
            elif shape[0] in ((15, 50, 113), (15, 50, 163)) and shape[1] == (
                    15, shape[0][-1], 64):
                pair = ('pv', left, right, shape[0][-1])
        mask = None
        if func == torch.ops.aten.where.self and len(args) >= 3:
            if tuple(args[0].shape) in ((1, 1, 50, 113), (1, 1, 50, 163)):
                mask = args[0]
        result = func(*args, **kwargs)
        if pair or mask is not None:
            with torch._C._DisableTorchDispatch():
                if pair:
                    kind, left, right, keys = pair
                    index = len(self.rows)
                    self.rows.append((kind, keys, left.detach().cpu().numpy().copy(),
                                      right.detach().cpu().numpy().copy(),
                                      result.detach().cpu().numpy().copy()))
                    self.events.append((kind, keys, index))
                if mask is not None:
                    index = len(self.masks)
                    self.masks.append(mask.detach().cpu().numpy().copy())
                    self.events.append(('mask', int(mask.shape[-1]), index))
        return result


cfg = _load_pretrained_config()
cfg.device = 'cpu'
policy = SmolVLAPolicy.from_pretrained(
    _MODEL_ID, config=cfg, revision=_MODEL_REVISION,
    local_files_only=True, strict=True)
model = policy.model.cpu().eval()
flow = SmolVLAFlowSession(model).eval()
with np.load(ROOT / 'inputs.npz', allow_pickle=False) as arrays:
    inputs = tuple(torch.from_numpy(arrays[f'in{i}'].copy()) for i in range(4))
with np.load(ROOT / 'session_goldens.npz', allow_pickle=False) as arrays:
    reference = arrays['actions'].copy()
assert reference.shape[0] == 10
state = inputs
receipts = []
for step in range(reference.shape[0]):
    tap = Tap()
    with torch.no_grad(), tap:
        outputs = flow(*state)
        observed = outputs[0].detach().cpu().numpy().copy()
    assert observed.shape == reference[step].shape
    rmse = float(np.sqrt(np.mean((observed.astype(np.float64) -
                                  reference[step].astype(np.float64)) ** 2)))
    qk = [row for row in tap.rows if row[0] == 'qk']
    pv = [row for row in tap.rows if row[0] == 'pv']
    assert len(qk) == len(pv) == len(tap.masks) == 16
    assert [row[1] for row in qk] == [163, 113] * 8
    assert [row[1] for row in pv] == [163, 113] * 8
    assert [int(mask.shape[-1]) for mask in tap.masks] == [163, 113] * 8
    arrays = {}
    for index, (score, value, mask) in enumerate(zip(qk, pv, tap.masks, strict=True)):
        assert score[1] == value[1] == mask.shape[-1]
        arrays[f'call{index:02d}_q'] = score[2]
        arrays[f'call{index:02d}_kt'] = score[3]
        arrays[f'call{index:02d}_scores'] = score[4]
        arrays[f'call{index:02d}_p'] = value[2]
        arrays[f'call{index:02d}_v'] = value[3]
        arrays[f'call{index:02d}_out'] = value[4]
        arrays[f'call{index:02d}_mask'] = mask
    path = OUT / f'step{step:02d}.npz'
    np.savez_compressed(path, **arrays)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {'step': step, 'calls': 16, 'heads_per_call': 15,
           'key_counts': [item[1] for item in qk],
           'rmse_vs_saved_gold': rmse, 'npz_sha256': sha,
           'npz_bytes': path.stat().st_size, 'events': tap.events}
    receipts.append(row)
    print(json.dumps({key: row[key] for key in ('step', 'calls',
               'rmse_vs_saved_gold', 'npz_bytes', 'npz_sha256')}), flush=True)
    state = (state[0], outputs[1].detach(), outputs[0].detach(),
             outputs[2].detach())
(OUT / 'session_receipt.json').write_text(json.dumps({
    'schema': 'mx_gemmini.smolvla_flow_attention_session_capture.v1',
    'model': f'{_MODEL_ID}@{_MODEL_REVISION}',
    'initial_inputs_sha256': hashlib.sha256((ROOT / 'inputs.npz').read_bytes()).hexdigest(),
    'reference_actions_sha256': hashlib.sha256((ROOT / 'session_goldens.npz').read_bytes()).hexdigest(),
    'steps': receipts,
}, indent=2, sort_keys=True) + '\n')
