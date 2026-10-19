from pathlib import Path
import json
import numpy as np
import torch
from m2m.capture.bundle import write_bundle
from mx_gemmini_support.execution_policy import make_policy
from mx_gemmini_support.target_profile import load_profile
class Mixed(torch.nn.Module):
 def forward(self,frequency,position,weight):
  angle=frequency @ position
  return angle, angle @ weight
out=Path('../.tmp/mx_model_goal_20261018/rotary_probe')
with np.load(out.parent/'onepass160/capture/extra.npz') as d:
 freq=torch.from_numpy(d['buf::lm.model.rotary_emb.inv_freq'].copy()).reshape(1,32,1)
inputs=(freq,torch.arange(160,dtype=torch.float32).reshape(1,1,160),torch.linspace(-0.25,0.25,160*16).reshape(1,160,16))
write_bundle(Mixed(),inputs,out/'capture_mixed',capture_regions=False)
profile=load_profile(Path('profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json'))
policy=make_policy((out/'capture_mixed/model.mlir').read_text(),profile,preserve_singleton_reductions=True)
(out/'mixed_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
print(policy['matrix_formats'])
