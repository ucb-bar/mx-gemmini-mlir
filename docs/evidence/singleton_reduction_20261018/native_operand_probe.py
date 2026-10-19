import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from mx_gemmini_support.source_attention_qk import decode_e4m3

root = Path.cwd()
out = root.parent / '.tmp/mx_model_goal_20261018/rotary_probe'
stage = out.parent / 'onepass160/capture'
source = root / 'mx_gemmini_support/runtime_quant.c'
library = out / 'quant.so'
subprocess.run(['cc','-O2','-fno-fast-math','-fPIC','-shared',str(source),'-o',str(library)],check=True)
lib = ctypes.CDLL(str(library))
with np.load(stage/'extra.npz') as data:
    freq = data['buf::lm.model.rotary_emb.inv_freq'].copy()
positions = np.arange(160,dtype=np.float32)
a = np.zeros((32,32),dtype=np.float32)
b = np.zeros((32,160),dtype=np.float32)
a[:,0] = freq
b[0,:] = positions

def pack(name,x,axis):
    codes=np.zeros_like(x,dtype=np.uint8)
    scales=np.zeros((x.shape[1]//32,x.shape[0]) if axis==1 else (x.shape[0]//32,x.shape[1]),dtype=np.uint8)
    fn=getattr(lib,name)
    fn.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p]
    fn.restype=ctypes.c_int
    assert fn(x.ctypes.data,*x.shape,8,codes.ctypes.data,scales.ctypes.data)==0
    decoded=np.array([decode_e4m3(int(c)) for c in codes.flat],dtype=np.float32).reshape(x.shape)
    s=np.exp2(scales.astype(np.int32)-127).astype(np.float32)
    if axis==1: s=np.repeat(s.T,32,axis=1)
    else: s=np.repeat(s,32,axis=0)
    return codes, scales, decoded*s
ac,asc,aq=pack('mx_pack_activation_f32',a,1)
bc,bsc,bq=pack('mx_pack_weight_f32',b,0)
exact=freq[:,None]*positions[None,:]
proxy=aq@bq
# Operand-isolation probe only: hardware product and accumulation are not modeled here.
np.savez(out/'operands.npz',a=a,b=b,activation_codes=ac,activation_scales=asc,weight_codes=bc,weight_scales=bsc,exact=exact,operand_proxy=proxy)
report={'scope':'native_operand_quantization_diagnostic_not_mesh_oracle','shape_mnk':[32,160,1],
 'inv_freq_sha256':hashlib.sha256(freq.tobytes()).hexdigest(),
 'maximum_frequency_error':float(np.max(np.abs(aq[:,0]-freq))),
 'maximum_position_error':float(np.max(np.abs(bq[0]-positions))),
 'maximum_angle_error_radians':float(np.max(np.abs(proxy-exact))),
 'maximum_cosine_error':float(np.max(np.abs(np.cos(proxy)-np.cos(exact)))),
 'maximum_sine_error':float(np.max(np.abs(np.sin(proxy)-np.sin(exact)))),
 'changed_positions':int(np.count_nonzero(bq[0]!=positions)),
 'position_examples':[{'position':int(p),'quantized':float(bq[0,p])} for p in [15,31,63,127,128,129,132,159]],
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
