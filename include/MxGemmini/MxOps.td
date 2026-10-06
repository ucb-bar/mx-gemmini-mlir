include "mlir/IR/OpBase.td"

def MxGemmini_Dialect : Dialect {
  let name = "mx_gemmini";
  let cppNamespace = "::mlir::mx_gemmini";
  let summary = "Source-scoped MX Gemmini contract handoff";
}

class Mx_Op<string mnemonic> : Op<MxGemmini_Dialect, mnemonic> {
  let hasVerifier = 1;
}

def Mx_EncodeOp : Mx_Op<"encode"> {
  let summary = "Prepare packed operand codes and E8M0 scales";
  let arguments = (ins AnyTensor:$source, StrAttr:$site_id, StrAttr:$format,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_ContractOp : Mx_Op<"contract"> {
  let summary = "Accumulate an MX contraction in the selected mesh";
  let arguments = (ins AnyTensor:$lhs_codes, AnyTensor:$lhs_scales,
                   AnyTensor:$rhs_codes, AnyTensor:$rhs_scales,
                   StrAttr:$site_id, StrAttr:$format,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$accumulator);
}

def Mx_ReadoutBF16Op : Mx_Op<"readout_bf16"> {
  let summary = "Read BF16 from the accumulator";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$value);
}

def Mx_ReadoutToSmemOp : Mx_Op<"readout_to_smem"> {
  let summary = "Write BF16 accumulator values to a shared-memory view";
  let arguments = (ins AnyTensor:$accumulator, AnyMemRef:$destination,
                   StrAttr:$site_id, StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_WaitOp : Mx_Op<"wait"> {
  let summary = "Wait for previously issued MX commands before a Muon consumer";
  let arguments = (ins StrAttr:$site_id, StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_RequantizeOp : Mx_Op<"requantize"> {
  let summary = "Requantize accumulator output for an on-chip consumer";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id, StrAttr:$format,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}
