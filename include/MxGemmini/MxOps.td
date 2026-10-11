include "mlir/IR/OpBase.td"

def MxGemmini_Dialect : Dialect {
  let name = "mx_gemmini";
  let cppNamespace = "::mlir::mx_gemmini";
  let summary = "Source-scoped MX Gemmini contract handoff";
}

class Mx_Op<string mnemonic> : Op<MxGemmini_Dialect, mnemonic> {
  let hasVerifier = 1;
}

def Mx_ResourceOp : Mx_Op<"resource"> {
  let summary = "A checked external MX operand, scale, or LUT resource";
  let arguments = (ins StrAttr:$site_id, StrAttr:$resource_name,
                   StrAttr:$resource_sha256, StrAttr:$resource_layout,
                   StrAttr:$payload_manifest_sha256,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$data);
}

def Mx_UploadLutOp : Mx_Op<"upload_lut"> {
  let summary = "Bind an indexed MX LUT bank before contraction";
  let arguments = (ins AnyTensor:$data, StrAttr:$site_id,
                   StrAttr:$lut_target, StrAttr:$payload_manifest_sha256,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}

def Mx_RuntimeLutOp : Mx_Op<"runtime_lut"> {
  let summary = "Bind a runtime LUT bank to one connected MX contraction site";
  let arguments = (ins AnyTensor:$data, StrAttr:$site_id,
                   StrAttr:$lut_target, StrAttr:$runtime_buffer,
                   I32Attr:$groups, I32Attr:$entry_bits,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}

def Mx_EncodeOp : Mx_Op<"encode"> {
  let summary = "Prepare packed operand codes and E8M0 scales";
  let arguments = (ins AnyTensor:$source, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$element_format,
                   OptionalAttr<StrAttr>:$projection,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_ContractOp : Mx_Op<"contract"> {
  let summary = "Accumulate an MX contraction in the selected mesh";
  let arguments = (ins AnyTensor:$lhs_codes, AnyTensor:$lhs_scales,
                   AnyTensor:$rhs_codes, AnyTensor:$rhs_scales,
                   StrAttr:$site_id, OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$activation_format,
                   OptionalAttr<StrAttr>:$weight_format,
                   OptionalAttr<StrAttr>:$activation_projection,
                   OptionalAttr<StrAttr>:$weight_projection,
                   OptionalAttr<I32Attr>:$pe_mode,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   OptionalAttr<StrAttr>:$payload_manifest_sha256,
                   OptionalAttr<StrAttr>:$payload_origin,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$accumulator);
}

def Mx_ReadoutBF16Op : Mx_Op<"readout_bf16"> {
  let summary = "Read BF16 through a selected MX output memory";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   OptionalAttr<StrAttr>:$memory_layout,
                   OptionalAttr<StrAttr>:$source_memory,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$value);
}

def Mx_ReadoutQuantizedOp : Mx_Op<"readout_quantized"> {
  let summary = "Read MX output codes, including packed LUT indices, and E8M0 scales";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   StrAttr:$output_format,
                   OptionalAttr<StrAttr>:$output_projection,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_HostRequantizeOp : Mx_Op<"host_requantize"> {
  let summary = "Convert BF16 MX output using an explicit source compatibility policy";
  let arguments = (ins AnyTensor:$value, Optional<AnyTensor>:$output_lut,
                   StrAttr:$site_id,
                   StrAttr:$output_format, StrAttr:$quant_policy,
                   StrAttr:$profile_sha256, StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_ReadoutToSmemOp : Mx_Op<"readout_to_smem"> {
  let summary = "Write BF16 accumulator values to a shared-memory view";
  let arguments = (ins AnyTensor:$accumulator, AnyMemRef:$destination,
                   StrAttr:$site_id, OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_WaitOp : Mx_Op<"wait"> {
  let summary = "Wait for previously issued MX commands before a Muon consumer";
  let arguments = (ins StrAttr:$site_id, OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_RequantizeOp : Mx_Op<"requantize"> {
  let summary = "Requantize accumulator output for an on-chip consumer";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$output_format,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_VpuExecuteOp : Mx_Op<"vpu_execute"> {
  let summary = "Issue a source-bound BF16 scratchpad VPU command";
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                   StrAttr:$site_id, StrAttr:$kind,
                   I32Attr:$src1_row, I32Attr:$src2_row, I32Attr:$dst_row,
                   I32Attr:$rows, I32Attr:$reduction_length,
                   BoolAttr:$broadcast, I32Attr:$immediate_bf16,
                   OptionalAttr<I32Attr>:$second_dst_row,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs Variadic<AnyTensor>:$outputs);
}

def Mx_SpadRequantOp : Mx_Op<"spad_requant"> {
  let summary = "Requantize BF16 scratchpad rows into resident MX operand A";
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                   StrAttr:$site_id,
                   I32Attr:$source_row, I32Attr:$destination_row,
                   I32Attr:$m, I32Attr:$n, StrAttr:$output_format,
                   BoolAttr:$tiled, BoolAttr:$resident,
                   I64Attr:$scale_dram_address,
                   OptionalAttr<StrAttr>:$scale_buffer,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs Variadic<AnyTensor>:$outputs);
}

def Mx_ResidentContractOp : Mx_Op<"resident_contract"> {
  let summary = "Consume a resident MX activation tile and scales in a second contraction";
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                   StrAttr:$site_id,
                   I32Attr:$activation_row, I32Attr:$weight_row,
                   I32Attr:$output_row, I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   StrAttr:$activation_format, StrAttr:$weight_format,
                   StrAttr:$output_format,
                   StrAttr:$weight_buffer, StrAttr:$weight_scales_buffer,
                   StrAttr:$output_scales_buffer,
                   OptionalAttr<StrAttr>:$weight_lut_buffer,
                   OptionalAttr<StrAttr>:$activation_lut_buffer,
                   OptionalAttr<StrAttr>:$output_lut_buffer,
                   OptionalAttr<I32Attr>:$lut_groups,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs Variadic<AnyTensor>:$outputs);
}

def Mx_DmaMatrixOp : Mx_Op<"dma_matrix"> {
  let summary = "Load a row-major byte matrix into MX scratchpad tiles";
  let arguments = (ins AnyMemRef:$buffer, StrAttr:$site_id,
                   I32Attr:$matrix_rows, I32Attr:$matrix_cols,
                   I32Attr:$burst_cols, I32Attr:$spad_row,
                   StrAttr:$profile_sha256, StrAttr:$source_sha256);
}

def Mx_MemorySetupOp : Mx_Op<"memory_setup"> {
  let summary = "Initialize a source-bound MX-only memory program";
  let arguments = (ins StrAttr:$site_id,
                   StrAttr:$profile_sha256, StrAttr:$source_sha256);
}

def Mx_SpadMvoutLinearOp : Mx_Op<"spad_mvout_linear"> {
  let summary = "Store consecutive MX scratchpad byte tiles to a flat buffer";
  let arguments = (ins AnyMemRef:$buffer, StrAttr:$site_id,
                   I32Attr:$total_bytes, I32Attr:$tile_cols,
                   I32Attr:$spad_row,
                   StrAttr:$profile_sha256, StrAttr:$source_sha256);
}

def Mx_LoadScalesOp : Mx_Op<"load_scales"> {
  let summary = "Load E8M0 scales into the selected MX scale memory";
  let arguments = (ins AnyMemRef:$buffer, StrAttr:$site_id,
                   I32Attr:$payload_bytes, StrAttr:$scale_target,
                   StrAttr:$profile_sha256, StrAttr:$source_sha256);
}
