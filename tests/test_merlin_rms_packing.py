"""RMS packing may remove only a complete, private, admitted call sequence."""

from mx_gemmini_support.merlin_rms_packing import rewrite_rms_packing


def fixture(escape=False, barrier=False, divisor=32):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    x, row, inv, scalar, bf16, packed = (
        "tensor<1x16x32xf32>",
        "tensor<1x16xf32>",
        "tensor<1x16x1xf32>",
        "tensor<f32>",
        "tensor<1x16x32xbf16>",
        "tensor<528xi8>",
    )
    signatures = {
        "square": ([x], x),
        "sum": ([x], row),
        "divide": ([row, scalar], row),
        "eps": ([inv, scalar], inv),
        "rsqrt": ([inv], inv),
        "multiply": ([x, inv, "tensor<32xf32>"], bf16),
        "pack": (["tensor<16x32xbf16>"], packed),
    }
    declarations = []
    for name, (inputs, out) in signatures.items():
        args = [
            f'%a{i}: {typ} {{bufferization.access = "read"}}'
            for i, typ in enumerate(inputs)
        ]
        args += [f'%out: {out} {{bufferization.access = "write"}}']
        declarations.append(f"func.func private @{name}({', '.join(args)}) -> {out}")
    source = f"""builtin.module {{
    {" ".join(declarations)}
    func.func private @unknown()
    func.func @forward(%x: {x}, %gamma: tensor<32xf32>) -> ({packed}{", " + x if escape else ""}) {{
      %a = tensor.empty() : {x}
      %s = func.call @square(%x, %a) : ({x}, {x}) -> {x}
      %b = tensor.empty() : {row}
      %r = func.call @sum(%s, %b) : ({x}, {row}) -> {row}
      %d = arith.constant {divisor}.0 : f32
      %dd = tensor.splat %d : {scalar}
      %c = tensor.empty() : {row}
      %mean = func.call @divide(%r, %dd, %c) : ({row}, {scalar}, {row}) -> {row}
      %view = tensor.expand_shape %mean [[0], [1, 2]] output_shape [1, 16, 1] : {row} into {inv}
      %e = arith.constant 1.000000e-05 : f32
      %ee = tensor.splat %e : {scalar}
      %dest = tensor.empty() : {inv}
      %added = func.call @eps(%view, %ee, %dest) : ({inv}, {scalar}, {inv}) -> {inv}
      %dst = tensor.empty() : {inv}
      %i = func.call @rsqrt(%added, %dst) : ({inv}, {inv}) -> {inv}
      {"func.call @unknown() : () -> ()" if barrier else ""}
      %dst2 = tensor.empty() : {bf16}
      %n = func.call @multiply(%x, %i, %gamma, %dst2) : ({x}, {inv}, tensor<32xf32>, {bf16}) -> {bf16}
      %flat = tensor.collapse_shape %n [[0, 1], [2]] : {bf16} into tensor<16x32xbf16>
      %dst3 = tensor.empty() : {packed}
      %p = func.call @pack(%flat, %dst3) : (tensor<16x32xbf16>, {packed}) -> {packed}
      func.return %p{", %s" if escape else ""} : {packed}{", " + x if escape else ""}
    }}
    }}"""
    catalog = {
        "rms_packing": [
            {
                "symbol": "fused",
                "shape": [1, 16, 32],
                "packed_bytes": 528,
                "producer_symbols": ["multiply"],
                "prepare_symbols": ["pack"],
            }
        ],
        "vpu_unary": {
            "kernels": [
                {"symbol": "square", "operation": "square", "shape": [1, 16, 32]},
                {"symbol": "rsqrt", "operation": "rsqrt", "shape": [1, 16, 1]},
            ]
        },
        "vpu_reduce": {
            "kernels": [
                {
                    "symbol": "sum",
                    "operation": "rsum",
                    "shape": [1, 16, 32],
                    "output_shape": [1, 16],
                }
            ]
        },
        "vpu_elementwise": {
            "kernels": [
                {
                    "symbol": "divide",
                    "operation": "div",
                    "shape": [1, 16],
                    "rhs_shape": [],
                },
                {
                    "symbol": "eps",
                    "operation": "add",
                    "shape": [1, 16, 1],
                    "rhs_shape": [],
                },
            ]
        },
    }
    return parse_mlir_text(source), catalog


def calls(module):
    return [
        op.callee.root_reference.data for op in module.walk() if op.name == "func.call"
    ]


def test_closed_rms_sequence_becomes_one_packed_call_without_mutating_input():
    module, catalog = fixture()
    result, records = rewrite_rms_packing(module, catalog)
    assert calls(result) == ["fused"]
    assert len(calls(module)) == 7
    assert len(records) == 1 and len(records[0]["removed_calls"]) == 7
    assert records[0]["shape"] == [1, 16, 32]


def test_live_intermediate_escape_keeps_original_sequence():
    module, catalog = fixture(escape=True)
    result, records = rewrite_rms_packing(module, catalog)
    assert not records and calls(result) == calls(module)


def test_unknown_effects_or_changed_divisor_keep_original_sequence():
    for options in ({"barrier": True}, {"divisor": 64}):
        module, catalog = fixture(**options)
        result, records = rewrite_rms_packing(module, catalog)
        assert not records and calls(result) == calls(module)
