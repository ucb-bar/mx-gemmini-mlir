.section .rodata
.balign 64
.globl activation
activation:
.incbin "activation.bin"
.balign 64
.globl activation_scales
activation_scales:
.incbin "activation_scales.bin"
.balign 64
.globl golden_bf16
golden_bf16:
.incbin "golden_bf16.bin"
.balign 64
.globl golden_fp8
golden_fp8:
.incbin "golden_fp8.bin"
.balign 64
.globl golden_output_scales
golden_output_scales:
.incbin "golden_output_scales.bin"
.balign 64
.globl nicolas_fp4
nicolas_fp4:
.incbin "nicolas_fp4.bin"
.balign 64
.globl nicolas_output_scales
nicolas_output_scales:
.incbin "nicolas_output_scales.bin"
.balign 64
.globl output_scales
output_scales:
.incbin "output_scales.bin"
.balign 64
.globl weight
weight:
.incbin "weight.bin"
.balign 64
.globl weight_scales
weight_scales:
.incbin "weight_scales.bin"
.balign 64
.section .note.GNU-stack,"",@progbits
