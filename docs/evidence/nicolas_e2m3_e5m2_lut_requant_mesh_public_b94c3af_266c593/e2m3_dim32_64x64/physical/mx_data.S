.section .rodata
.balign 64
.globl activation
activation:
.incbin "activation.bin"
.balign 64
.globl activation_lut
activation_lut:
.incbin "activation_lut.bin"
.balign 64
.globl activation_scales
activation_scales:
.incbin "activation_scales.bin"
.balign 64
.globl golden_bf16
golden_bf16:
.incbin "golden_bf16.bin"
.balign 64
.globl golden_lut_indices
golden_lut_indices:
.incbin "golden_lut_indices.bin"
.balign 64
.globl golden_output_scales
golden_output_scales:
.incbin "golden_output_scales.bin"
.balign 64
.globl output_lut
output_lut:
.incbin "output_lut.bin"
.balign 64
.globl weight
weight:
.incbin "weight.bin"
.balign 64
.globl weight_lut
weight_lut:
.incbin "weight_lut.bin"
.balign 64
.globl weight_scales
weight_scales:
.incbin "weight_scales.bin"
.balign 64
.section .note.GNU-stack,"",@progbits
