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
