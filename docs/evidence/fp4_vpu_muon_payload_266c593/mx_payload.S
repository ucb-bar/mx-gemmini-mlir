.section .rodata
.balign 64
.globl activation
.type activation, @object
activation:
.incbin "activation.bin"
.size activation, .-activation
.balign 64
.globl activation_scales
.type activation_scales, @object
activation_scales:
.incbin "activation_scales.bin"
.size activation_scales, .-activation_scales
.balign 64
.globl mx_expected_bf16
.type mx_expected_bf16, @object
mx_expected_bf16:
.incbin "mx_expected_bf16.bin"
.size mx_expected_bf16, .-mx_expected_bf16
.balign 64
.globl weight
.type weight, @object
weight:
.incbin "weight.bin"
.size weight, .-weight
.balign 64
.globl weight_scales
.type weight_scales, @object
weight_scales:
.incbin "weight_scales.bin"
.size weight_scales, .-weight_scales
.balign 64
.section .note.GNU-stack,"",@progbits
