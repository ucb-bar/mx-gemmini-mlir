.section .rodata

.p2align 6
.globl first_activation
first_activation:
.incbin "data/first_activation.bin"

.p2align 6
.globl first_activation_lut
first_activation_lut:
.incbin "data/first_activation_lut.bin"

.p2align 6
.globl first_activation_scales
first_activation_scales:
.incbin "data/first_activation_scales.bin"

.p2align 6
.globl first_output_lut
first_output_lut:
.incbin "data/first_output_lut.bin"

.p2align 6
.globl first_weight
first_weight:
.incbin "data/first_weight.bin"

.p2align 6
.globl first_weight_lut
first_weight_lut:
.incbin "data/first_weight_lut.bin"

.p2align 6
.globl first_weight_scales
first_weight_scales:
.incbin "data/first_weight_scales.bin"

.p2align 6
.globl first_golden_bf16
first_golden_bf16:
.incbin "data/first_golden_bf16.bin"

.p2align 6
.globl second_activation
second_activation:
.incbin "data/second_activation.bin"

.p2align 6
.globl second_activation_lut
second_activation_lut:
.incbin "data/second_activation_lut.bin"

.p2align 6
.globl second_activation_scales
second_activation_scales:
.incbin "data/second_activation_scales.bin"

.p2align 6
.globl second_output_lut
second_output_lut:
.incbin "data/second_output_lut.bin"

.p2align 6
.globl second_weight
second_weight:
.incbin "data/second_weight.bin"

.p2align 6
.globl second_weight_lut
second_weight_lut:
.incbin "data/second_weight_lut.bin"

.p2align 6
.globl second_weight_scales
second_weight_scales:
.incbin "data/second_weight_scales.bin"

.p2align 6
.globl second_golden_bf16
second_golden_bf16:
.incbin "data/second_golden_bf16.bin"
