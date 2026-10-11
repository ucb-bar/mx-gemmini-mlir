.section .rodata
.balign 64
.globl a1_activation
a1_activation:
.incbin "a1_activation.bin"
.balign 64
.globl a1_scales
a1_scales:
.incbin "a1_scales.bin"
.balign 64
.globl b1_scales
b1_scales:
.incbin "b1_scales.bin"
.balign 64
.globl b1_weight
b1_weight:
.incbin "b1_weight.bin"
.balign 64
.globl b2_scales
b2_scales:
.incbin "b2_scales.bin"
.balign 64
.globl b2_weight
b2_weight:
.incbin "b2_weight.bin"
.balign 64
.globl c1_codes_ref
c1_codes_ref:
.incbin "c1_codes_ref.bin"
.balign 64
.globl c1_scales_ref
c1_scales_ref:
.incbin "c1_scales_ref.bin"
.balign 64
.globl c2_codes_ref
c2_codes_ref:
.incbin "c2_codes_ref.bin"
.balign 64
.globl c2_scales_ref
c2_scales_ref:
.incbin "c2_scales_ref.bin"
.balign 64
.section .note.GNU-stack,"",@progbits
