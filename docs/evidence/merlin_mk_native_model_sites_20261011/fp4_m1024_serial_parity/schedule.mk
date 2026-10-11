kernel mx_native_schedule(activation: i8[32768] read, activation_scales: i8[2048] read, weight: i8[1024] read, weight_scales: i8[64] read, output_bf16: i16[32768] write, scratch_output_scales: i8[2048] read)
  attrs {precision=FP4, tm=128, nc=32, kt=64}
  configure()
  for row in 0..8:
    for chunk in 0..1:
      for wave in 0..1:
        native_loop(a=&activation[((row * 4096) + (wave * 64))], b=&weight[((wave * 1024) + (chunk * 16))], c=&output_bf16[((row * 8192) + (chunk * 64))], asc=&activation_scales[((wave * 2048) + (row * 128))], bsc=&weight_scales[((wave * 64) + (chunk * 32))], write=select(wave == 0, 1, 0), accumulate=select(wave == 0, 0, 1), slot=(mod((row + chunk + wave), 2) + 1))
