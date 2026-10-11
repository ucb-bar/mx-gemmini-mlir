kernel mx_issue_encoded_schedule(activation: i8[1024] read, activation_scales: i8[32] read, weight: i8[2048] read, weight_scales: i8[64] read, output_bf16: i16[512] write, scratch_output_scales: i8[2048] read)
  attrs {precision=FP8, tm=16, nc=32, kt=64}
  configure()
  for row in 0..1:
    for chunk in 0..1:
      for wave in 0..1:
        native_loop(a=&activation[((row * 1024) + (wave * 64))], b=&weight[((wave * 2048) + (chunk * 32))], c=&output_bf16[((row * 1024) + (chunk * 64))], asc=&activation_scales[((wave * 32) + (row * 16))], bsc=&weight_scales[((wave * 64) + (chunk * 32))], write=select(wave == 0, 1, 0), accumulate=select(wave == 0, 0, 1), slot=(mod((row + chunk + wave), 2) + 1))
