kernel mx_issue_encoded_schedule(activation: i8[2048] read, activation_scales: i8[128] read, weight: i8[22528] read, weight_scales: i8[1408] read, output_bf16: i16[11264] write, scratch_output_scales: i8[2048] read)
  attrs {precision=FP4, nc=32, kt=128}
  configure()
  for chunk in 0..11:
    for wave in 0..1:
      native_loop(a=&activation[(wave * 128)], b=&weight[((wave * 22528) + (chunk * 16))], c=&output_bf16[(chunk * 64)], asc=&activation_scales[(wave * 128)], bsc=&weight_scales[((wave * 1408) + (chunk * 32))], write=select(wave == 0, 1, 0), accumulate=select(wave == 0, 0, 1), slot=(mod((chunk + wave), 2) + 1))
