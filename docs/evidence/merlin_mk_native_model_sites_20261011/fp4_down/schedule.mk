kernel mx_issue_encoded_schedule(activation: i8[5632] read, activation_scales: i8[352] read, weight: i8[22528] read, weight_scales: i8[1408] read, output_bf16: i16[4096] write, scratch_output_scales: i8[2048] read)
  attrs {precision=FP4, nc=128, kt=32}
  configure()
  for chunk in 0..1:
    for wave in 0..11:
      native_loop(a=&activation[(wave * 32)], b=&weight[((wave * 2048) + (chunk * 64))], c=&output_bf16[(chunk * 256)], asc=&activation_scales[(wave * 32)], bsc=&weight_scales[((wave * 128) + (chunk * 128))], write=select(wave == 10, 1, 0), accumulate=select(wave == 0, 0, 1), slot=(mod(((chunk * 11) + wave), 2) + 1))
