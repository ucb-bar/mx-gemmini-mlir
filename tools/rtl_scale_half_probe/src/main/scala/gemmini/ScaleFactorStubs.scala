package gemmini

import chisel3._
import chisel3.util._

class ScalingFactorWriteReq(addrWidth: Int, dataWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val data = UInt(dataWidth.W)
}

class ScalingFactorCntl(maxBlock: Int) extends Bundle {
  val counter_a = UInt(log2Up(maxBlock).W)
  val counter_b = UInt(log2Up(maxBlock).W)
  val fire_a = Bool()
  val fire_b = Bool()
  val baseAddress_act = UInt(32.W)
  val baseAddress_w = UInt(32.W)
  val scale_mem_read_w_sel = UInt(1.W)
  val scale_mem_read_act_sel = UInt(1.W)
  val loop_bound_i = UInt(9.W)
  val loop_bound_j = UInt(9.W)
  val loop_bound_k = UInt(9.W)
  val scale_mem_counter_reset_flag = Bool()
}
