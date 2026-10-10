package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ScaleHalfWaveSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Nicolas ScalingFactorMem FP6 wave alternation"

  it should "select activation and weight halves independently" in {
    test(new ScalingFactorMem(depth = 16, sramWidth = 128, numBanks = 8,
                              meshRows = 4, tileRows = 1)) { dut =>
      dut.io.dataType.poke(1.U)
      dut.io.mx_multi_elem.poke(true.B)
      dut.io.mx_multi_elem_act.poke(true.B)
      dut.io.mx_fp8_altfmt.poke(false.B)
      dut.io.scaleMemCntl.counter_a.poke(0.U)
      dut.io.scaleMemCntl.counter_b.poke(0.U)
      dut.io.scaleMemCntl.fire_a.poke(false.B)
      dut.io.scaleMemCntl.fire_b.poke(false.B)
      dut.io.scaleMemCntl.baseAddress_act.poke(0.U)
      dut.io.scaleMemCntl.baseAddress_w.poke(0.U)
      dut.io.scaleMemCntl.scale_mem_read_act_sel.poke(0.U)
      dut.io.scaleMemCntl.scale_mem_read_w_sel.poke(0.U)
      dut.io.scaleMemCntl.loop_bound_i.poke(1.U)
      dut.io.scaleMemCntl.loop_bound_j.poke(1.U)
      dut.io.scaleMemCntl.loop_bound_k.poke(1.U)
      dut.io.scaleMemCntl.scale_mem_counter_reset_flag.poke(false.B)
      dut.io.counter_i.poke(0.U)
      dut.io.counter_j.poke(0.U)
      dut.io.counter_k.poke(0.U)
      dut.io.i.poke(0.U)
      dut.io.j.poke(0.U)
      dut.io.k.poke(0.U)
      dut.io.scale_mem_write_act.valid.poke(false.B)
      dut.io.scale_mem_write_w.valid.poke(false.B)
      dut.io.scale_mem_write_act.bits.addr.poke(0.U)
      dut.io.scale_mem_write_w.bits.addr.poke(0.U)
      dut.io.scale_mem_write_act.bits.data.poke(0.U)
      dut.io.scale_mem_write_w.bits.data.poke(0.U)
      dut.io.read_req.valid.poke(false.B)
      dut.io.read_req.bits.addr.poke(0.U)
      dut.io.read_req.bits.scaling_enable.poke(true.B)
      dut.io.read_resp.ready.poke(true.B)
      dut.clock.step(2)

      def word(byte: Int): BigInt = (0 until 8).foldLeft(BigInt(0))(
        (acc, lane) => acc | (BigInt(byte) << (8 * lane)))
      def write(activation: Boolean, address: Int, byte: Int): Unit = {
        val port = if (activation) dut.io.scale_mem_write_act else dut.io.scale_mem_write_w
        port.bits.addr.poke(address.U)
        port.bits.data.poke(word(byte).U)
        port.valid.poke(true.B)
        port.ready.expect(true.B)
        dut.clock.step(2)
        port.valid.poke(false.B)
        dut.clock.step(1)
      }
      for ((activation, half, value) <- Seq(
          (true, 0, 10), (true, 1, 40),
          (false, 0, 1), (false, 1, 4))) {
        for (bank <- 0 until 2) write(activation, half * 512 + bank * 16, value)
      }
      for ((actHalf, wHalf, expected) <- Seq(
          (0, 0, 11), (0, 1, 14), (1, 0, 41), (1, 1, 44), (0, 0, 11))) {
        dut.io.scaleMemCntl.scale_mem_read_act_sel.poke(actHalf.U)
        dut.io.scaleMemCntl.scale_mem_read_w_sel.poke(wHalf.U)
        dut.io.read_req.valid.poke(true.B)
        dut.clock.step(1)
        dut.io.read_req.valid.poke(false.B)
        dut.io.read_resp.valid.expect(true.B)
        val got = (dut.io.read_resp.bits.combined_scales(0).peek().litValue & 0x1ff).toInt
        assert(got == expected, s"act=$actHalf weight=$wHalf got=$got expected=$expected")
        println(s"act=$actHalf weight=$wHalf combined_e8m0=$got")
        dut.clock.step(1)
        for (_ <- 1 until 4) {
          dut.io.read_req.valid.poke(true.B)
          dut.clock.step(1)
          dut.io.read_req.valid.poke(false.B)
          dut.io.read_resp.valid.expect(true.B)
          dut.clock.step(1)
        }
      }
    }
  }
}
