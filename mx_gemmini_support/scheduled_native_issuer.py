"""MX-owned native-loop instruction set and Merlin ``mk`` schedule lowering.

The schedule is the codegen input: changing its loops or calls changes the
issued commands. The target profile supplies capacities and format legality;
Merlin supplies only target-neutral IR, checking, and C emission.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planned_runtime_issuer import PlannedRuntimeIssuer
from .source_gemm import plan_mx_gemm
from .target_profile import profile_sha256


@dataclass(frozen=True)
class NativeSchedule:
    issuer: PlannedRuntimeIssuer
    text: str
    digest: str
    instruction_set_digest: str
    chunks: tuple[int, int, int]


def _pick_tiles(profile: dict, shape: tuple[int, int, int], precision: str) -> tuple[int, int, int]:
    m, n, k = shape
    dim = profile["geometry"]["mesh_columns"]
    pe = dim * (2 if precision == "FP4" else 1)
    resources = profile["resources"]
    if (profile.get("transport") != "rocket_rocc" or dim != 16 or
            resources["accumulator_banks"] != 2 or
            resources["scale_mem_config"]["size_bytes"] != 16384 or
            resources["scale_block_elements"] != 32 or
            precision not in {"FP4", "FP8"} or
            any(type(x) is not int or x <= 0 for x in shape) or
            m % pe or n % pe or k % 32):
        raise ValueError("MX native schedule needs aligned DIM16 direct FP4/FP8 Rocket shape")
    # reference mx_native.h: one 256-row accumulator half; two 8192-row
    # scratchpad halves. An FP4 operand uses half the FP8 scratchpad rows.
    half_rows = resources["scratchpad_bytes"] // dim // 2
    if half_rows != 8192 or resources["accumulator_bytes"] != 65536:
        raise ValueError("MX native schedule has no qualified scratchpad/accumulator geometry")
    plan_mx_gemm(shape=(pe, pe, 32), tile=(pe, pe, 32),
                 datatype=precision, quant_output=False, acc_to_gmem=False,
                 scratchpad_bytes=resources["scratchpad_bytes"], profile=profile)
    for tm in (m, 128, 64, 32, 16):
        if tm > m or tm < pe or m % tm:
            continue
        for nc in (512, 256, 128, 64, 32, 16):
            if nc < pe or n % nc or tm * nc // (dim * 4) > 256:
                continue
            for kt in (512, 256, 128, 64, 32):
                factor = 32 if precision == "FP4" else 16
                if (k % kt == 0 and (tm + nc) * kt // factor <= half_rows and
                        max(tm, nc) * (kt // 32) <= 4096):
                    return tm, nc, kt
    raise ValueError("MX native schedule has no legal M/N/K tile")


def validate_native_tiles(profile: dict, shape: tuple[int, int, int], precision: str,
                          tiles: tuple[int, int, int]) -> None:
    """Check an explicit native partition; this grants no numerical equivalence."""
    _pick_tiles(profile, shape, precision)  # Keep the original profile/mode admission.
    if (type(tiles) is not tuple or len(tiles) != 3
            or any(type(value) is not int or value <= 0 for value in tiles)):
        raise ValueError("native tiles require three positive integer extents")
    tm, nc, kt = tiles
    dim = profile["geometry"]["mesh_columns"]
    pe = dim * (2 if precision == "FP4" else 1)
    resources = profile["resources"]
    if (any(extent % tile for extent, tile in zip(shape, tiles))
            or tm % pe or nc % pe or kt % resources["scale_block_elements"]):
        raise ValueError("native tiles must divide complete mesh and scale groups")
    bits = 4 if precision == "FP4" else 8
    if ((tm + nc) * kt * bits // 8 > resources["scratchpad_bytes"] // 2
            or tm * nc * 2 > resources["accumulator_bytes"] // resources["accumulator_banks"]
            or max(tm, nc) * (kt // resources["scale_block_elements"])
               > resources["scale_mem_config"]["size_bytes"] // 4):
        raise ValueError("native tiles exceed scratchpad, accumulator or scale capacity")


def select_native_tiles(profile: dict, shape: tuple[int, int, int], precision: str):
    """Search a structural movement proxy while retaining the original K partition.

    Byte counts describe encoded input/scale traffic, not cache misses or cycles.
    The selected schedule remains experimental and needs native qualification.
    """
    from dataclasses import asdict

    from merlin.sched.tile_search import select_divisor_tiles

    original = _pick_tiles(profile, shape, precision)
    if precision != "FP8":
        return original, {"selection": "legacy_unqualified_precision", "performance": "UNMEASURED"}
    m, n, k = shape
    dim = profile["geometry"]["mesh_columns"]
    group = profile["resources"]["scale_block_elements"]

    def admit(tile):
        try:
            validate_native_tiles(profile, shape, precision, tile)
        except ValueError:
            return False
        return True

    def work(tile):
        tm, nc, kt = tile
        loops = (m // tm) * (n // nc) * (k // kt)
        data = loops * (tm + nc) * kt
        scales = loops * (tm + nc) * (kt // group)
        return data + scales, 5 + 8 * loops, (tm + nc) * (kt + kt // group), tm * nc * 2

    result = select_divisor_tiles(shape, (dim, dim, group), original, admit=admit, work=work, frozen_axes=(2,))
    return result.selected, {"selection": "experimental_nonincreasing_input_bytes_and_commands",
                             "axes": ["encoded_input_and_scale_bytes", "issued_commands",
                                      "max_single_launch_input_and_scale_bytes", "live_accumulator_bytes"],
                             "frozen_axes": [2], "performance": "UNMEASURED", **asdict(result)}


def emit_scheduled_native_issuer(profile: dict, shape_mnk: tuple[int, int, int], *,
                                 precision: str, symbol: str = "mx_issue_encoded",
                                 tile_mnk: tuple[int, int, int] | None = None,
                                 tile_policy: str = "legacy") -> NativeSchedule:
    """Lower one padded dense shape to checked native DRAM-loop commands."""
    from merlin.sched.check.static import check_kernel
    from merlin.sched.codegen import emit_c_function
    from merlin.sched.ir.expr import Var, select
    from merlin.sched.ir.kernel import Kernel, Ptr, TensorArg, call, loop
    from merlin.sched.ir.parse import parse_kernel
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    if not symbol.isidentifier():
        raise ValueError("MX native issuer needs a C identifier")
    m, n, k = shape_mnk
    if tile_policy not in {"legacy", "movement"} or (tile_policy != "legacy" and tile_mnk is not None):
        raise ValueError("native tile policy must be legacy or movement, without an override conflict")
    selection = None
    if tile_policy == "movement":
        tiles, selection = select_native_tiles(profile, shape_mnk, precision)
    else:
        tiles = _pick_tiles(profile, shape_mnk, precision) if tile_mnk is None else tile_mnk
    validate_native_tiles(profile, shape_mnk, precision, tiles)
    tm, nc, kt = tiles
    fp4 = precision == "FP4"
    pe = 32 if fp4 else 16
    kts, chunks, rows = k // kt, n // nc, m // tm
    bits = 4 if fp4 else 8
    lda, ldb, ldc = k, n // (2 if fp4 else 1), n
    lengths = {
        "activation": m * k * bits // 8,
        "activation_scales": m * k // 32,
        "weight": k * n * bits // 8,
        "weight_scales": k * n // 32,
        "output_bf16": m * n * 2,
        "scratch_output_scales": 2048,
    }
    if any(x <= 0 or x > 0x7fffffff for x in lengths.values()):
        raise ValueError("MX native issuer buffer extent exceeds runtime ABI")
    fmt = 2 if fp4 else 0
    config_ex = (1 << 16) | (fmt << 12) | (fmt << 10) | (1 << 2) | (3 << 14)

    def render_config(_args):
        return ("MX_CMD(7, 0, 0);\n"
                f"    MX_CMD(0, UINT64_C({config_ex}), UINT64_C(1) << 48);\n"
                f"    MX_CMD(0, UINT64_C({(16 << 16) | (1 << 8) | 1}), UINT64_C({lda}));\n"
                f"    MX_CMD(0, UINT64_C({(16 << 16) | (1 << 8) | (1 << 3) | 1}), UINT64_C({ldb}));\n"
                f"    MX_CMD(0, 2, UINT64_C({ldc * (4 if fp4 else 2)}));")

    def render_loop(args):
        a, b, c, asc, bsc, write, accumulate, slot = args
        return ("do {\n"
                f"      MX_CMD(31, (uint64_t)(uintptr_t){asc}, (uint64_t)(uintptr_t){bsc});\n"
                f"      MX_CMD(32, UINT64_C({m}), UINT64_C({n}));\n"
                f"      MX_CMD(9, 0, UINT64_C({(kt // 16 << 32) | (nc // pe << 16) | (tm // pe)}));\n"
                f"      MX_CMD(10, (uint64_t)(uintptr_t){a}, (uint64_t)(uintptr_t){b});\n"
                f"      MX_CMD(11, 0, ({write}) ? (uint64_t)(uintptr_t){c} : 0);\n"
                f"      MX_CMD(12, UINT64_C({lda}), UINT64_C({ldb}));\n"
                f"      MX_CMD(13, 0, UINT64_C({ldc}));\n"
                f"      MX_CMD(8, (UINT64_C(1) << 18) | ((uint64_t)({slot}) << 16) | "
                f"(uint64_t)({accumulate}), 0);\n"
                "    } while (0);")

    def check_loop(values, state):
        errors = []
        if not state.get("configured"):
            errors.append("native loop has no selected MX configuration")
        if values["slot"] not in (1, 2):
            errors.append("native loop must use scratchpad half 1 or 2")
        if state.get("last_slot") == values["slot"]:
            errors.append("consecutive native loops reuse one scratchpad half")
        state["last_slot"] = values["slot"]
        if values["write"] not in (0, 1) or values["accumulate"] not in (0, 1):
            errors.append("native loop has invalid store/accumulate flag")
        return errors

    def footprints(v):
        # The DMA traverses strided rows. Bounding each row family also checks
        # the final byte, without pretending the transfer is contiguous.
        result = [
            ("a", (tm // (2 if fp4 else 1) - 1) * lda + kt, "read"),
            ("b", (kt - 1) * ldb + nc // (2 if fp4 else 1), "read"),
            ("asc", (kt // 32 - 1) * m + tm, "read"),
            ("bsc", (kt // 32 - 1) * n + nc, "read"),
        ]
        if v["write"]:
            result.append(("c", (tm - 1) * n * 2 + nc * 2, "write"))
        return result

    names = ("a", "b", "c", "asc", "bsc", "write", "accumulate", "slot")
    kinds = ("ptr", "ptr", "ptr", "ptr", "ptr", "flag", "flag", "int")
    iset = InstructionSet(
        "mx_gemmini_native", {
            "configure": InstrDef("configure", (), render_config,
                                  check=lambda _v, s: (s.__setitem__("configured", True) or [])),
            "native_loop": InstrDef(
                "native_loop", tuple(Operand(a, b) for a, b in zip(names, kinds)),
                render_loop, check=check_loop, footprint=footprints),
        }, {"i8": "uint8_t", "i16": "uint16_t"}, "MX_FENCE();",
        facts={"profile_sha256": profile_sha256(profile), "precision": precision,
               "tm": tm, "nc": nc, "kt": kt},
        provenance={"recipe": "nicolas mx_native.h native DRAM loops"},
    )

    row, chunk, wave = Var("row"), Var("chunk"), Var("wave")
    body = (loop("row", rows, loop("chunk", chunks, loop("wave", kts, call(
        "native_loop",
        a=Ptr("activation", row * (tm * lda // (2 if fp4 else 1)) + wave * kt),
        b=Ptr("weight", wave * (kt * ldb) + chunk * (nc // (2 if fp4 else 1))),
        c=Ptr("output_bf16", row * (tm * n * 2) + chunk * (nc * 2)),
        asc=Ptr("activation_scales", wave * (kt // 32 * m) + row * tm),
        bsc=Ptr("weight_scales", wave * (kt // 32 * n) + chunk * nc),
        write=select(wave, kts - 1, 1, 0),
        accumulate=select(wave, 0, 0, 1),
        slot=1 + ((row * (chunks * kts) + chunk * kts + wave) % 2))))),)
    kernel = Kernel(symbol + "_schedule", tuple(
        TensorArg(name, (size // (2 if name == "output_bf16" else 1),),
                  "i16" if name == "output_bf16" else "i8",
                  "write" if name == "output_bf16" else "read")
        for name, size in lengths.items()),
        (call("configure"), *body),
        attrs=(("precision", precision), ("tm", str(tm)),
               ("nc", str(nc)), ("kt", str(kt))))
    errors = check_kernel(kernel, iset, max_instances=2_000_000)
    if errors:
        raise ValueError("MX native schedule fails static check: " + "; ".join(errors[:5]))
    if parse_kernel(kernel.text()) != kernel:
        raise ValueError("MX native schedule failed canonical IR round trip")
    inner = emit_c_function(kernel, iset, symbol=symbol + "_scheduled")
    preamble = '''#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(uintptr_t) == 8, "MX native commands require RV64");
#ifdef MX_TRACE_COMMANDS
extern void mx_trace_command(uint32_t funct, uint64_t rs1, uint64_t rs2);
extern void mx_trace_fence(void);
#define MX_CMD(F, A, B) mx_trace_command((F), (uint64_t)(A), (uint64_t)(B))
#define MX_FENCE() mx_trace_fence()
#else
#define MX_CMD(F, A, B) __asm__ volatile (".insn r 0x7b, 3, " #F ", x0, %0, %1" \\
    : : "r"((uint64_t)(A)), "r"((uint64_t)(B)) : "memory")
#define MX_FENCE() __asm__ volatile ("fence" ::: "memory")
#endif
'''
    wrapper = f'''\nvoid {symbol}(const void *activation, const void *activation_scales,
              const void *weight, const void *weight_scales,
              const void *output_bf16, const void *scratch_output_scales) {{
  if (!activation || !activation_scales || !weight || !weight_scales ||
      !output_bf16 || !scratch_output_scales ||
      (((uintptr_t)activation_scales | (uintptr_t)weight_scales) & 7)) __builtin_trap();
  {symbol}_scheduled((uint8_t *)activation, (uint8_t *)activation_scales,
                     (uint8_t *)weight, (uint8_t *)weight_scales,
                     (uint16_t *)output_bf16, (uint8_t *)scratch_output_scales);
}}
'''
    plan = {"execution_transport": "native_dram_loop", "m_tile": tm,
            "n_chunk": nc, "k_tile": kt,
            "output_layout": "row_major_bf16", "schedule_ir": "merlin.mk"}
    if selection is not None:
        plan["tile_selection"] = selection
    issuer = PlannedRuntimeIssuer(preamble + inner + wrapper, shape_mnk,
                                  (tm, nc, kt), precision, profile_sha256(profile),
                                  lengths, plan)
    return NativeSchedule(issuer, kernel.text(), kernel.digest(), iset.digest(),
                          (tm, nc, kt))
