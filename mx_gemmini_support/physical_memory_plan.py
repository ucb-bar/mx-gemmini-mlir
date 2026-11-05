"""Static row liveness for a typed, ordered MX command graph.

This is a physical counterpart to Merlin's DispatchProgram/arena_plan: values
have stable identities, views alias storage, and last use controls reuse. MX
commands can complete after issue, so an access remains live through the fence
that explicitly completes its queue. The pass never guesses an unknown
resource capacity, command effect, or opaque dispatch footprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

from .command_ir import Command, Fence, VPU_OPCODES
from .target_profile import profile_sha256


class PhysicalMemoryError(ValueError):
    """The selected program has no provably safe physical row placement."""


SPACES = frozenset({"scratchpad", "accumulator", "scale_a", "scale_b", "lut", "host"})

# The packed single-throughput formula below was audited against this exact
# Gemmini source. Re-audit before accepting another source revision or geometry.
_PACKED_ACC_GEMMINI_REVISION = "12912323f1aa66d7b28687f01faf0a33af455f6f"
_PACKED_ACC_SOURCE_SHA256 = {
    "src/main/scala/gemmini/LoopMatmul.scala":
        "151535eb0ddd4154e713c253c86145eef91cd0627f1b420b178890e7bb598991",
    "src/main/scala/gemmini/MxGeom.scala":
        "f325b6ebd9d28dd789795f9e00568d661fbf6625b50b8a510bd76b0e793cc1e9",
    "src/main/scala/gemmini/ExecuteController.scala":
        "f72641f3bc338d8c9b30d7a3dcc53bfff0871c89d8dd17a263028f67fd695c72",
    "src/main/scala/gemmini/AccumulatorMem.scala":
        "e7e8834ef9b44e13656f0a0982baf75dd6327cb97ef22763db35321299704ee6",
}


def _single_fp8_accumulator_rows(tiles_i: int, tiles_j: int) -> int:
    """DIM16 LOOP_WS packed accumulator extent for direct E4M3 operands.

    LoopMatmulExecute uses ``c_addr=(i*round_up(J,4)+j)*4``. The execute
    controller aligns that address to a 16-row window and writes the 16 mesh
    rows with four output column tiles sharing it. LoopMatmulStCSpad reads
    ``ceil(J/4)`` 16-row windows per i-tile. Both extents must agree.
    """
    if (type(tiles_i) is not int or type(tiles_j) is not int or
            tiles_i <= 0 or tiles_j <= 0):
        raise PhysicalMemoryError("packed accumulator needs positive tile counts")
    rounded_j = (tiles_j + 3) // 4 * 4
    execute_end = max((((i * rounded_j + j) * 4) // 16) * 16 + 16
                      for i in range(tiles_i) for j in range(tiles_j))
    store_end = tiles_i * ((tiles_j + 3) // 4) * 16
    if execute_end != store_end:
        raise PhysicalMemoryError("packed execute/store accumulator extents disagree")
    return execute_end


def source_bound_single_fp8_accumulator(profile: dict, *, tiles_i: int,
                                        tiles_j: int, rtl_root: Path) -> dict:
    """Certify one zero-based DIM16 direct-E4M3 LOOP_WS accumulator span.

    The caller must separately prove the emitted loop arguments, direct FP8
    selector, BF16 output, no alternate accumulator base, and completion fence.
    """
    capacity = source_bound_capacities(profile)["accumulator"]
    resources = profile["resources"]
    qualified = {
        "MxFp4E3M2E4M3VpuGemminiRocketConfig": (512, 65536),
        "MxFp4E3M2E4M3VpuAcc32GemminiRocketConfig": (256, 32768),
    }
    if (profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("name") not in qualified or
            (capacity, resources.get("accumulator_bytes")) != qualified[profile["name"]] or
            profile.get("source", {}).get("gemmini_revision") !=
            _PACKED_ACC_GEMMINI_REVISION):
        raise PhysicalMemoryError("packed accumulator needs a certified DIM16 source profile")
    rtl_root = Path(rtl_root).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(rtl_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != _PACKED_ACC_GEMMINI_REVISION:
        raise PhysicalMemoryError("packed accumulator RTL revision differs from profile")
    source_sha256 = {}
    for name, expected in _PACKED_ACC_SOURCE_SHA256.items():
        digest = hashlib.sha256((rtl_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise PhysicalMemoryError(f"packed accumulator RTL source changed: {name}")
        source_sha256[name] = digest
    rows = _single_fp8_accumulator_rows(tiles_i, tiles_j)
    if rows > capacity:
        raise PhysicalMemoryError(
            f"packed LOOP_WS needs {rows} accumulator rows; source capacity is {capacity}")
    return {"rows": rows, "capacity_rows": capacity,
            "address_base_row": 0, "tiles_i": tiles_i, "tiles_j": tiles_j,
            "tiles_j_rounded_to_four": (tiles_j + 3) // 4 * 4,
            "execute_and_store_extent_agree": True,
            "gemmini_revision": revision, "rtl_source_sha256": source_sha256,
            "scope": "DIM16 direct FP8 E4M3 operands, BF16 output, zero-based single LOOP_WS"}


@dataclass(frozen=True)
class MemoryValue:
    id: str
    space: str
    rows: int  # bytes for host; source-derived physical rows for every other space
    source: str
    fixed_row: int | None = None
    alias_of: str | None = None
    external: bool = False


@dataclass(frozen=True)
class Access:
    value: str
    offset: int
    length: int


@dataclass(frozen=True)
class MemoryEvent:
    id: str
    kind: str  # host | view | command | opaque | fence
    source: str
    reads: tuple[Access, ...] = ()
    writes: tuple[Access, ...] = ()
    command: Command | Fence | None = None
    queue: str | None = None
    completes: tuple[str, ...] = ()
    certificate: Mapping[str, str] | None = None


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_bound_capacities(profile: dict) -> dict[str, int | None]:
    """Derive row capacities from v3 source facts; LUT rows are not yet pinned."""
    if profile.get("schema") != "mx_gemmini.target_profile.v3":
        raise PhysicalMemoryError("physical row planning needs a source-bound v3 profile")
    geometry, resources = profile["geometry"], profile["resources"]
    dim = geometry.get("mesh_columns")
    if dim not in (4, 8, 16, 32) or geometry.get("mesh_rows") != dim:
        raise PhysicalMemoryError("physical row planning needs a qualified square mesh")
    spad_bytes = resources.get("scratchpad_bytes")
    spad_bits = resources.get("scratchpad_row_bits")
    acc_bytes = resources.get("accumulator_bytes")
    acc_bits = resources.get("accumulator_row_bits")
    acc_rows = resources.get("accumulator_rows")
    scale = resources.get("scale_mem_config")
    if (type(spad_bytes) is not int or type(spad_bits) is not int or
            spad_bits != dim * 8 or spad_bytes <= 0 or
            spad_bytes * 8 % spad_bits or
            type(acc_bytes) is not int or type(acc_bits) is not int or
            type(acc_rows) is not int or acc_rows <= 0 or
            acc_bytes * 8 != acc_rows * acc_bits or
            not isinstance(scale, dict)):
        raise PhysicalMemoryError("physical row capacities disagree with selected source")
    scale_bytes = scale.get("size_bytes")
    scale_line = scale.get("subbank_line_bytes")
    if (type(scale_bytes) is not int or type(scale_line) is not int or
            scale_bytes <= 0 or scale_line <= 0 or
            scale_bytes % (2 * scale_line)):
        raise PhysicalMemoryError("scale memory has no source-derived A/B row capacity")
    return {"scratchpad": spad_bytes * 8 // spad_bits,
            "accumulator": acc_rows,
            "scale_a": scale_bytes // (2 * scale_line),
            "scale_b": scale_bytes // (2 * scale_line),
            "lut": None, "host": None}


def _root(values: Mapping[str, MemoryValue], name: str) -> str:
    visited: set[str] = set()
    while True:
        if name not in values or name in visited:
            raise PhysicalMemoryError("unknown or cyclic physical value alias")
        visited.add(name)
        target = values[name].alias_of
        if target is None:
            return name
        name = target


def _access(value: Access, values: Mapping[str, MemoryValue]) -> str:
    if (type(value.offset) is not int or type(value.length) is not int or
            value.offset < 0 or value.length <= 0):
        raise PhysicalMemoryError("memory access needs a positive bounded extent")
    root = _root(values, value.value)
    if value.offset + value.length > values[root].rows:
        raise PhysicalMemoryError(f"access to {value.value} exceeds its declared extent")
    return root


def _check_command(event: MemoryEvent, values: Mapping[str, MemoryValue],
                   offsets: Mapping[str, int], capacities: Mapping[str, int | None],
                   row_bytes: int) -> None:
    """Check the currently supported physical command effects against encoded rows."""
    command = event.command
    if not isinstance(command, Command):
        raise PhysicalMemoryError("command event has no typed MX command")
    reads, writes = event.reads, event.writes
    if command.funct in (0, 7):
        if reads or writes or command.rs1.buffer or command.rs2.buffer:
            raise PhysicalMemoryError("MX configuration/flush has unexpected memory effects")
        return
    if command.funct in (2, 3):
        if (command.rs1.buffer is None or command.rs2.immediate is None or
                len(reads) != 1 or len(writes) != 1):
            raise PhysicalMemoryError("MX DMA needs one explicit host and scratchpad effect")
        word = command.rs2.immediate
        count, width, row = (word >> 48) & 0xffff, (word >> 32) & 0xffff, word & 0xffffffff
        if count == 0 or width != row_bytes:
            raise PhysicalMemoryError("MX DMA row count/width differs from selected source")
        host, spad = (reads[0], writes[0]) if command.funct == 2 else (writes[0], reads[0])
        hr, sr = _root(values, host.value), _root(values, spad.value)
        if (values[hr].space != "host" or values[sr].space != "scratchpad" or
                command.rs1.buffer != host.value or command.rs1.byte_offset != host.offset or
                offsets[sr] + spad.offset != row or spad.length != count or
                host.length < count * width):
            raise PhysicalMemoryError("MX DMA effect differs from its encoded row/buffer footprint")
        return
    if command.funct == 33:
        if command.rs1.immediate is None or command.rs2.immediate is None:
            raise PhysicalMemoryError("VPU command needs literal encoded row fields")
        opcode = command.rs2.immediate & 0xf
        if opcode not in {VPU_OPCODES[name] for name in ("muls", "exp", "adds", "rcp")}:
            raise PhysicalMemoryError("VPU operation has no checked physical effect model")
        width = max(14, (capacities["scratchpad"] - 1).bit_length())
        mask = (1 << width) - 1
        src = command.rs1.immediate & mask
        src2 = (command.rs1.immediate >> width) & mask
        dst = (command.rs1.immediate >> (2 * width)) & mask
        count = (command.rs1.immediate >> (3 * width)) & 0xffff
        if (len(reads) != 1 or len(writes) != 1 or src2 != 0 or count <= 0 or
                any(values[_root(values, access.value)].space != "scratchpad"
                    for access in (reads[0], writes[0])) or
                offsets[_root(values, reads[0].value)] + reads[0].offset != src or
                offsets[_root(values, writes[0].value)] + writes[0].offset != dst or
                reads[0].length != count or writes[0].length != count):
            raise PhysicalMemoryError("VPU effect differs from its encoded scratchpad rows")
        return
    raise PhysicalMemoryError(f"MX funct {command.funct} has no checked physical effect model")


def plan_physical_memory(profile: dict, values: tuple[MemoryValue, ...],
                         events: tuple[MemoryEvent, ...], *,
                         bindings: Mapping[str, str]) -> dict:
    """Place physical rows and prove no live allocations overlap.

    Command and opaque effects complete at a named fence. A fence releases only
    queues listed in ``completes``; a pending access with no matching fence is
    refused. Host views and materializations complete at issue.
    """
    capacities = source_bound_capacities(profile)
    if (not events or not values or len({v.id for v in values}) != len(values) or
            len({e.id for e in events}) != len(events)):
        raise PhysicalMemoryError("physical graph needs unique values and events")
    table = {v.id: v for v in values}
    for value in values:
        if (not value.id or not value.source or value.space not in SPACES or
                type(value.rows) is not int or value.rows <= 0 or
                (value.fixed_row is not None and
                 (type(value.fixed_row) is not int or value.fixed_row < 0))):
            raise PhysicalMemoryError("physical value has unknown source, space, or extent")
        root = _root(table, value.id)
        if value.alias_of is not None and (value.fixed_row is not None or value.external or
                value.space != table[root].space or value.rows != table[root].rows):
            raise PhysicalMemoryError("view must alias an equal-extent value in the same space")
        capacity = capacities[value.space]
        if value.space == "lut" or (capacity is None and value.space != "host"):
            raise PhysicalMemoryError("requested LUT/unknown physical capacity is unqualified")
        if (value.fixed_row is not None and capacity is not None and
                value.fixed_row + value.rows > capacity):
            raise PhysicalMemoryError(f"fixed {value.id} rows exceed source capacity")
    profile_hash = profile_sha256(profile)
    if bindings.get("profile_sha256") != profile_hash:
        raise PhysicalMemoryError("physical graph profile binding differs")
    definitions = {v.id: -1 for v in values if v.external and v.alias_of is None}
    end_use: dict[str, int] = {}
    pending: dict[str, list[tuple[str, int]]] = {}
    aliases_defined: dict[str, int] = {}
    for i, event in enumerate(events):
        if not event.id or not event.source or event.kind not in {
                "host", "view", "command", "opaque", "fence"}:
            raise PhysicalMemoryError("physical event has unknown source or kind")
        if event.kind == "fence":
            if (not isinstance(event.command, Fence) or event.reads or event.writes or
                    event.queue is not None or not event.completes):
                raise PhysicalMemoryError("completion fence needs explicit queue set")
            for queue in event.completes:
                if queue not in pending:
                    raise PhysicalMemoryError(f"fence completes unknown queue {queue}")
                for root, _issued in pending.pop(queue):
                    end_use[root] = max(end_use.get(root, -1), i)
            continue
        if event.kind == "view":
            if (event.command is not None or event.queue or event.completes or
                    event.reads or len(event.writes) != 1):
                raise PhysicalMemoryError("view needs one alias definition and no memory effect")
            alias = event.writes[0]
            if (alias.value not in table or table[alias.value].alias_of is None or
                    alias.value in aliases_defined or alias.offset != 0 or
                    alias.length != table[alias.value].rows or
                    _root(table, alias.value) not in definitions):
                raise PhysicalMemoryError("view alias has no earlier storage definition")
            aliases_defined[alias.value] = i
            root = _root(table, alias.value)
            end_use[root] = max(end_use.get(root, -1), i)
            continue
        if event.kind == "host":
            if event.command is not None or event.queue or event.completes or event.certificate:
                raise PhysicalMemoryError("host event must complete at issue")
        elif event.kind == "command":
            if not isinstance(event.command, Command) or not event.queue or event.completes:
                raise PhysicalMemoryError("MX command needs a named completion queue")
        elif event.kind == "opaque":
            cert = event.certificate
            if (event.command is not None or not event.queue or event.completes or
                    not cert or set(cert) != {"profile_sha256", "plan_sha256",
                                               "driver_sha256", "effects_sha256"} or
                    any(cert[key] != bindings.get(key) for key in (
                        "profile_sha256", "plan_sha256", "driver_sha256")) or
                    cert["effects_sha256"] != _digest({
                        "source": event.source,
                        "reads": [asdict(x) for x in event.reads],
                        "writes": [asdict(x) for x in event.writes]})):
                raise PhysicalMemoryError("opaque dispatch lacks exact certified effects/completion")
        roots_read: list[str] = []
        for access in event.reads:
            root = _access(access, table)
            if (access.value in table and table[access.value].alias_of is not None and
                    access.value not in aliases_defined):
                raise PhysicalMemoryError("read precedes view alias definition")
            if root not in definitions:
                raise PhysicalMemoryError(f"read of {root} precedes definition")
            roots_read.append(root)
        roots_written: list[str] = []
        for access in event.writes:
            root = _access(access, table)
            if table[access.value].alias_of is not None:
                raise PhysicalMemoryError("writes through a view alias are not qualified")
            definitions.setdefault(root, i)
            roots_written.append(root)
        if (event.kind == "host" and any(
                table[root].space != "host" for root in roots_read + roots_written)):
            raise PhysicalMemoryError("host materialization cannot access on-chip rows directly")
        for root in set(roots_read + roots_written):
            outstanding = {queue for queue, accesses in pending.items()
                           if any(value == root for value, _issued in accesses)}
            if (outstanding and (event.kind == "host" or
                                 outstanding - {event.queue})):
                raise PhysicalMemoryError(
                    f"{event.id} accesses {root} before its async producer completes")
        for root in roots_read + roots_written:
            if event.kind in {"command", "opaque"}:
                pending.setdefault(event.queue, []).append((root, i))
            else:
                end_use[root] = max(end_use.get(root, -1), i)
    if pending:
        raise PhysicalMemoryError(f"async physical effects lack a completion fence: {sorted(pending)}")
    for value in values:
        if value.alias_of is None and value.id not in definitions:
            raise PhysicalMemoryError(f"physical value {value.id} is never defined")
        if value.alias_of is not None and value.id not in aliases_defined:
            raise PhysicalMemoryError(f"view alias {value.id} is never defined")
    all_roots = [v for v in values if v.alias_of is None]
    roots = [v for v in all_roots if v.space != "host"]
    intervals = {v.id: (definitions[v.id], end_use.get(v.id, definitions[v.id]))
                 for v in roots}
    order = sorted(roots, key=lambda v: (intervals[v.id][0], v.id))
    offsets: dict[str, int] = {}
    for value in order:
        capacity = capacities[value.space]
        assert capacity is not None
        candidates = ([value.fixed_row] if value.fixed_row is not None
                      else range(0, capacity - value.rows + 1))
        for start in candidates:
            if all(other.space != value.space or
                   intervals[other.id][1] < intervals[value.id][0] or
                   intervals[value.id][1] < intervals[other.id][0] or
                   start + value.rows <= offsets[other.id] or
                   offsets[other.id] + other.rows <= start
                   for other in order if other.id in offsets):
                offsets[value.id] = start
                break
        if value.id not in offsets:
            raise PhysicalMemoryError(f"{value.id} overlaps live physical rows or exceeds capacity")
    placements = {value.id: {"space": value.space, "row": offsets[value.id],
                             "rows": value.rows, "source": value.source,
                             "defined_at": intervals[value.id][0],
                             "last_use_completed_at": intervals[value.id][1]}
                  for value in roots}
    for value in all_roots:
        if value.space == "host":
            placements[value.id] = {
                "space": "host", "row": None, "bytes": value.rows,
                "source": value.source, "external": value.external,
                "defined_at": definitions[value.id],
                "last_use_completed_at": end_use.get(value.id, definitions[value.id])}
    for value in values:
        if value.alias_of is not None:
            placements[value.id] = {"space": value.space, "alias_of": _root(table, value.id),
                                     "source": value.source, "rows": value.rows,
                                     "defined_at": aliases_defined[value.id],
                                     "last_use_completed_at": end_use.get(
                                         _root(table, value.id), aliases_defined[value.id])}
    for event in events:
        if event.kind == "command":
            _check_command(event, table, offsets, capacities,
                           profile["geometry"]["mesh_columns"])
    peaks = {}
    for space in ("scratchpad", "accumulator", "scale_a", "scale_b", "lut"):
        members = [v for v in roots if v.space == space]
        high_water = max((offsets[v.id] + v.rows for v in members), default=0)
        live_peak = max((sum(v.rows for v in members
                             if intervals[v.id][0] <= i <= intervals[v.id][1])
                         for i in range(len(events))), default=0)
        peaks[space] = {"capacity_rows": capacities[space], "high_water_rows": high_water,
                        "peak_live_rows": live_peak, "active_values": len(members)}
    return {"schema": "mx_gemmini.physical_memory_plan.v1",
            "profile_sha256": profile_hash,
            "bindings": dict(bindings),
            "placement": placements, "peaks": peaks,
            "events": [{"id": e.id, "kind": e.kind, "source": e.source,
                        "queue": e.queue, "completes": list(e.completes),
                        "reads": [asdict(x) for x in e.reads],
                        "writes": [asdict(x) for x in e.writes],
                        "command": (asdict(e.command) if isinstance(e.command, Command)
                                    else "fence" if isinstance(e.command, Fence) else None)}
                       for e in events],
            "status": "fixed_and_dynamic_rows_verified_no_live_overlap"}
