import re
import numpy as np

def reject_deadlocked_timing(console: str) -> None:
    """An exit-zero functional result cannot authorize deadlocked cycle counts.

    The provider writes this diagnostic at fence or process completion. Stderr
    may be interleaved with a partial guest console line. Absence of this marker
    is not, by itself, a proof of complete modeled device work.
    """
    if "gemmini perf: MODEL DEADLOCK" in console:
        raise ValueError("Spike timing model deadlocked; cycle qualification refused")

def parse_console(console, checks, phase_count, phase_timing):
    """Require terminal completion, intact guards and every emitted tensor byte."""
    reject_deadlocked_timing(console)
    lines = console.splitlines()
    completed = [line for line in lines if line.startswith("PORT_PHASES_COMPLETED ")]
    totals = [line for line in lines if line.startswith("PORT_TOTAL ")]
    if (
        lines.count("PORT_DONE") != 1
        or completed != [f"PORT_PHASES_COMPLETED {phase_count}"]
        or len(totals) != 1
    ):
        raise ValueError("port execution or timed phase sequence did not complete")
    match = re.fullmatch(r"PORT_TOTAL ([1-9][0-9]*) guards=0", totals[0])
    if match is None:
        raise ValueError("invalid cycles or corrupted arena guards")
    outputs, phases = {}, []
    for line in lines:
        if line.startswith("PORT_PHASE "):
            _, index, operation, cycles = line.split()
            index, operation, cycles = int(index), int(operation), int(cycles)
            if index != len(phases) or cycles <= 0:
                raise ValueError("incomplete or reordered phase timing")
            phases.append({"phase": index, "operation": operation, "cycles": cycles})
        elif line.startswith("PORT_BYTES "):
            _, index, offset, count, *words = line.split()
            index, offset, count = int(index), int(offset), int(count)
            if not 0 <= index < len(checks):
                raise ValueError("unknown output tensor")
            target = outputs.setdefault(index, bytearray())
            if offset != len(target) or not 1 <= count <= 32 or count != len(words):
                raise ValueError("incomplete, duplicated or reordered tensor output")
            if any(not re.fullmatch("[0-9a-f]{2}", word) for word in words):
                raise ValueError("invalid observed tensor byte")
            target.extend(int(word, 16) for word in words)
    if len(phases) != (phase_count if phase_timing else 0):
        raise ValueError("missing or unexpected phase timings")
    if set(outputs) != set(range(len(checks))) or any(
        len(outputs[i]) != check["bytes"] for i, check in enumerate(checks)
    ):
        raise ValueError("complete output tensors are required")
    return int(match.group(1)), phases, {i: bytes(v) for i, v in outputs.items()}

def quality(logits, reference):
    if logits.shape != reference.shape or logits.shape != (33, 32000):
        raise ValueError(
            "cached quality needs every vocabulary logit at all 33 positions"
        )
    if not np.isfinite(logits).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite cached logits")

    def log_prob(x):
        x = x.astype(np.float64)
        x -= x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    p, q = log_prob(reference), log_prob(logits)
    kl = np.maximum(np.sum(np.exp(p) * (p - q), axis=1), 0)
    top = logits.argmax(axis=1) == reference.argmax(axis=1)
    return {
        "mean_kl_nats": float(kl.mean()),
        "worst_kl_nats": float(kl.max()),
        "top_token_matches": int(top.sum()),
        "positions": 33,
        "per_position_kl_nats": kl.tolist(),
        "per_position_top_token_equal": top.tolist(),
    }

def gates(candidate, source):
    return {
        "mean_kl_within_5_percent": candidate["mean_kl_nats"]
        <= 1.05 * source["mean_kl_nats"],
        "worst_kl_within_5_percent": candidate["worst_kl_nats"]
        <= 1.05 * source["worst_kl_nats"],
        "top_token_matches_at_least_source": candidate["top_token_matches"]
        >= source["top_token_matches"],
    }
