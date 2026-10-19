"""Reject explicit failures reported by the selected Spike timing provider."""


def reject_deadlocked_timing(console: str) -> None:
    """An exit-zero functional result cannot authorize deadlocked cycle counts.

    The provider writes this diagnostic at fence or process completion. Stderr
    may be interleaved with a partial guest console line. Absence of this marker
    is not, by itself, a proof of complete modeled device work.
    """
    if "gemmini perf: MODEL DEADLOCK" in console:
        raise ValueError("Spike timing model deadlocked; cycle qualification refused")
