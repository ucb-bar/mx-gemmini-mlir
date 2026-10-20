"""Paired source timings need complete observed arrays, not passing counters alone."""

import pytest

from tools.measure_native_kernel_pair import check_output


def completed_console():
    records = []
    for trial in range(2):
        records.append(
            f"PAIR trial={trial} native=100 compiler=90 mismatch=0 oracle=0 guard=0"
        )
        for role in ("native", "compiler"):
            records.append(f"PAIR_WORDS {role} {trial} 0 2 3f80 4000")
    return "\n".join([*records, "PAIR_DONE"])


def test_observed_words_and_timing_records_are_both_required():
    rows = check_output(completed_console(), bytes.fromhex("803f0040"), 2)
    assert [(row["native"], row["compiler"]) for row in rows] == [(100, 90), (100, 90)]


@pytest.mark.parametrize(
    "before,after",
    [
        ("PAIR_DONE", ""),
        ("PAIR_WORDS compiler 1 0 2 3f80 4000", ""),
        ("PAIR_WORDS compiler 1 0 2 3f80 4000", "PAIR_WORDS compiler 1 0 2 3f80 4001"),
        ("PAIR_WORDS compiler 1 0 2 3f80 4000", "PAIR_WORDS compiler 1 1 2 3f80 4000"),
        ("trial=1 native=100", "trial=0 native=100"),
        ("native=100", "native=0"),
        ("guard=0", "guard=1"),
    ],
)
def test_partial_corrupt_and_misordered_observations_refuse(before, after):
    with pytest.raises(ValueError):
        check_output(
            completed_console().replace(before, after), bytes.fromhex("803f0040"), 2
        )
