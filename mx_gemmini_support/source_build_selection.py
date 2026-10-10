"""Read the MX GEMM source driver's selection from Radiance's Makefile."""

from __future__ import annotations

from pathlib import Path
import re


def selected_mx_gemm_drivers(makefile: str) -> set[str]:
    """Return exactly the C++ filenames in Radiance's continued MU_SRCS value."""
    lines = makefile.splitlines()
    assignments = [i for i, line in enumerate(lines)
                   if re.match(r"^MU_SRCS\s*=", line)]
    if len(assignments) != 1:
        raise ValueError("Radiance MX Makefile needs one MU_SRCS assignment")
    i = assignments[0]
    value = lines[i].split("=", 1)[1].strip()
    parts = []
    while True:
        continued = value.endswith("\\")
        parts.extend(value.rstrip("\\").split())
        if not continued:
            break
        i += 1
        if i >= len(lines) or lines[i].lstrip().startswith("#"):
            raise ValueError("Radiance MX MU_SRCS continuation is incomplete")
        value = lines[i].strip()
    if not parts or any(re.fullmatch(r"[A-Za-z0-9_.-]+\.cpp", part) is None
                        for part in parts) or len(set(parts)) != len(parts):
        raise ValueError("Radiance MX MU_SRCS contains an unsupported or duplicate driver")
    return set(parts)


def source_makefile_selection(source_root: Path, drivers: set[str]) -> tuple[set[str], bytes]:
    """Check that named source drivers exist and read their Makefile selection."""
    directory = source_root / "kernels/gemm_mxgemmini"
    makefile = (directory / "Makefile").read_bytes()
    available = {path.name for path in directory.glob("mxgemm.fp*.cpp")}
    if drivers != available:
        raise ValueError("MX roster and Radiance's named source drivers differ")
    selected = selected_mx_gemm_drivers(makefile.decode("utf-8"))
    if not selected <= {path.name for path in directory.glob("*.cpp")}:
        raise ValueError("MX Makefile selects a missing source driver")
    return selected & drivers, makefile
