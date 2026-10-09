"""Bind an existing model2MLIR MX handoff to one exact RTL target profile.

The legacy handoff still describes symmetric operand quantization. This
binding preserves its operand formats; asymmetric quantization needs a new
frontend manifest and cannot be inferred from the selected hardware.
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path

from .target_profile import load_profile, profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


_LEGACY_FORMATS = {"mxfp4": "fp4_e2m1", "mxfp6": "fp6_e3m2", "mxfp8": "fp8_e4m3"}
_CAPTURED_PROJECTION = {"mxfp4": "direct", "mxfp6": "lut", "mxfp8": "direct"}
_SELECTION_FIELDS = frozenset(("activation_format", "weight_format",
                               "activation_projection", "weight_projection", "pe_mode"))


def bind_handoff(mlir_text: str, profile: dict, selections: dict[str, dict] | None = None) -> str:
    """Preserve captured quantization while choosing a legal physical PE mode."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    selections = selections or {}
    if not isinstance(selections, dict) or any(not isinstance(key, str) or not isinstance(value, dict)
                                                for key, value in selections.items()):
        raise ValueError("site selections must map site IDs to compute tuples")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if "mx.profile_sha256" in module.attributes:
        raise ValueError("MX handoff is already profile-bound")
    digest = profile_sha256(profile)
    module.attributes["mx.profile_sha256"] = StringAttr(digest)
    selected_sites: dict[str, dict] = {}
    contracts = 0

    for op in module.walk():
        if _operation_name(op) != "mx_gemmini.contract":
            continue
        site = _text_attr(op, "site_id")
        legacy = _text_attr(op, "format")
        if not site or legacy not in _LEGACY_FORMATS:
            raise ValueError("profile binder needs legacy symmetric contracts with site IDs")
        fmt = _LEGACY_FORMATS[legacy]
        projection = _CAPTURED_PROJECTION[legacy]
        cells = [cell for cell in profile["legal_compute"]
                 if cell["activation_format"] == cell["weight_format"] == fmt and
                 cell["activation_projection"] == cell["weight_projection"] == projection]
        choice = selections.get(site)
        if choice is not None:
            if set(choice) != _SELECTION_FIELDS:
                raise ValueError(f"{site}: selection needs exactly the five compute tuple fields")
            cells = [cell for cell in cells if cell == choice]
        if len(cells) != 1:
            raise ValueError(f"{site}: {len(cells)} legal {fmt}/{projection} choices; "
                             "the capture cannot change operand projection")
        if site in selected_sites and selected_sites[site] != cells[0]:
            raise ValueError(f"{site}: conflicting repeated contraction choices")
        selected_sites[site] = cells[0]
        del op.attributes["format"]
        for key, value in cells[0].items():
            op.attributes[key] = IntegerAttr(value, 32) if key == "pe_mode" else StringAttr(value)
        op.attributes["profile_sha256"] = StringAttr(digest)
        contracts += 1

    if not contracts:
        raise ValueError("MX handoff has no legacy contraction")
    unknown = set(selections) - set(selected_sites)
    if unknown:
        raise ValueError(f"site selections do not occur in handoff: {sorted(unknown)}")
    for op in module.walk():
        name = _operation_name(op)
        if not name.startswith("mx_gemmini.") or name == "mx_gemmini.contract":
            continue
        site = _text_attr(op, "site_id")
        if not site or site not in selected_sites:
            raise ValueError(f"{name}: site has no selected contraction")
        cell = selected_sites[site]
        legacy = _text_attr(op, "format")
        if legacy is not None:
            if legacy not in _LEGACY_FORMATS:
                raise ValueError(f"{name}: unknown legacy output format {legacy}")
            output = _LEGACY_FORMATS[legacy]
            if name == "mx_gemmini.encode":
                if output != cell["activation_format"]:
                    raise ValueError(f"{name}: captured input format differs from selected activation")
                op.attributes["element_format"] = StringAttr(output)
                op.attributes["projection"] = StringAttr(cell["activation_projection"])
            elif name == "mx_gemmini.requantize":
                if output not in profile["candidate_output_modes"]:
                    raise ValueError(f"{name}: output format is unavailable in selected profile")
                op.attributes["output_format"] = StringAttr(output)
            else:
                raise ValueError(f"{name}: unexpected legacy format attribute")
            del op.attributes["format"]
        op.attributes["profile_sha256"] = StringAttr(digest)

    output = StringIO()
    Printer(stream=output).print_op(module)
    rendered = output.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--selections", type=Path, help="JSON map from site ID to one legal_compute tuple")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    selections = json.loads(args.selections.read_text()) if args.selections else None
    rendered = bind_handoff(args.mlir.read_text(), profile, selections)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"bound {args.mlir} to {profile['name']} ({profile_sha256(profile)}) -> {args.out}")


if __name__ == "__main__":
    main()
