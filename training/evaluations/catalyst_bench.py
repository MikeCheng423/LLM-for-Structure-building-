"""Drive the fine-tuned agent over catalysts common in the literature.

This is a *capability* probe, not a promotion gate. The corpus is generated from
a fixed family grammar; this asks a different question -- when a researcher types
the kind of request that appears in a catalysis paper, does the shipped command
produce the right structure?

Two reference points per catalyst:

* the **deterministic** recipe (no LLM). If that fails, no amount of training can
  help: the tool registry cannot express the structure at all.
* the **model** run through the deployment path (``LocalModelChat`` ->
  ``AgentController`` -> ``ASEWorkspace``, full registry). A gap between the two
  is a model/data problem; a deterministic failure is a code problem.

Requests state every slot the family requires, in the wording of
``training/CORPUS_RULE.md``, so a failure is not just a dropped slot.

Usage::

    python training/evaluations/catalyst_bench.py --output report.json
    python training/evaluations/catalyst_bench.py --deterministic-only --output r.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ase_auto_build.ase_agent.controller import AgentController, ControllerState
from ase_auto_build.ase_agent.policy import AgentPolicy
from ase_auto_build.ase_agent.tools import create_default_registry
from ase_auto_build.ase_agent.validation import atoms_hash, structure_invariants
from ase_auto_build.ase_agent.workspace import ASEWorkspace

# (id, family, why it matters, request, reference recipe or None when the
#  registry cannot express it at all)
CATALYSTS: list[tuple[str, str, str, str, list[tuple[str, dict]] | None]] = [
    # ---------- elemental metal facets ----------
    ("pt-111", "surface", "ORR / CO oxidation benchmark",
     "Build a 3x3 Pt(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [3, 3, 1]}), ("finish", {"name": "slab"})]),
    ("cu-111", "surface", "CO2 reduction",
     "Build a 3x3 Cu(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Cu", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [3, 3, 1]}), ("finish", {"name": "slab"})]),
    ("cu-100", "surface", "CO2RR facet dependence",
     "Build a 2x2 Cu(100) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Cu", "crystal": "fcc", "miller": [1, 0, 0],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}), ("finish", {"name": "slab"})]),
    ("pt-211", "surface", "step-site activity",
     "Build a Pt(211) stepped surface, 6 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [2, 1, 1],
                         "layers": 6, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("ag-111", "surface", "ethylene epoxidation",
     "Build an Ag(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Ag", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("au-111", "surface", "low-temperature CO oxidation",
     "Build an Au(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Au", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("pd-111", "surface", "selective hydrogenation",
     "Build a Pd(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Pd", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("ni-111", "surface", "steam methane reforming",
     "Build a Ni(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Ni", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("rh-111", "surface", "NO reduction / three-way catalyst",
     "Build a Rh(111) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Rh", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("fe-110", "surface", "ammonia synthesis",
     "Build a bcc Fe(110) slab, 5 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Fe", "crystal": "bcc", "miller": [1, 1, 0],
                         "layers": 5, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("ru-0001", "surface", "ammonia synthesis / methanation",
     "Build an hcp Ru(001) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Ru", "crystal": "hcp", "miller": [0, 0, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("co-0001", "surface", "Fischer-Tropsch",
     "Build an hcp Co(001) slab, 4 layers, 15 A vacuum.",
     [("build_surface", {"name": "slab", "element": "Co", "crystal": "hcp", "miller": [0, 0, 1],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),

    # ---------- adsorbates ----------
    ("co-on-pt111", "molecular_adsorption", "canonical CO adsorption",
     "Build a 2x2 Pt(111) slab, 4 layers, 15 A vacuum, and adsorb CO at an ontop site "
     "1.85 A above the surface, anchored through its carbon.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("add_molecular_adsorbate", {"name": "slab", "species": "CO", "anchor": 2,
                                   "site": "ontop", "height": 1.85}), ("finish", {"name": "slab"})]),
    ("o-on-pt111", "atomic_adsorption", "ORR intermediate",
     "Build a 2x2 Pt(111) slab, 4 layers, 15 A vacuum, and adsorb O at a hollow site "
     "1.3 A above the surface.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("add_atomic_adsorbate", {"name": "slab", "element": "O", "site": "hollow", "height": 1.3}),
      ("finish", {"name": "slab"})]),
    ("h-on-pt111", "atomic_adsorption", "HER volcano",
     "Build a 2x2 Pt(111) slab, 4 layers, 15 A vacuum, and adsorb H at a hollow site "
     "1.0 A above the surface.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("add_atomic_adsorbate", {"name": "slab", "element": "H", "site": "hollow", "height": 1.0}),
      ("finish", {"name": "slab"})]),
    ("oh-on-pt111", "molecular_adsorption", "ORR / surface oxidation",
     "Build a 2x2 Pt(111) slab, 4 layers, 15 A vacuum, and adsorb OH at an ontop site "
     "1.9 A above the surface, anchored through its first atom.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("add_molecular_adsorbate", {"name": "slab", "species": "OH", "anchor": 1,
                                   "site": "ontop", "height": 1.9}), ("finish", {"name": "slab"})]),
    ("n2-on-fe110", "molecular_adsorption", "ammonia synthesis, N2 activation",
     "Build a 2x2 bcc Fe(110) slab, 5 layers, 15 A vacuum, and adsorb N2 at an ontop site "
     "1.9 A above the surface, anchored through its first atom.",
     [("build_surface", {"name": "slab", "element": "Fe", "crystal": "bcc", "miller": [1, 1, 0],
                         "layers": 5, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("add_molecular_adsorbate", {"name": "slab", "species": "N2", "anchor": 1,
                                   "site": "ontop", "height": 1.9}), ("finish", {"name": "slab"})]),
    ("frozen-pt111", "surface_constraint", "standard slab relaxation protocol",
     "Build a 2x2 Pt(111) slab, 5 layers, 15 A vacuum, and fix the bottom two layers.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 5, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("freeze_layers", {"name": "slab", "side": "bottom", "layers": 2, "axes": "xyz"}),
      ("finish", {"name": "slab"})]),

    # ---------- alloy / single atom ----------
    ("ptni-111", "substitution", "ORR alloy catalysts",
     "Build a 2x2 Pt(111) slab, 4 layers, 15 A vacuum, then replace atoms 1 and 2 with Ni.",
     [("build_surface", {"name": "slab", "element": "Pt", "crystal": "fcc", "miller": [1, 1, 1],
                         "layers": 4, "vacuum": 15.0, "repeat": [2, 2, 1]}),
      ("substitute", {"name": "slab", "selector": {"indices": [1, 2]}, "element": "Ni"}),
      ("finish", {"name": "slab"})]),
    ("pt-sac-graphene", "prototype", "single-atom catalysis",
     "Build a graphene sheet with 15 A vacuum, repeat it 3x3, and adsorb a Pt atom at a "
     "hollow site 2.0 A above the sheet.",
     [("build_prototype", {"name": "sheet", "prototype": "graphene", "vacuum": 15.0}),
      ("repeat", {"name": "sheet", "repeat": [3, 3, 1]}),
      ("add_atomic_adsorbate", {"name": "sheet", "element": "Pt", "site": "hollow", "height": 2.0}),
      ("finish", {"name": "sheet"})]),
    ("n-doped-graphene", "substitution", "metal-free ORR / SAC anchor",
     "Build a graphene sheet with 15 A vacuum, repeat it 3x3, then replace atom 1 with N.",
     [("build_prototype", {"name": "sheet", "prototype": "graphene", "vacuum": 15.0}),
      ("repeat", {"name": "sheet", "repeat": [3, 3, 1]}),
      ("substitute", {"name": "sheet", "selector": {"indices": [1]}, "element": "N"}),
      ("finish", {"name": "sheet"})]),

    # ---------- oxide bulk ----------
    ("rutile-tio2-bulk", "prototype", "photocatalysis",
     "Build bulk rutile TiO2.",
     [("build_prototype", {"name": "bulk", "prototype": "rutile-TiO2"}), ("finish", {"name": "bulk"})]),
    ("ceo2-bulk", "prototype", "oxygen storage / WGS",
     "Build bulk CeO2 in the fluorite structure.",
     [("build_prototype", {"name": "bulk", "prototype": "fluorite-CeO2"}), ("finish", {"name": "bulk"})]),
    ("nio-bulk", "prototype", "OER",
     "Build bulk NiO in the rocksalt structure.",
     [("build_prototype", {"name": "bulk", "prototype": "rocksalt-NiO"}), ("finish", {"name": "bulk"})]),
    ("mgo-bulk", "prototype", "classic oxide support",
     "Build bulk MgO in the rocksalt structure.",
     [("build_prototype", {"name": "bulk", "prototype": "rocksalt-MgO"}), ("finish", {"name": "bulk"})]),
    ("zno-bulk", "prototype", "methanol synthesis component",
     "Build bulk ZnO in the wurtzite structure.",
     [("build_prototype", {"name": "bulk", "prototype": "wurtzite-ZnO"}), ("finish", {"name": "bulk"})]),
    ("ceo2-vacancy", "vacancy", "reducible-oxide catalysis",
     "Build bulk CeO2 in the fluorite structure and remove atom 4 to make an oxygen vacancy.",
     [("build_prototype", {"name": "bulk", "prototype": "fluorite-CeO2"}),
      ("make_vacancy", {"name": "bulk", "selector": {"indices": [4]}}), ("finish", {"name": "bulk"})]),

    # ---------- oxide surfaces (need build_surface source=) ----------
    ("tio2-110-slab", "surface", "THE model photocatalytic surface",
     "Build bulk rutile TiO2, then cut a (110) surface from it, 4 layers, 15 A vacuum.",
     [("build_prototype", {"name": "bulk", "prototype": "rutile-TiO2"}),
      ("build_surface", {"name": "slab", "source": "bulk", "miller": [1, 1, 0],
                         "layers": 4, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("tio2-101-anatase-slab", "surface", "anatase photocatalysis",
     "Build bulk anatase TiO2, then cut a (101) surface from it, 3 layers, 15 A vacuum.",
     [("build_prototype", {"name": "bulk", "prototype": "anatase-TiO2"}),
      ("build_surface", {"name": "slab", "source": "bulk", "miller": [1, 0, 1],
                         "layers": 3, "vacuum": 15.0}), ("finish", {"name": "slab"})]),
    ("h2o-on-tio2-110", "molecular_adsorption", "water splitting / photocatalysis",
     "Build bulk rutile TiO2, cut a (110) surface from it with 4 layers and 15 A vacuum, "
     "then adsorb H2O at an ontop site 2.2 A above the surface, anchored through its first atom.",
     [("build_prototype", {"name": "bulk", "prototype": "rutile-TiO2"}),
      ("build_surface", {"name": "slab", "source": "bulk", "miller": [1, 1, 0],
                         "layers": 4, "vacuum": 15.0}),
      ("add_molecular_adsorbate", {"name": "slab", "species": "H2O", "anchor": 1,
                                   "site": "ontop", "height": 2.2}), ("finish", {"name": "slab"})]),

    # ---------- not expressible by the registry at all ----------
    ("mos2-bulk", "prototype", "HER edge-site catalyst",
     "Build bulk 2H MoS2.", None),
    ("srtio3-perovskite", "prototype", "photocatalysis / OER",
     "Build bulk SrTiO3 in the perovskite structure.", None),
    ("co3o4-spinel", "prototype", "OER benchmark",
     "Build bulk Co3O4 in the spinel structure.", None),
]


def deterministic(steps: list[tuple[str, dict]]) -> tuple[str, dict, str]:
    ws = ASEWorkspace(create_default_registry(), session_id="catalyst-ref")
    for tool, args in steps:
        ws.execute_or_raise(tool, args)
    atoms = ws.final_atoms()
    return atoms_hash(atoms), structure_invariants(atoms), atoms.get_chemical_formula()


def run_model(model, tokenizer, request: str, max_turns: int, max_new_tokens: int) -> dict[str, Any]:
    from ase_auto_build.ase_agent.llm_local import LocalModelChat

    registry = create_default_registry()
    policy = AgentPolicy(max_model_turns=max_turns, max_tool_calls=max_turns)
    ws = ASEWorkspace(registry, policy=policy, session_id="catalyst-model")
    chat = LocalModelChat(model, tokenizer, tool_override=None, max_new_tokens=max_new_tokens)
    started = time.monotonic()
    try:
        result = AgentController(ws, chat).run(request)
    except Exception as exc:  # routing fail-closed is a case failure, not a crash
        return {"state": "ROUTING_FAILED", "error": f"{type(exc).__name__}: {exc}",
                "hash": None, "formula": None, "calls": [], "failures": [],
                "elapsed": round(time.monotonic() - started, 2),
                "texts": chat.generated_texts}
    out: dict[str, Any] = {
        "state": result.state.value,
        "calls": [item["tool"] for item in result.transcript],
        "failures": [{"tool": item["tool"], "args": item["args"], "error": item["result"]}
                     for item in result.transcript if not item["success"]],
        "elapsed": round(time.monotonic() - started, 2),
        "texts": chat.generated_texts,
        "error": None,
        "hash": None,
        "formula": None,
        "invariants": None,
    }
    if result.state is ControllerState.FINISHED:
        atoms = ws.final_atoms()
        out["hash"] = atoms_hash(atoms)
        out["formula"] = atoms.get_chemical_formula()
        out["invariants"] = structure_invariants(atoms)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path,
                        default=Path("training/runs/pilot-qwen3-4b-r5/adapter"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--base-only", action="store_true", help="no adapter (frozen base)")
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", default=None,
                        help="run a single catalyst id, so each case gets a fresh process. "
                             "Results inside one long-lived process were not reproducible "
                             "standalone, so per-case isolation is the trustworthy mode.")
    args = parser.parse_args()

    selected = [c for c in CATALYSTS if args.only in (None, c[0])]
    if not selected:
        parser.error(f"unknown catalyst id {args.only!r}")

    rows: list[dict[str, Any]] = []
    for cid, family, why, request, steps in selected:
        row: dict[str, Any] = {"id": cid, "family": family, "why": why, "request": request}
        if steps is None:
            row["deterministic"] = {"ok": False, "error": "no expressible recipe"}
        else:
            try:
                ref_hash, ref_inv, formula = deterministic(steps)
                row["deterministic"] = {"ok": True, "hash": ref_hash, "formula": formula,
                                        "invariants": ref_inv,
                                        "steps": [t for t, _ in steps]}
            except Exception as exc:  # noqa: BLE001
                row["deterministic"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    if not args.deterministic_only:
        from ase_auto_build.ase_agent.llm_local import load_model
        model, tokenizer = load_model(
            model=args.model, revision=args.revision, cache_dir=args.cache_dir,
            adapter=None if args.base_only else args.adapter,
        )
        for row, (_, _, _, request, _) in zip(rows, selected):
            got = run_model(model, tokenizer, request, args.max_turns, args.max_new_tokens)
            ref = row["deterministic"]
            got["exact_match"] = bool(ref.get("ok") and got.get("hash") == ref.get("hash"))
            row["model"] = got
            print(f"[{row['id']:24}] det={'ok ' if ref.get('ok') else 'GAP'} "
                  f"model={got['state']:18} exact={got['exact_match']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))

    det_ok = sum(1 for r in rows if r["deterministic"].get("ok"))
    print(f"\ndeterministically buildable: {det_ok}/{len(rows)}")
    if not args.deterministic_only:
        exact = sum(1 for r in rows if r.get("model", {}).get("exact_match"))
        fin = sum(1 for r in rows if r.get("model", {}).get("state") == "FINISHED")
        print(f"model finished: {fin}/{len(rows)}   model exact: {exact}/{len(rows)}"
              f"   (of {det_ok} expressible: {exact}/{det_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
