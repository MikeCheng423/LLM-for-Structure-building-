"""Scope items named directly in CATALYST_STRUCTURE_LLM.md.

Section 2 (compound and 2D supports), section 4 (paper/DOI input contract),
section 7 and milestone 4 (named candidates instead of a silent choice),
section 11 (structure summary and source-backed facts), and section 14's six
suggested golden examples.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from ase.build import molecule

from ase_auto_build.ase_agent.catalyst_cli import load_request
from ase_auto_build.ase_agent.catalyst_dispatch import dispatch_spec
from ase_auto_build.ase_agent.catalyst_validation import catalyst_validation_rules
from ase_auto_build.ase_agent.catalyst_pipeline import run_candidate_set, run_catalyst_pipeline
from ase_auto_build.ase_agent.tools_surface import _anchor_down

from training.evaluations.run_vertical_slice_gate import load_fixtures

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "tests" / "golden"

SECTION_14 = (
    "slice-co-on-pt111",
    "slice-o-on-pd100",
    "slice-pt-substituted-au111",
    "slice-n-doped-graphene-single-atom",
    "slice-o-vacancy-tio2-110",
    "slice-cu-cluster-on-ceo2-111",
)


def _fixture(case_id: str) -> dict:
    return json.loads((GOLDEN_ROOT / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Section 14: the six suggested golden examples
# --------------------------------------------------------------------------


def test_all_six_section_14_examples_are_fixtures():
    present = {f["case_id"] for f in load_fixtures(GOLDEN_ROOT) if f.get("section") == "14"}
    assert present == set(SECTION_14)


@pytest.mark.parametrize("case_id", SECTION_14)
def test_section_14_example_builds_and_validates(case_id):
    fixture = _fixture(case_id)
    spec = fixture["spec_proposal"]["catalyst_spec"]
    dispatched = dispatch_spec(spec, request_id=fixture["request_id"])
    assert len(dispatched.atoms) > 0
    failed = [
        rule for rule in catalyst_validation_rules(spec, dispatched.workspace)
        if not rule["passed"]
    ]
    assert not failed, [rule["rule"] for rule in failed]


def test_registered_prototypes_reach_the_surface_builder():
    """Section 2 wants oxide supports and 2D sheets, not only ase.build.bulk phases."""
    formulas = {}
    for case_id in ("slice-o-vacancy-tio2-110", "slice-cu-cluster-on-ceo2-111",
                    "slice-n-doped-graphene-single-atom"):
        fixture = _fixture(case_id)
        dispatched = dispatch_spec(
            fixture["spec_proposal"]["catalyst_spec"], request_id=fixture["request_id"]
        )
        formulas[case_id] = dispatched.atoms.get_chemical_formula()
    assert "Ti" in formulas["slice-o-vacancy-tio2-110"]
    assert "Ce" in formulas["slice-cu-cluster-on-ceo2-111"]
    assert "N" in formulas["slice-n-doped-graphene-single-atom"]


# --------------------------------------------------------------------------
# Adsorbate orientation: the anchor is the binding atom
# --------------------------------------------------------------------------


def test_ase_molecule_geometry_would_bury_the_anchor_without_orientation():
    """The defect this guards: molecule('CO') is (O, C) with the carbon lower."""
    raw = molecule("CO")
    assert raw.get_chemical_symbols() == ["O", "C"]
    assert raw.positions[1, 2] < raw.positions[0, 2]  # carbon below the anchor


@pytest.mark.parametrize("species,anchor", [
    ("CO", 1), ("CO", 2), ("H2O", 1), ("NH3", 1), ("OH", 1), ("O2", 1), ("NO", 1),
])
def test_anchor_down_makes_the_anchor_the_lowest_atom(species, anchor):
    oriented = _anchor_down(molecule(species), anchor - 1)
    z = oriented.positions[:, 2]
    assert float(z[anchor - 1]) == pytest.approx(float(z.min()), abs=1e-9)
    # Bond lengths survive a mirror; only the orientation changes.
    original = molecule(species)
    assert sorted(original.get_all_distances().ravel()) == pytest.approx(
        sorted(oriented.get_all_distances().ravel()), abs=1e-9
    )


def test_anchor_down_rejects_an_out_of_range_anchor():
    with pytest.raises(IndexError):
        _anchor_down(molecule("CO"), 5)


def test_placed_molecule_keeps_every_atom_above_the_surface():
    fixture = _fixture("slice-co-on-pt111")
    spec = fixture["spec_proposal"]["catalyst_spec"]
    dispatched = dispatch_spec(spec, request_id=fixture["request_id"])
    atoms = dispatched.atoms
    slab = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Pt"]
    adsorbate = [i for i in range(len(atoms)) if i not in slab]
    top = max(atoms.positions[i, 2] for i in slab)
    assert min(atoms.positions[i, 2] for i in adsorbate) >= top + 1.8
    rule = next(
        item for item in catalyst_validation_rules(spec, dispatched.workspace)
        if item["rule"] == "adsorbate_1_anchor_is_lowest"
    )
    assert rule["passed"]


# --------------------------------------------------------------------------
# Section 4 and 11: paper metadata, DOI per fact, structure summary
# --------------------------------------------------------------------------


def test_request_contract_accepts_paper_metadata(tmp_path):
    path = tmp_path / "request.json"
    value = {
        "request_id": "paper-1", "request": "Build Pt(111)",
        "sources": [{"source_id": "paper", "locator": "Methods, p. 4", "text": "Pt(111)"}],
        "paper": {"title": "A study", "doi": "10.1000/x", "year": 2025},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_request(path) == value


@pytest.mark.parametrize("paper", [
    {"doi": "10.1000/x", "shell": "rm -rf /"},
    {"doi": ""},
    {"year": "2025"},
])
def test_request_contract_still_rejects_a_malformed_paper_block(tmp_path, paper):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({
        "request_id": "paper-1", "request": "Build Pt(111)",
        "sources": [{"source_id": "paper", "locator": "p. 4", "text": "Pt(111)"}],
        "paper": paper,
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_request(path)


def test_review_packet_carries_doi_and_a_structure_summary(tmp_path):
    fixture = _fixture("slice-co-on-pt111")
    paper = {"title": "CO on Pt(111)", "doi": "10.1000/example.2025.1", "year": 2025}
    result = run_catalyst_pipeline(
        fixture["evidence_ledger"], fixture["spec_proposal"], tmp_path, paper=paper,
    )
    assert result.status == "review_ready"
    packet = json.loads((result.request_dir / "review_packet.json").read_text(encoding="utf-8"))

    assert packet["source_backed_facts"]
    assert all(fact["doi"] == paper["doi"] for fact in packet["source_backed_facts"])
    assert all(fact["locator"] for fact in packet["source_backed_facts"])
    assert packet["reproduction"]["paper"] == paper

    summary = packet["structure_summary"]["model"]
    assert summary["kind"] == "surface"
    assert summary["miller_indices"] == [1, 1, 1]
    assert summary["layers"] == 4
    assert summary["adsorbates"] == [{
        "species": "CO", "site": "ontop", "site_index": 1,
        "height_angstrom": 1.85, "anchor": 2,
    }]


def test_structure_summary_without_a_paper_omits_doi(tmp_path):
    fixture = _fixture("slice-o-on-pd100")
    result = run_catalyst_pipeline(fixture["evidence_ledger"], fixture["spec_proposal"], tmp_path)
    packet = json.loads((result.request_dir / "review_packet.json").read_text(encoding="utf-8"))
    assert all("doi" not in fact for fact in packet["source_backed_facts"])
    assert "paper" not in packet["reproduction"]


# --------------------------------------------------------------------------
# Section 7 and milestone 4: named candidates, never a silent choice
# --------------------------------------------------------------------------


def _ambiguous_fetch(fixture):
    from ase.build import bulk as ase_bulk

    def fetch(query):
        candidates = []
        for candidate in fixture["mp_candidates"]:
            item = dict(candidate)
            item["structure"] = ase_bulk(**candidate["structure"]["bulk"])
            candidates.append(item)
        return candidates

    return fetch


def test_an_ambiguous_parent_produces_one_package_per_candidate(tmp_path):
    fixture = _fixture("mp-bulk-ambiguous")
    outcome = run_candidate_set(
        fixture["evidence_ledger"], fixture["spec_proposal"], tmp_path,
        query=fixture["mp_query"], fetch=_ambiguous_fetch(fixture),
    )
    assert len(outcome.record["candidates"]) == 2
    assert [item["status"] for item in outcome.record["candidates"]] == ["review_ready"] * 2
    # Separately named, and genuinely different structures.
    hashes = {item["atoms_hash"] for item in outcome.record["candidates"]}
    assert len(hashes) == 2
    assert (tmp_path / f"{fixture['request_id']}-candidate_set.json").is_file()


def test_each_candidate_labels_its_own_selection_as_derived(tmp_path):
    fixture = _fixture("mp-bulk-ambiguous")
    outcome = run_candidate_set(
        fixture["evidence_ledger"], fixture["spec_proposal"], tmp_path,
        query=fixture["mp_query"], fetch=_ambiguous_fetch(fixture),
    )
    for result in outcome.results:
        packet = json.loads((result.request_dir / "review_packet.json").read_text(encoding="utf-8"))
        selection = [
            item for item in packet["assumptions"] if item["field"] == "material.reference_id"
        ]
        assert len(selection) == 1
        assert selection[0]["evidence_type"] == "derived"
        assert "candidate" in selection[0]["reason"]
        # A system choice must never be presented as reported.
        assert all(
            fact["field"] != "material.reference_id" for fact in packet["source_backed_facts"]
        )


def test_candidate_sets_refuse_an_unambiguous_parent(tmp_path):
    fixture = _fixture("mp-bulk-00")
    with pytest.raises(ValueError, match="unresolved"):
        run_candidate_set(
            fixture["evidence_ledger"], fixture["spec_proposal"], tmp_path,
            query=fixture["mp_query"], fetch=_ambiguous_fetch(fixture),
        )
