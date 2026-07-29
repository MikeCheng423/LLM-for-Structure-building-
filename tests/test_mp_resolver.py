from __future__ import annotations

from ase.build import bulk

from ase_auto_build.ase_agent.mp_resolver import ReferenceResolutionError, resolve_reference


def candidate(material_id, *, stable, formula="Pt", crystal_system="Cubic"):
    return {
        "material_id": material_id,
        "formula_pretty": formula,
        "elements": ["Pt"] if formula == "Pt" else ["Ce", "O"],
        "symmetry": {"crystal_system": crystal_system, "number": 225},
        "is_stable": stable,
        "run_type": "GGA",
        "task_id": f"task-{material_id}",
        "structure": bulk("Pt", "fcc", cubic=True),
    }


def test_explicit_mp_id_is_preserved_with_provenance() -> None:
    rows = [candidate("mp-2", stable=False), candidate("mp-1", stable=True)]
    result = resolve_reference("request-1", {"material_id": "mp-2"}, lambda query: rows)
    assert result.status == "resolved"
    assert result.record["material_id"] == "mp-2"
    assert result.record["selection_rule"] == "user-specified mp-id"
    assert result.record["citation_required"] is True


def test_ranked_query_prefers_the_only_stable_exact_candidate() -> None:
    rows = [
        candidate("mp-9", stable=False),
        candidate("mp-3", stable=True),
        candidate("mp-oxide", stable=True, formula="CeO2"),
    ]
    result = resolve_reference(
        "request-1", {"formula": "Pt", "elements": ["Pt"], "crystal_system": "Cubic"},
        lambda query: rows,
    )
    assert result.status == "resolved"
    assert result.record["material_id"] == "mp-3"
    assert result.candidate_material_ids == ("mp-3", "mp-9")


def test_scientifically_equivalent_candidates_require_clarification() -> None:
    rows = [candidate("mp-9", stable=True), candidate("mp-3", stable=True)]
    result = resolve_reference("request-1", {"formula": "Pt"}, lambda query: rows)
    assert result.status == "needs_clarification"
    assert result.candidate_material_ids == ("mp-3", "mp-9")


def test_resolver_rejects_credentials_and_unknown_query_fields() -> None:
    for query in ({"formula": "Pt", "api_key": "secret"}, {"formula": "Pt", "raw_filter": {}}):
        try:
            resolve_reference("request-1", query, lambda value: [])
        except ReferenceResolutionError:
            pass
        else:  # pragma: no cover
            raise AssertionError("unsafe query was accepted")
