"""CatalystSpec-aware deterministic validation rules."""

from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Any

import numpy as np
from ase.data import covalent_radii
from ase.formula import Formula

from .selectors import geometric_layers, select_atoms
from .validation import _constrained_count


def _rule(
    name: str,
    measured: Any,
    threshold: Any,
    passed: bool,
    message: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "rule": name, "measured": measured, "threshold": threshold,
        "severity": severity, "passed": bool(passed), "message": message,
    }


def _expected_atom_delta(
    spec: dict[str, Any],
    vacancy_parents: list[Any],
    cluster_atoms: int,
) -> int:
    """Atom-count change the CatalystSpec's modifications must produce.

    Vacancy sizes are re-derived by applying the spec's selector to the parent
    revision, so a modification that deleted more or fewer atoms than it selected
    is caught rather than assumed away.
    """
    delta, removals = 0, iter(vacancy_parents)
    for modification in spec["modifications"]:
        operation = modification["operation"]
        if operation == "make_vacancy":
            parent = next(removals, None)
            if parent is None:
                continue
            delta -= len(select_atoms(parent, modification["selector"]))
        elif operation == "add_adsorbate":
            species = modification.get("species") or modification["element"]
            delta += int(sum(Formula(species).count().values()))
        elif operation == "add_supported_cluster":
            delta += cluster_atoms
    return delta


def _reduced_ratio(counts: dict[str, int]) -> dict[str, int]:
    divisor = reduce(gcd, counts.values()) if counts else 1
    return {symbol: value // max(1, divisor) for symbol, value in sorted(counts.items())}


def catalyst_validation_rules(spec: dict[str, Any], workspace) -> list[dict[str, Any]]:
    atoms = workspace.final_atoms()
    model = spec["model"]
    rules: list[dict[str, Any]] = []

    expected_pbc = model["periodic_boundary_conditions"]
    actual_pbc = [bool(value) for value in atoms.pbc]
    rules.append(_rule(
        "periodic_boundary_conditions", actual_pbc, expected_pbc,
        actual_pbc == expected_pbc, "PBC matches CatalystSpec",
    ))

    allowed = set(Formula(spec["material"]["formula"]).count())
    for modification in spec["modifications"]:
        if modification["operation"] == "substitute":
            allowed.add(modification["element"])
        elif modification["operation"] == "add_adsorbate":
            allowed.update(Formula(modification.get("species") or modification["element"]).count())
        elif modification["operation"] == "add_supported_cluster":
            allowed.add(modification["element"])
    actual = set(atoms.get_chemical_symbols())
    rules.append(_rule(
        "allowed_composition", sorted(actual), sorted(allowed), actual <= allowed,
        "final elements are declared by the host and modifications",
    ))

    history = workspace.structures[workspace.result_name]
    revisions = list(history.revisions.values())
    build = next((revision for revision in revisions if revision.tool in {
        "build_bulk", "build_surface", "load_reference_bulk",
        "build_reference_surface", "build_nanoparticle",
    }), None)
    if build is not None:
        host_counts = dict(Formula(spec["material"]["formula"]).count())
        measured_counts: dict[str, int] = {}
        for symbol in build.atoms.get_chemical_symbols():
            measured_counts[symbol] = measured_counts.get(symbol, 0) + 1
        expected_ratio, measured_ratio = _reduced_ratio(host_counts), _reduced_ratio(measured_counts)
        rules.append(_rule(
            "host_stoichiometry", measured_ratio, expected_ratio,
            measured_ratio == expected_ratio,
            "built host composition matches the declared material formula",
        ))

    if model["kind"] == "surface" and build is not None:
        slab = build.atoms
        measured_layers = len(geometric_layers(slab))
        rules.append(_rule(
            "slab_layer_count", measured_layers, int(model["layers"]),
            measured_layers == int(model["layers"]),
            "slab thickness matches the requested layer count",
        ))
        z_min, z_max = float(np.min(slab.positions[:, 2])), float(np.max(slab.positions[:, 2]))
        cell_z = float(slab.cell.lengths()[2])
        lower, upper = z_min, cell_z - z_max
        expected_total = 2.0 * float(model["vacuum_angstrom"])
        measured_total = lower + upper
        rules.append(_rule(
            "slab_vacuum_angstrom", round(measured_total, 6), expected_total,
            abs(measured_total - expected_total) <= 0.05,
            "substrate vacuum matches the requested ASE per-side vacuum",
        ))
        rules.append(_rule(
            "slab_centering", round(abs(lower - upper), 6), "<= 0.05 angstrom",
            abs(lower - upper) <= 0.05, "substrate is centered along the nonperiodic axis",
        ))
    elif model["kind"] == "nanoparticle" and build is not None:
        particle = build.atoms
        lower = np.min(particle.positions, axis=0)
        upper = particle.cell.lengths() - np.max(particle.positions, axis=0)
        expected_total = 2.0 * float(model["vacuum_angstrom"])
        measured_totals = lower + upper
        rules.append(_rule(
            "nanoparticle_vacuum_angstrom",
            [round(float(value), 6) for value in measured_totals], expected_total,
            bool(np.all(np.abs(measured_totals - expected_total) <= 0.05)),
            "nanoparticle vacuum matches the requested ASE per-side vacuum",
        ))
        rules.append(_rule(
            "nanoparticle_centering", round(float(np.max(np.abs(lower - upper))), 6),
            "<= 0.05 angstrom", bool(np.all(np.abs(lower - upper) <= 0.05)),
            "nanoparticle is centered along all axes",
        ))

    fixed = int(model.get("fixed_layers_from_bottom", 0))
    if fixed:
        layers = geometric_layers(atoms)
        expected_fixed = sum(len(layer) for layer in layers[:fixed])
        actual_fixed = _constrained_count(atoms)
        rules.append(_rule(
            "fixed_layer_atom_count", actual_fixed, expected_fixed,
            actual_fixed == expected_fixed, "constraint count matches the requested bottom layers",
        ))

    adsorbate_revisions = [
        revision for revision in revisions
        if revision.tool in {"add_atomic_adsorbate", "add_molecular_adsorbate"}
    ]
    adsorbates = [item for item in spec["modifications"] if item["operation"] == "add_adsorbate"]
    for index, (modification, revision) in enumerate(zip(adsorbates, adsorbate_revisions), start=1):
        parent = history.revisions[revision.parent_revision_ids[0]].atoms
        final = revision.atoms
        first_new = len(parent)
        anchor = first_new + int(modification.get("anchor", 1)) - 1
        measured_height = float(final.positions[anchor, 2] - np.max(parent.positions[:, 2]))
        expected_height = float(modification["height_angstrom"])
        rules.append(_rule(
            f"adsorbate_{index}_anchor_height", round(measured_height, 6), expected_height,
            abs(measured_height - expected_height) <= 1e-5,
            "binding anchor height matches CatalystSpec",
        ))
        distances = final.get_all_distances(mic=True)[first_new:, :first_new]
        closest = np.unravel_index(int(np.argmin(distances)), distances.shape)
        ads_index, host_index = first_new + closest[0], closest[1]
        minimum = float(distances[closest])
        hard_limit = max(
            0.25,
            0.35 * float(covalent_radii[final.numbers[ads_index]] + covalent_radii[final.numbers[host_index]]),
        )
        rules.append(_rule(
            f"adsorbate_{index}_support_separation", round(minimum, 6), f">= {hard_limit:.6f}",
            minimum >= hard_limit, "adsorbate and support do not overlap",
        ))
        requested_site = modification["site"]
        used_site = revision.arguments.get("site", "ontop")
        rules.append(_rule(
            f"adsorbate_{index}_site_identity", used_site, requested_site,
            used_site == requested_site, "deterministic site identity matches CatalystSpec",
        ))

    cluster_revisions = [revision for revision in revisions if revision.tool == "combine"]
    clusters = [
        item for item in spec["modifications"]
        if item["operation"] == "add_supported_cluster"
    ]
    for index, (modification, revision) in enumerate(zip(clusters, cluster_revisions), start=1):
        host = history.revisions[revision.parent_revision_ids[0]].atoms
        final = revision.atoms
        first_guest = len(host)
        measured_gap = float(np.min(final.positions[first_guest:, 2]) - np.max(host.positions[:, 2]))
        expected_gap = float(modification["gap_angstrom"])
        rules.append(_rule(
            f"supported_cluster_{index}_interface_gap", round(measured_gap, 6), expected_gap,
            abs(measured_gap - expected_gap) <= 1e-5,
            "supported-cluster interface gap matches CatalystSpec",
        ))

    if build is not None:
        cluster_atoms = sum(
            len(revision.atoms) - len(history.revisions[revision.parent_revision_ids[0]].atoms)
            for revision in cluster_revisions
        )
        vacancy_parents = [
            history.revisions[revision.parent_revision_ids[0]].atoms
            for revision in revisions if revision.tool == "make_vacancy"
        ]
        expected_atoms = len(build.atoms) + _expected_atom_delta(spec, vacancy_parents, cluster_atoms)
        rules.append(_rule(
            "atom_count", len(atoms), expected_atoms, len(atoms) == expected_atoms,
            "final atom count matches the host plus the requested modifications",
        ))

    rules.append(_rule(
        "symmetry_standardization", "not requested", "not_applicable", True,
        "no target symmetry or standardization setting was supplied", "not_applicable",
    ))
    rules.append(_rule(
        "chemistry_plausibility", "not configured", "not_applicable", True,
        "oxidation-state and coordination heuristics are optional warnings", "not_applicable",
    ))
    return rules


__all__ = ["catalyst_validation_rules"]
