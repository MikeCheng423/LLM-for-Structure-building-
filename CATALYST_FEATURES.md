# Catalyst structure builder — feature description

What the journal-to-structure system can do today, what it refuses, and how each
claim is checked. This is the capability catalogue; the scientific contract is
[`CATALYST_STRUCTURE_LLM.md`](CATALYST_STRUCTURE_LLM.md) and the engineering
contract is
[`CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md`](CATALYST_STRUCTURE_LLM_AGENT_DESIGN.md).

Status markers used below:

| marker | meaning |
| --- | --- |
| **Ready** | Implemented, covered by a golden fixture, and measured by the section 9 gate. |
| **Python API** | Implemented and tested, but not reachable from the command line. |
| **Fail-closed** | Deliberately refused with a clarification or `unsupported` packet. |
| **Blocked** | Not implemented; the reason is stated. |

---

## 1. What it is

A request describing a catalyst — a paper excerpt, an SI passage, a table
caption, or a hand-written summary — becomes a validated `ase.Atoms` structure
plus an audit package. Two bounded model calls interpret the text; **all**
retrieval, construction, validation and export is deterministic Python.

```text
evidence  →  Evidence Extractor  →  Spec Planner  →  policy gate
                                                        │
          review packet  ←  validation  ←  ASE registry ┘
```

The model emits typed JSON against a fixed schema. It never runs code, never
picks a file path, never names an unregistered tool, and never writes outside its
own request directory. Those four properties are measured, not asserted — see
§7.

---

## 2. Entry points

```bash
# Journal workflow: evidence in, immutable request package out.
ASE_catalyst_build --input request.json --out catalyst_requests

# Direct tool-calling agent (separate, older track).
ASE_auto_build "Build a 2x2 Cu(100) slab, 4 layers, 12 A vacuum."
```

`request.json` carries only `request_id`, `request`, `sources`, and an optional
`paper` block. Any other key is rejected before a model is loaded.

```json
{
  "request_id": "smith-2025-co-pt111",
  "request": "Build the CO/Pt(111) catalyst described in the paper.",
  "paper": {"title": "CO oxidation on Pt(111)", "doi": "10.1000/x", "year": 2025},
  "sources": [
    {"source_id": "methods-1", "locator": "Methods, p. 4",
     "text": "A four-layer 2 x 2 Pt(111) slab with 15 A of vacuum was used."}
  ]
}
```

Exit codes: `0` review ready, `1` failed, `3` unsupported, `5` needs
clarification, `6` environment error.

---

## 3. Structures it can build

### 3.1 Hosts — **Ready**

| kind | support |
| --- | --- |
| Bulk | Elemental `fcc`, `bcc`, `hcp`, `diamond`, `sc`, with optional supercell. |
| Bulk from Materials Project | Any resolved `mp-id` parent, hash-verified before use. |
| Surface | Elemental phases at any Miller index. |
| Surface, compound | `fluorite`, `rocksalt`, `wurtzite`, `zincblende` families combined with a composition — `fluorite` + `CeO2`. |
| Surface, fixed prototype | `rutile`/`anatase` TiO2, `graphene`, `graphite`, `hBN`. |
| Nanoparticle | Elemental icosahedron, octahedron, cuboctahedron by shell count. |

Layers, vacuum, supercell, centering and bottom-layer constraints are all
explicit spec fields, and each needs its own provenance entry.

### 3.2 Modifications — **Ready**

- **Vacancies and substitutions** through a typed selector: atom indices, element
  symbols, a fractional or Cartesian region, a layer from either side, or an
  ordinal within any of those. No "nearest atom".
- **Adsorbates** — single atoms, or any `ase.build.molecule` species — at a named
  `ontop`, `bridge` or `hollow` site, with an explicit height and, for
  multi-element species, an explicit binding anchor.
- **One supported cluster** on a surface host, with a declared interface gap.

**Adsorbate orientation.** `anchor` is 1-based over `ase.build.molecule`'s atom
ordering, which is not the formula's: `molecule("CO")` is stored as `(O, C)`, so
the carbon is atom 2. The builder mirrors the molecule about the anchor plane
whenever another atom would sit below it, so the declared binding anchor is
always the atom nearest the surface. Bond lengths are unchanged. This is a
placement rule; a user-supplied `orientation` remains fail-closed.

### 3.3 Deliberately refused — **Fail-closed**

Each returns a clarification or `unsupported` packet and builds nothing:

- non-default compound terminations,
- adsorbate coverage and repeated co-adsorption,
- user-supplied molecular orientation,
- lattice matching between a cluster and its support,
- more than one supported cluster per candidate,
- any unresolved element, phase, facet, defect site, adsorbate identity or
  binding anchor,
- contradictory evidence — never averaged, never silently resolved.

Ten golden fixtures cover exactly these cases, so the refusals are tested rather
than hoped for.

---

## 4. Provenance and ambiguity

Every structure-determining field carries a provenance label. The policy gate
refuses to build otherwise.

| label | meaning |
| --- | --- |
| `reported` | Stated in a supplied source; must link an EvidenceLedger claim whose field, type **and value** match. |
| `user_supplied` | Added by the requester; same linkage requirement. |
| `derived` | Deterministically computed; must carry a reason. |
| `assumed_default` | Chosen by the system and disclosed; must carry a reason. |
| `ambiguous` | Blocks the build. |

Model knowledge can never be labelled `reported` — the link is checked against
the ledger, not trusted.

**Named candidates — Python API.** When the reference resolver cannot reduce a
bulk parent to one entry, `run_candidate_set` builds *one separately named
package per candidate* and writes a `candidate_set.json` index, rather than
silently choosing a polymorph. Each package labels its own `material.reference_id`
as `derived`, with the reason naming its position in the set.

---

## 5. Outputs

Every request writes one immutable directory. Re-running a request never
overwrites a package.

| file | contents |
| --- | --- |
| `POSCAR` | Always written. |
| `structure.cif` / `.xyz` / `.traj` | Written on request. |
| `structure.json` | Recipe, recipe hash, atoms hash, invariants, tool sequence, model identity. |
| `supplied_evidence.json` | The request exactly as supplied, including the `paper` block. |
| `evidence_ledger.json` | Extracted claims with source ids and locators. |
| `spec_proposal.json` | The proposed `CatalystSpec` and its field sources. |
| `reference_record.json` | Materials Project selection, query, response hash, run type. |
| `build_record.json` | Approved tool calls, registry fingerprint, ASE version, input and output hashes. |
| `validation_report.json` | Every rule with its measured value, threshold, severity and result. |
| `review_packet.json` + `review.md` | The human-facing audit packet. |
| `clarification_request.json` | Written instead of a build when a material value is missing. |
| `failure_record.json` | Written instead of a packet when a stage raises. |

The review packet reports, in the order §11 of the product document asks for:
status; a structure summary naming the facet, layers, vacuum, termination,
adsorbates, defects and supported clusters as well as composition, atom count,
cell and PBC; source-backed facts with **value + DOI + locator**; assumptions
with reasons; warnings; files; and a reproduction record carrying the schema
version, registry fingerprint, recipe and atoms hashes, ASE version, model
revision, adapter SHA-256 and decoding parameters.

---

## 6. Validation

Every candidate runs every applicable rule. A warning is never converted to a
pass, and a rule that does not apply is recorded as `not_applicable` rather than
silently skipped.

**Geometry** — empty structure, atom budget, valid element symbols, finite
positions and cell, non-singular cell, axis lengths, per-pair atomic overlap,
constraint index range, low vacuum.

The overlap check compares **every** pair against its own element-pair threshold.
Comparing only the globally closest pair — as an earlier version did — hides a
violation whenever a larger pair happens to sit nearer.

**Spec conformance** — PBC against the spec; composition within the declared
elements; host stoichiometry against the declared formula; final atom count
against the host plus each requested modification; slab layer count (exact for
elemental hosts, a whole-number multiple for compounds, whose layers resolve into
sublayers); vacuum and centering; constrained atom count against the requested
bottom layers.

**Adsorbates** — anchor height against the spec; the anchor is the lowest
adsorbate atom; adsorbate-support separation against the element-pair threshold;
site identity against the recorded call; supported-cluster interface gap.

**Export** — each written file is read back and compared on symbols, positions,
cell and constraints, and every requested format must have produced a file.

**Not yet covered** — `spglib` symmetry and standardization, and chemistry
plausibility (charge balance, oxidation state, coordination outliers). Both are
recorded as `not_applicable` rather than passing quietly. **Blocked**: neither
`spglib` nor a chemistry heuristic is a dependency.

---

## 7. Gates and test sets

### Vertical-slice gate — **Ready**

```bash
python training/evaluations/run_vertical_slice_gate.py \
    --report training/evaluations/vertical_slice_gate.json
```

106 immutable fixtures in `tests/golden/`, SHA-256 pinned and byte-identical on
regeneration: the 100 that design §7.2 specifies (30 Materials Project bulk, 30
slabs, 15 defects, 15 adsorbates, 10 refusals, every numeric axis varied on its
own stride) plus the six examples §14 names as the first implementation target —
CO/Pt(111), O/Pd(100), Pt-substituted Au(111), N-doped graphene with a supported
metal atom, an oxygen vacancy on rutile TiO2(110), and a Cu cluster on CeO2(111).

The six §9 criteria, all currently at threshold:

| criterion | value |
| --- | --- |
| schema conformance | 106 / 106 |
| build and export round trip | 95 / 95 buildable |
| writes outside a request directory | 0 |
| unregistered tool calls | 0 |
| fields without provenance | 0 |
| ambiguities returned as clarification | 11 / 11 |

`tests/test_golden_corpus.py` keeps this a standing barrier, with negative
controls proving each rule can fail.

Two limits are recorded in the manifest rather than glossed: the fixtures have
**not** had the §7.1 domain-expert review (every case says
`domain_review: pending`), and `physical_reference_golden` is empty because §7.1
requires licensed literature data or a declared converged calculation.

### Held-out evaluation sets

```bash
python training/generators/build_journal_holdouts.py
```

| set | variable held out | status |
| --- | --- | --- |
| `template_holdout` | Wording only — five phrase templates and both request sentences disjoint from training; chemistry familiar. | **Ready**, 200 records |
| `family_holdout` | Chemistry only — elements Rh, Ir, Pb, Mo, Ta; adsorbates N, C, NO, O2, OH; supports CeO2 and CaO, none of which appear in training. Wording in-distribution. | **Ready**, 200 records |
| `journal_holdout` | Complete unseen papers. | **Blocked** — needs licensed journal text this repository does not contain. |

Generation fails closed if a "held-out" source text also appears in the training
corpus, or if held-out chemistry turns out to be present in it.

Note on interpretation: the planner is prompted with the EvidenceLedger JSON, not
the prose, so `template_holdout` mainly stresses the **extractor** role.

---

## 8. Model tracks

| track | state |
| --- | --- |
| Direct ASE tool calling (r5) | Promoted, `production_ready: true`. 99.2% exact on novel phrasing. |
| Journal roles (r1) | **Not promoted.** `journal_role_ready` is `false`. |

The journal adapter scored 1.000 schema-valid on the teacher-forced offline
harness and failed its first free-running request two minutes after being
flagged ready. The promotion path now reads that live smoke (`--smoke-dir`) and
measures `forbidden_action_rate` instead of hard-coding it, so an offline report
alone cannot promote an adapter.

**Regenerate `journal_roles_v1` before retraining.** Its 100 CO adsorption cases
were generated with the pre-fix anchor and encode a 0.7497 Å C–Pt contact as a
successful build.

`promoted` never implies `production_ready`; a human review flips that flag with
a provenance block, and nothing automatic may do it.

---

## 9. Safety boundary

- The registry is the entire action surface. No arbitrary Python, shell,
  filesystem, network or calculator-constructor execution is reachable from a
  model.
- Evidence text is data. Instructions embedded in a source — a golden fixture
  carries a `subprocess.run(['rm','-rf','/'])` injection — never become actions;
  that case returns a clarification and builds nothing.
- Materials Project queries are restricted to five fields, reject any key or
  token, and go through an injected transport, so tests and offline runs need no
  credential.
- Writes are confined to one directory per request; the gate measures this by
  snapshotting the tree around every build and running each from a fresh working
  directory.
- Relaxation has a record schema and a policy boundary but **no backend**
  (**Blocked**). A paper-derived candidate can therefore never be overwritten by
  a relaxed structure.

---

## 10. Verification

```bash
export PYTHONPATH="$PWD/src:$PWD"

python -m pytest -q                                          # full suite
python training/evaluations/run_vertical_slice_gate.py        # section 9 gate
python training/generators/build_golden_fixtures.py           # must be byte-identical
python training/evaluations/evaluate_journal_corpus.py training/datasets/journal_holdout_family
```

The gate runner exits non-zero if any criterion drops below threshold or a
fixture hash no longer matches the manifest.
