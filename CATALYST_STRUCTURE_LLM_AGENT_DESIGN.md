# Catalyst Structure LLM — Agent Design

## 1. Purpose and design decision

This document defines the executable multi-agent design for the project in
`CATALYST_STRUCTURE_LLM.md`.

Keep the two documents separate:

- `CATALYST_STRUCTURE_LLM.md` is the scientific and product contract: scope,
  `CatalystSpec`, data policy, validation policy, and milestones.
- This document is the engineering contract: agent responsibilities,
  permissions, hand-off objects, deterministic services, failure states, and
  test gates.

The system reconstructs *auditable candidate structures*. It does not infer an
unknown experimental surface structure, predict catalytic activity, or claim
that a DFT-relaxed bulk structure is identical to an experimental catalyst.

## 2. Core principle

Use agents for interpretation and orchestration, not for truth-critical
execution:

```text
Evidence / user request
  -> evidence extraction agent
  -> specification agent
  -> deterministic policy gate
  -> deterministic Materials Project + ASE services
  -> deterministic validation
  -> optional calculation service
  -> review packet for scientist
```

Every transition produces a versioned JSON artifact. The system must retain
the original supplied evidence, but never send private papers or API keys to a
model provider without explicit authorization.

## 3. Roles and permissions

| Component | Input | May do | Must not do | Output |
| --- | --- | --- | --- | --- |
| Evidence Extractor | User-provided text/SI/table caption | Extract claims, units, source locators, contradictions | Invent citations or values | `EvidenceLedger` |
| Spec Planner | Evidence ledger + request | Propose `CatalystSpec`, assumptions, clarifying questions | Run code or mark inferred values as reported | `SpecProposal` |
| Clarification Agent | Missing-field report | Ask the smallest set of material questions | Guess topology-changing values | `ClarificationRequest` |
| Reference Resolver | Approved MP query | Request a bulk parent by `mp-id` or constrained search | Choose an arbitrary polymorph without a recorded rule | `ReferenceRecord` |
| ASE Executor | Validated spec + reference record | Call allow-listed deterministic tools | Execute model-authored Python/shell or write outside job directory | `BuildRecord` |
| Structure Verifier | Built structure + spec | Run deterministic geometry and consistency checks | Relax geometry or alter the structure | `ValidationReport` |
| Relaxation Service | Explicitly approved calculation job | Run configured MLIP/DFT workflow with resource budget | Replace source-derived structure or hide parameters | `RelaxationRecord` |
| Review Reporter | All prior records | Produce a human-readable audit packet | Convert warning into a pass | `ReviewPacket` |

Only the Evidence Extractor, Spec Planner, Clarification Agent, and Review
Reporter require LLM reasoning. All other components are ordinary,
deterministic services. An implementation may run them in one process at first;
the logical separation is retained through schemas and tests.

## 4. State machine

```text
NEW
  -> EVIDENCE_PARSED
  -> NEEDS_CLARIFICATION ----user reply----> EVIDENCE_PARSED
  -> SPEC_READY
  -> REFERENCE_RESOLVED
  -> BUILT
  -> VALIDATED
  -> [RELAXATION_PENDING -> RELAXED -> VALIDATED]
  -> REVIEW_READY

Any state -> REFUSED | UNSUPPORTED | FAILED
```

`SPEC_READY` means schema-valid and sufficiently specified for the selected
tool family. It does not mean physically validated. `VALIDATED` means the
candidate passes deterministic checks. `REVIEW_READY` is the only state that
may be delivered as a completed reconstruction.

## 5. Hand-off contracts

All records carry `schema_version`, `request_id`, UTC timestamp, producer
version, and SHA-256 hashes of referenced artifacts.

### 5.1 EvidenceLedger

```json
{
  "schema_version": "evidence-ledger/v1",
  "request_id": "uuid",
  "claims": [
    {
      "field": "model.layers",
      "value": 4,
      "unit": null,
      "evidence_type": "reported",
      "source_id": "user-evidence-1",
      "locator": "Methods, p. 4",
      "verbatim_span": "A four-layer Pt(111) slab was used.",
      "confidence": "high"
    }
  ],
  "contradictions": [],
  "unresolved_fields": ["model.vacuum_angstrom"]
}
```

`verbatim_span` is retained only where permitted by the supplied source and
local data policy. It is not training text by default.

### 5.2 SpecProposal

```json
{
  "schema_version": "spec-proposal/v1",
  "request_id": "uuid",
  "task_status": "ready",
  "catalyst_spec": { "schema_version": "0.1" },
  "field_sources": [
    {"field": "model.layers", "claim_index": 0},
    {"field": "model.vacuum_angstrom", "evidence_type": "assumed_default", "reason": "not reported"}
  ],
  "clarification_questions": [],
  "agent_warnings": []
}
```

The Policy Gate rejects any `reported` field without an EvidenceLedger link.

### 5.3 ReferenceRecord

```json
{
  "schema_version": "reference-record/v1",
  "source": "materials_project",
  "material_id": "mp-XXXX",
  "retrieved_at": "2026-07-28T00:00:00Z",
  "structure_role": "bulk_parent",
  "formula": "Pt",
  "structure_sha256": "...",
  "calculation_method": "GGA | GGA_U | r2SCAN | unknown",
  "task_id": "...",
  "selection_rule": "user-specified mp-id | deterministic ranked query",
  "citation_required": true
}
```

Materials Project data is a computational bulk reference. A slab, termination,
defect, adsorbate, or support construction derived from it has separate
provenance and is never labelled as directly reported by MP.

### 5.4 Build and validation records

`BuildRecord` stores the approved tool calls, registry fingerprint, input
hashes, ASE version, output hashes, and output paths. `ValidationReport` stores
each rule, measured value, threshold, severity, and pass/fail result. Both are
written before report generation.

## 6. Deterministic services

### 6.1 Policy Gate

Before any retrieval or ASE action, require:

1. Valid `CatalystSpec` JSON Schema.
2. Field-level provenance for every structure-determining field.
3. No unresolved chemical identity, phase, facet, defect site, adsorbate
   identity, anchor, or required adsorption site.
4. Explicit acknowledgement of defaults for vacuum, centering, atom ordering,
   and output format.
5. One supported tool-family route.

Otherwise return `needs_clarification` or `unsupported`; do not build.

### 6.2 Materials Project resolver

Use MP only after the policy gate. Prefer a user/paper-specified `mp-id`. If a
search is necessary, use a deterministic ranking and retain all candidates:

1. exact reduced formula;
2. required elements and crystal system;
3. requested space group where present;
4. stable entry preference, then fixed ordering by material ID.

If more than one candidate remains scientifically meaningful, return a
clarification or a named candidate set. Store `material_id`, retrieved
structure, task ID, functional/run type, query arguments, retrieval time, and
response hash. API keys stay in environment variables or a secret manager and
never appear in prompts, logs, datasets, or reports.

### 6.3 ASE builder registry

First release exposes only these typed tools:

- `load_reference_bulk(reference_id)`
- `standardize_cell(setting)`
- `build_surface(miller, layers, supercell, vacuum, termination)`
- `make_vacancy(selector)`
- `substitute(selector, element)`
- `add_adsorbate(species, site, site_index, height, anchor, orientation)`
- `freeze_layers(side, count, axes)`
- `validate_structure(profile)`
- `export_structure(formats)`

The dispatcher maps `CatalystSpec` to these calls. The LLM cannot send raw
coordinates, source code, file paths, calculator commands, or unregistered
tool names.

### 6.4 Validation profiles

Every final structure runs all applicable checks:

- schema, finite coordinates, valid elements, cell/PBC, and file round-trip;
- element-pair minimum distance table, not one universal threshold;
- stoichiometry, atom count, constraint count, slab thickness, vacuum;
- periodic slab centering and termination metadata;
- adsorbate-support separation, binding anchor, coverage, and site identity;
- symmetry/standardization checks where applicable using `spglib`;
- optional chemistry warnings: charge balance, oxidation-state plausibility,
  coordination outliers.

Validation distinguishes `error`, `warning`, and `not_applicable`. Warnings are
not converted to a pass by an agent.

### 6.5 Relaxation service

Relaxation is opt-in and requires calculator, model/version, parameters,
convergence thresholds, resource limit, and explicit user confirmation when
the job is costly. Preserve:

- source-derived `pre_relax` structure;
- calculator input and job log;
- `post_relax` structure;
- energy, forces, convergence state, and failure reason.

Never overwrite the paper-derived candidate with the relaxed structure.

## 7. Golden-case programme

### 7.1 What is golden

A golden case is an immutable, reviewed test fixture with a known expected
result for a declared scope. It is not necessarily an experimentally unique
structure.

Use three labels:

| Label | Meaning |
| --- | --- |
| `database_bulk_golden` | MP structure and deterministic standardization are expected to match recorded output. |
| `construction_golden` | A reviewed rule turns a parent bulk plus explicit settings into a slab/defect/adsorbate candidate. |
| `physical_reference_golden` | A structure is checked against licensed literature data and/or a declared converged calculation. |

Materials Project is appropriate for the first label and as the parent for the
second. It is not sufficient alone for adsorption geometries, surface
terminations, reconstructed surfaces, supported clusters, or experimental
claims.

### 7.2 Initial corpus

Create 100 reviewed fixtures before model training:

- 30 MP bulk/reference and standardization cases;
- 30 elemental/binary slabs with stated facet, termination, layers, and vacuum;
- 15 substitutions/vacancies;
- 15 atomic/molecular adsorbates with varied height and coverage;
- 10 refusal/clarification/contradiction cases.

Vary all numerically important fields independently. In particular, do not use
one constant adsorption height, slab thickness, vacuum, or supercell per tool.

## 8. Training and evaluation workflow

Train only the Spec Planner after the deterministic vertical slice passes.

1. Generate/curate evidence-to-spec records from reviewed fixtures.
2. Validate every target spec through the same policy gate and ASE registry.
3. Group splits by paper/source and final-structure family; also keep wording
   templates disjoint in a linguistic OOD set.
4. Train QLoRA with assistant-only loss and schema-constrained inference.
5. Evaluate end-to-end using the deployment registry, not an oracle-reduced
   tool list.
6. Promote only if deterministic validity, OOD performance, provenance
   accuracy, clarification quality, and zero forbidden execution all pass.

Required test sets:

- `iid_construction`: held-out parameter combinations in known families;
- `linguistic_ood`: unseen human-written Chinese/English wording;
- `compositional_ood`: new element × facet × adsorbate combinations;
- `journal_holdout`: licensed, reviewed evidence from unseen papers;
- `negative`: unsupported, contradictory, injection-like, and missing-field
  requests.

## 9. Acceptance criteria

### Vertical slice gate

Before an LLM is connected:

- 100% schema validation for all golden specifications;
- 100% deterministic build and export round-trip for supported golden cases;
- zero writes outside per-request output folders;
- zero arbitrary-code execution paths;
- every golden field has provenance;
- every intentional ambiguity returns clarification rather than a build.

### Model promotion gate

Before an adapter is made available by default:

- >= 98% schema-valid `SpecProposal` rate;
- >= 95% end-to-end executable rate on supported held-out cases;
- 100% zero forbidden execution on negative/adversarial cases;
- no regression from the frozen base in matched evaluation;
- field-level provenance precision and recall are reviewed and reported;
- all physical-reference goldens and all OOD failures receive domain review.

The numerical thresholds are release criteria, not proof of experimental
correctness.

## 10. Implementation order

1. Create Pydantic models/JSON schemas for all hand-off records.
2. Implement policy gate and deterministic ASE registry.
3. Add MP resolver with a mocked API test suite; do not embed credentials.
4. Implement validators and 100 golden fixtures.
5. Add exporter and review packet generator.
6. Add an LLM-backed Evidence Extractor and Spec Planner using constrained JSON.
7. Create training corpus, OOD sets, and promotion scripts.
8. Add optional MLIP/DFT relaxation as a separately approved backend.

### 10.1 Implemented boundary

The implementation covers steps 1 through 5 and the bounded role wrappers from
step 6 in `src/ase_auto_build/ase_agent/`. The runtime packages all schemas from
`schemas/`, validates every hand-off, and exposes the journal workflow through
`ASE_catalyst_build`. The MP resolver accepts an injected transport so tests and
offline runs need no credential. The pipeline accepts only a resolved
`ReferenceResolution`; it records the selected MP material and structure hash
before ASE construction.

Version 1 uses two model calls: evidence extraction and specification planning.
The planner supplies clarification questions, and deterministic code renders the
review packet. This removes separate clarification and review model calls while
preserving their hand-off records and state transitions.

The existing promoted r5 adapter targets direct ASE tool calls. It has not passed
the Evidence Extractor/Spec Planner gate. Three live prompt-only smokes failed
closed before deterministic construction, so step 7 remains open for the journal
roles: build a dedicated corpus, train an adapter, and run the held-out promotion
suite.

The current relaxation support stops at the record schema and explicit policy
boundary. No calculator backend runs from the journal command. Coverage,
orientation, non-default termination, and lattice-matching requests return an
unsupported or clarification packet until the deterministic registry supports
them.

## 11. Repository layout

```text
schemas/
  catalyst_spec.schema.json
  evidence_ledger.schema.json
  spec_proposal.schema.json
  reference_record.schema.json
src/catalyst_llm/
  agents/            # LLM wrappers; no filesystem or calculator authority
  policy.py
  mp_resolver.py
  dispatcher.py
  validation.py
  reporting.py
  tools/
tests/
  golden/
  fixtures/
  test_policy.py
  test_mp_resolver.py
  test_builders.py
  test_validation.py
  test_agent_contracts.py
```

This design permits future deployment as separate processes or services without
changing the scientific contract. Start as one local process with explicit
objects and tests; distribute only after the local implementation is reliable.
