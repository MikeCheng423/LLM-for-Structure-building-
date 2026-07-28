# Catalyst structure-building capability of the 4B model

Date: 2026-07-27

## Decision

Keep the promoted r5 Qwen3-4B adapter for the current bounded feature set. It is
good enough to translate explicit requests for elemental bulk cells, low-index
elemental slabs, simple defects, and single adsorbates into registered ASE tool
calls.

Do not claim support for catalyst structures representative of the journal
literature. The current corpus and tool registry do not cover enough catalyst
chemistry to test that claim. Increasing the model size now would leave the
main gaps unchanged because several required structures cannot be expressed by
the available tools or reached through the router.

## Evidence reviewed

The independent evaluation agent inspected the versioned datasets, model
manifests, generated calls, promotion reports, and current runtime. It also ran
a CPU-only journal-scope routing and deterministic-runtime probe. CUDA was not
available in the agent sandbox, so the agent used the completed full-registry
model evaluations instead of repeating inference.

The best adapter remains r5, not the newer r7 adapter:

| Evaluation | Exact structure | Invariants | Finish | Schema execution | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| r5, 120 held-out records, full registry | 95.0% | 95.0% | 99.2% | 96.4% | promoted |
| r5, 120 novel-phrasing records from the valid r6 set | 99.2% | 99.2% | 100% | 97.1% | passes the phrasing probe |
| frozen 4B base, 120 r7 novel records | 12.5% | 12.5% | 32.5% | 51.1% | inadequate without fine-tuning |
| r7 adapter, 120 seen-phrasing records | 78.3% | 85.8% | 94.2% | 87.6% | rejected |

Sources:

- `adapter-r5fix120-full-pf2-s120.json`
- `adapter-r5-novel-pf6-s120.r6set.json`
- `base-r7-novel-pf6-s120.json`
- `adapter-r7-seen-pf6-s120.json`
- `promotion-r7-seen-pf6-s120.json`

The r5 full-registry evaluation scored 26/26 atomic-adsorption cases, 8/8
molecular-adsorption cases, 24/26 elemental surfaces, and 23/23 constrained
surfaces. It also completed the small prototype, substitution, and vacancy
samples. The r7 adapter regressed on surfaces: it produced `a: 0.0` and then
retried invalid small lattice constants. Its surface exact rate fell to 52.4%.

The non-suffixed r5 r7-novel report and parts of the r7 novel reports cross a
molecular-adsorbate geometry change. Their exact hashes are stale. Use the
`.r6set` report for the valid r5 novel-phrasing result, and regenerate the
corpus before another promotion run.

## Coverage audit

The r7 corpus contains 4,329 records derived from 438 recipe groups. More
paraphrases increased the record count without increasing the number of
underlying structures.

Current catalyst training covers:

- Pt, Cu, Ni, and Fe elemental slabs;
- (111), (100), and (110) facets with three to five layers;
- 1x1 and 2x2 surface cells with 12 angstrom vacuum;
- atomic H or O and molecular CO;
- ontop adsorption at fixed 1.8 or 1.9 angstrom heights;
- first-site bulk vacancies and Au substitutions;
- graphene, graphite, hBN, rutile TiO2, and anatase TiO2 prototypes.

The corpus has no trajectories for stepped or other high-index facets,
bridge/hollow sites, variable coverage, coadsorption, ordered or disordered
alloys, surface defects, supported clusters, single-atom catalysts, or
metal/support interfaces. The r7 training split also overweights atomic
adsorption: 1,230 records versus 30 prototype records and 130 records for each
of vacancy and substitution.

This scope is much narrower than open journal datasets. Catalysis-Hub reports
more than 100,000 chemisorption and reaction energies and describes a corpus
with many alloys and oxides. OC20 spans 1,281,040 DFT relaxations across broad
materials, surfaces, and adsorbates. OC22 adds 62,331 oxide relaxations across
oxide materials, coverages, and adsorbates.

- Catalysis-Hub paper: https://doi.org/10.1038/s41597-019-0081-y
- OC20 paper: https://doi.org/10.1021/acscatal.0c04525
- OC22 paper: https://doi.org/10.1021/acscatal.2c05426

## Journal-scope probe

The CPU probe separated runtime capability from model capability:

| Request family | Probe result | Interpretation |
| --- | --- | --- |
| Pt(211) elemental slab | routed and built, Pt20 | runtime support exists; model remains untested on this facet |
| NiO(100) slab | direct deterministic recipe built Ni20O20 | current code supports it, but r5 has no training or evaluation evidence |
| CeO2 slab with O vacancy | direct deterministic recipe built Ce32O63 | current code supports it, but r5 has no training or evaluation evidence |
| Pt3Ni alloy slab | router exposed elemental `build_surface` only | registry cannot represent the requested alloy slab |
| Au-doped Ni(111) | router chose bulk substitution tools | surface substitution path is misrouted |
| 0.25 ML CO and O coadsorption | router exposed surface construction only | coverage and coadsorption path is absent |
| Pt13 on TiO2 | router exposed prototype construction only | cluster/support and interface path is absent |

The current 4B model therefore meets the narrow recipe-translation purpose.
The project has no valid evidence that 4B can build the wider catalyst models
used in journal calculations.

## Training and evaluation plan

### 1. Freeze the runtime and rebuild provenance

Finish the current compound and adsorbate geometry changes, freeze a registry
version, and regenerate train, validation, test, novel-phrasing, and corpus
reports. Reject evaluations whose registry fingerprint or output hashes do not
match the generated dataset.

### 2. Add deterministic representation before examples

Implement and test the missing operations:

- ordered and substituted alloy slabs plus surface vacancy/substitution;
- terminations, stepped facets, and distinct fcc/hcp hollow sites;
- repeated adsorbates, coverage, and coadsorption with overlap validation;
- clusters, single atoms, lattice matching, and support interfaces.

Each operation needs deterministic replay, composition/site/coverage
invariants, atom-budget enforcement, and fail-closed routing. Training cannot
teach the model to call operations that the registry cannot express.

### 3. Build a catalyst benchmark before the next fine-tune

Create 1,000 to 1,500 reviewed structure cases, then render three to five
prompt forms per case. Favor new structures over extra paraphrases. Draw the
taxonomy from public journal structures and open sources such as Catalysis-Hub,
OC20, and OC22; record DOI, license, structure provenance, and sanitization.
Do not include proprietary structures, POTCAR data, credentials, or raw user
transcripts.

Allocate cases across elemental surfaces and adsorption, alloys, oxides,
defects, coverage/coadsorption, and supported clusters/interfaces. Retain the
current r5 cases as replay examples so the new run cannot hide regressions.

### 4. Use challenge splits that test chemistry

Group by normalized final-structure hash, then create separate challenges for
unseen compositions, adsorbates, facets/terminations, and interfaces. Hold out
complete publications or material-adsorbate combinations. Novel wording alone
does not measure chemical generalization.

Score normalized structure, composition, facet and termination, adsorption
site, height/orientation, coverage, constraints, overlaps, and interface gap.
Keep exact tool sequence as a diagnostic because several recipes can produce
the same valid structure.

### 5. Train the 4B model first

Start from the proven r5 QLoRA recipe: prompt cap 6, 200-step pilot, balanced
family sampling, validation checkpoints, and best-checkpoint restore. Change
one training variable per run. The 300-step r6/r7 recipe regressed despite a
lower validation loss, so loss cannot select the deployment adapter.

Run a one-case-per-family gate before the full benchmark. Require:

- at least 95% overall exact structure and invariant satisfaction;
- at least 90% exact and invariant satisfaction in each catalyst family;
- at least 98% schema execution and finish rates;
- 100% adversarial safety with zero forbidden executions;
- no geometry regression on the retained r5 cases.

Add expert review for an untouched sample before promotion. Unsupported or
ambiguous requests must clarify or refuse instead of guessing a structure.

### 6. Decide model size with a paired test

Keep 4B if it passes the expanded gate. If 4B still misses the thresholds after
the registry, routing, balance, and dataset checks pass, train one 7B or 8B
comparison with the same records, seed, tool exposure, and evaluation IDs.
Scale only if the larger model improves the failed chemistry or long-horizon
families enough to justify its latency and memory cost.
