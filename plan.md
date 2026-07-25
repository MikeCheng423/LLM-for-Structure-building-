# Plan: a production-grade LLM controller for ASE inside `vasp_auto`

## Outcome

Build a reliable, conversational structure-building agent in `vasp_auto` that
turns natural-language requests into typed, deterministic ASE operations. The
model plans and selects tools; `vasp_auto` owns geometry, validation, provenance,
filesystem access, and calculation execution.

The finished system should handle requests such as:

> Build a 3x3 four-layer Pt(111) slab with 15 A vacuum, adsorb CO at an ontop
> site, fix the bottom two layers, rotate CO by 20 degrees, and prepare an EMT
> relaxation. Then make a second version with an O vacancy.

It should show the proposed steps, execute reversible structure edits in a
session, render the result in the existing editor, and require confirmation
before saving files or starting a calculation.

Success is not measured by whether the model writes plausible Python. Success
means that the requested final structure is produced through valid tools, its
provenance is reproducible, invalid operations roll back cleanly, and expensive
work cannot start without authorization.

## Existing foundation to preserve

`vasp_auto` already contains most of the deterministic substrate:

- `nl_agent.py`: OpenAI-compatible tool-calling loop and named workspace.
- `nl_builder.py`: single-command quick builder and ASE-backed base builders.
- `structure.py`: POSCAR manipulation, selection, vacancy, substitution,
  interstitials, cell operations, constraints, matching, combination, and simple
  adsorbates.
- `ase_tools.py`: ASE import, bulk/slab/molecule/crystal/nanotube builders and
  NEB interpolation.
- `ase_engine.py`: calculator configuration and ASE SCF/relax job execution.
- `ai_providers.py`: provider-independent chat-completions transport.
- `vasp_auto_ui/server.py` and `static/index.html`: AI builder endpoints, remote
  machine awareness, structure editor, preview, save, and job submission.

The implementation should reuse these modules and preserve the current
`/api/nl_build` quick mode while replacing the internals of `/api/nl_agent`
incrementally. The lean remote engine must not acquire a mandatory ASE or LLM
dependency; ASE remains lazy-loaded through the existing `builder`/`local`
extras.

## Design principles

1. **The LLM never owns coordinates.** It emits typed tool calls, not POSCAR
   text, arrays of atomic positions, Python, shell commands, or filesystem paths
   outside approved roots.
2. **Keep live ASE state.** A structure-building session stores `ase.Atoms`
   objects directly so tags, constraints, magnetic moments, charges, velocities,
   and calculator metadata are not lost in POSCAR round-trips.
3. **Every edit is transactional.** Execute on a copy, validate, then commit a
   new revision. Errors leave the prior revision untouched.
4. **Planning and execution are separate.** The model may propose multiple
   steps. `vasp_auto` validates each call and decides whether it is safe to run,
   requires confirmation, or is forbidden.
5. **Structure edits are cheap; calculations are jobs.** Geometry operations may
   execute interactively. Optimization, MD, ML potentials, VASP, and QE go
   through the existing preview/job/submission machinery.
6. **Observations are compact and physical.** Return formula, atom count, cell,
   PBC, constraints, extrema, layer/site summaries, warnings, and calculation
   status—not full coordinates—to the model.
7. **Reproducibility is a first-class output.** Every result includes a canonical
   recipe and content hash that can be replayed without an LLM.
8. **Ambiguity must be visible.** Defaults are allowed for low-risk details;
   important ambiguity uses an explicit clarification request.

## Target architecture

```text
user conversation
       |
       v
ASE agent controller -------- ai_providers.py
       |
       +---- plan/clarify/execute loop
       |
       v
declarative tool registry ---- policy + JSON-schema validation
       |
       v
transactional ASEWorkspace --- named Atoms + revision DAG + provenance
       |
       +---- build/edit/inspect tools (interactive, reversible)
       |
       +---- calculation proposal (approval required)
       |          |
       |          v
       |     ase_engine / workflow / remote submission
       |
       v
validators + compact observations
       |
       v
existing UI structure editor -> preview -> save/submit
```

## Package layout

Create a package rather than continuing to grow `nl_agent.py`:

```text
vasp_auto/ase_agent/
    __init__.py
    schemas.py          # ToolCall, observation, plan, revision and error types
    registry.py         # declarative tool registry + schema export
    workspace.py        # ASEWorkspace, revisions, transactions, undo/redo
    selectors.py        # safe atom/layer/site selection language
    validation.py       # structural and physical sanity checks
    policy.py           # limits, path rules, confirmation classes
    observations.py     # compact model/UI summaries
    tools_build.py      # bulk, surface, molecule, crystal, nanotube, import
    tools_edit.py       # repeat, vacancy, substitute, add/delete/move/rotate
    tools_surface.py    # layers, adsorption sites, adsorbates, interfaces
    tools_constraints.py# FixAtoms/FixCartesian and layer-based constraints
    tools_inspect.py    # formula/cell/distance/coordination/sites/diff
    tools_compute.py    # calculation proposals, never direct unapproved runs
    controller.py       # provider-neutral tool-calling state machine
    recipe.py           # canonical replayable recipe + hashing
```

Keep `nl_agent.agent_build_from_text()` as a compatibility adapter that creates
a one-turn session, calls the new controller, and returns the current
`(structure_dict, transcript)` shape.

## Core data model

### Workspace

`ASEWorkspace` should own:

```python
session_id: str
structures: dict[str, StructureHistory]
active_name: str | None
result_name: str | None
conversation: list[Message]
pending_plan: Plan | None
pending_confirmation: Confirmation | None
budgets: SessionBudget
```

Each `StructureHistory` contains immutable revisions. A revision records:

- revision ID and parent revision(s);
- copied `ase.Atoms` state;
- tool name and normalized arguments;
- input/output content hashes;
- timestamp and warnings;
- optional calculation proposal/result reference.

Combination operations have two parents. Undo moves the active pointer; it does
not mutate history. Sessions should support branching (`slab`, `slab_o`,
`slab_vacancy`) without copying provenance manually.

### Transaction contract

Every mutating tool follows one path:

```text
resolve inputs -> copy Atoms -> execute -> normalize -> validate -> commit
                                                |
                                                +-> error: discard copy
```

The executor, not each tool, implements this contract. This guarantees uniform
rollback, timing, observation generation, and audit records.

### Canonical recipe

Store recipes as versioned JSON, for example:

```json
{
  "schema_version": 1,
  "steps": [
    {"tool": "build_surface", "args": {"name": "slab", "element": "Pt",
      "miller": [1, 1, 1], "layers": 4, "vacuum": 15.0}},
    {"tool": "repeat", "args": {"name": "slab", "repeat": [3, 3, 1]}},
    {"tool": "add_adsorbate", "args": {"name": "slab", "species": "CO",
      "site": "ontop", "height": 1.9}},
    {"tool": "freeze_layers", "args": {"name": "slab", "side": "bottom",
      "layers": 2, "axes": "xyz"}}
  ]
}
```

Recipes must replay offline without a provider or API key. Save the recipe next
to a committed case as `structure_recipe.json`.

## Tool registry

Define tools once using a `ToolSpec` containing:

- stable name and schema version;
- plain-language description;
- JSON Schema for arguments;
- executor callable;
- `read_only`, `mutates_structure`, or `proposes_compute` classification;
- estimated cost/risk;
- confirmation requirement;
- capability/dependency predicate;
- aliases only for backward compatibility.

Generate the model's function schemas, server metadata, UI help, validation,
and documentation from this registry. Do not maintain separate hand-written
schemas in the prompt and executor.

### Phase-one build and import tools

- `build_bulk(name, element, crystal?, a?, c?, cubic?, repeat?)`
- `build_surface(name, source|element, miller, layers, vacuum, termination?,
  orthogonal?, repeat?)`
- `build_molecule(name, species, box?, charge?, multiplicity?)`
- `build_crystal(name, symbols, basis, spacegroup, cell_parameters)`
- `build_nanotube(name, n, m, length?, bond?, vacuum?)`
- `build_prototype(name, prototype, parameters?)`
- `import_structure(name, location, format?)`
- `clone(source, name)` and `rename(old, new)`

`import_structure` accepts a server-resolved `{path, machine}` reference, not an
arbitrary path invented by the model. Apply the same remote-root and local-root
boundaries already used by the UI file pickers.

### Phase-one edit tools

- `repeat`, `set_cell`, `scale_cell`, `center`, `wrap`
- `make_vacancy`, `substitute`, `add_atom`, `delete_atoms`
- `move_atoms`, `translate`, `rotate`
- `combine`, `match_and_combine`
- `freeze_atoms`, `freeze_layers`, `clear_constraints`
- `add_atomic_adsorbate`, `add_molecular_adsorbate`

Wrap existing `structure.py` behavior first. Convert to a pure `Atoms` operation
only when needed to preserve ASE metadata or remove duplicate geometry logic.

### Selection language

The model must not pass arbitrary Python expressions. Implement a typed selector
object with combinators:

```json
{"element": ["Pt"], "layer": {"side": "bottom", "count": 2}}
{"indices": [1, 4, 8]}
{"region": {"axis": "z", "op": ">", "value": 0.75, "fractional": true}}
{"coordination": {"element": "O", "max": 2}}
```

Keep the existing string selector parser for CLI compatibility, but convert it
to the typed selector internally. Selection observations return the matched
count and a bounded list of indices before mutation. Empty or unexpectedly
large selections should require correction or confirmation.

### Surface and adsorption tools

This is the most important extension beyond the current agent:

- identify geometric layers with a configurable tolerance;
- enumerate ontop, bridge, fcc-hollow, hcp-hollow, and custom fractional sites;
- orient a molecular adsorbate by anchor atom, axis, azimuth, and tilt;
- detect overlaps using covalent radii;
- preserve slab cell/PBC and constraints;
- return site and minimum-distance summaries.

Use ASE surface/adsorption utilities where they express the requested operation,
and add deterministic project code where ASE lacks named-site semantics.

### Inspection and validation tools

- `inspect_structure(name)`
- `compare_revisions(name, revision_a, revision_b)`
- `measure_distance`, `measure_angle`, `coordination_summary`
- `list_layers`, `list_adsorption_sites`, `list_constraints`
- `validate_structure(name, profile)`
- `preview_recipe(name)`
- `finish(name)`
- `ask_clarification(question, choices?, field?)`

The controller must call `finish` exactly once for a completed request. If the
request has a material ambiguity (unknown surface, unspecified interface order,
multiple plausible atom selections), it should use `ask_clarification` instead
of silently finishing.

## Calculation control

Do not let interactive tools attach arbitrary Python calculator objects or run
long calculations inside the HTTP request.

Expose proposal tools:

- `propose_single_point(name, calculator, parameters)`
- `propose_relaxation(name, calculator, fmax, max_steps, relax_cell?)`
- `propose_md(name, calculator, ensemble, temperature, timestep, steps)`
- `propose_vasp_workflow(name, workflow, cpus?, machine?)`
- `propose_qe_workflow(name, workflow, cpus?, machine?)`

The proposal is validated against an allowlisted calculator catalog and shown
to the user with estimated atom count, steps, calculator, machine, and relevant
resource limits. Only a separate confirmation endpoint may translate it into an
existing `ase_engine`, VASP, QE, workflow, or remote job submission.

Initial calculator allowlist:

- interactive/test: EMT and Lennard-Jones;
- configured external ASE calculators only when explicitly enabled;
- ML calculators through the existing `ml_tools` integration;
- VASP/QE only through current job preparation and submission paths.

No tool may accept module names, class names, command strings, shell snippets,
or arbitrary calculator constructors from the model.

## Controller behavior

Implement an explicit state machine rather than an open-ended chat loop:

```text
NEW -> PLANNING -> EXECUTING -> (NEEDS_CLARIFICATION | NEEDS_CONFIRMATION)
                         |                         |
                         +---------- resume -------+
                         |
                         +-> FINISHED | FAILED | BUDGET_EXHAUSTED
```

Controller rules:

- validate provider responses and tool-call arguments before dispatch;
- permit parallel tool calls only when all are read-only or operate on
  independent structures;
- stop after configurable model turns, tool calls, atoms, wall time, and errors;
- feed structured errors back once so the model can repair a call;
- prevent repeating the same failed call more than twice;
- require `finish` and final validation before returning a structure;
- keep full internal messages for protocol correctness but expose a sanitized
  transcript to the UI;
- support a dry-run planning mode that validates without mutating a session.

The system prompt should be short and generated partly from registry metadata.
Rules that can be enforced in code must not depend on prompt compliance.

## Validation and policy

### Structural validation after every mutation

- finite 3x3 non-singular cell when PBC requires it;
- finite positions and valid chemical symbols;
- atom count between 1 and configured maximum;
- no accidental atom pairs below a radii-based hard threshold;
- valid PBC shape and constraints referencing existing atoms;
- positive volume for periodic structures;
- molecule/slab-specific vacuum and boundary warnings;
- preserve requested formula or report exactly how it changed;
- surface operations preserve intended Miller/layer metadata when available.

Warnings may commit; hard failures roll back. The final validator uses a stricter
profile than intermediate revisions.

### Default resource policy

- maximum 2,000 interactive atoms;
- maximum 24 model turns and 64 tool calls per session;
- maximum 20 stored revisions per named structure unless persisted;
- maximum 200 optimization steps and 10,000 MD steps without elevated approval;
- maximum import size and parser timeout;
- case writes limited to configured local/remote case roots;
- no deletion or overwrite of an existing saved case;
- no network access from structure tools;
- no arbitrary Python or shell execution.

Make limits configurable, but apply server-side ceilings that request bodies
cannot raise.

## API and UI integration

Add session-oriented endpoints while keeping existing endpoints:

```text
POST /api/ase-agent/session        create a session
GET  /api/ase-agent/session?id=   current state and sanitized transcript
POST /api/ase-agent/message        add user text and advance controller
POST /api/ase-agent/clarify        answer a pending clarification
POST /api/ase-agent/confirm        approve/reject a compute or write action
POST /api/ase-agent/undo           move active revision back
POST /api/ase-agent/redo           move active revision forward
POST /api/ase-agent/commit         send chosen revision to editor/save flow
DELETE /api/ase-agent/session      discard server state
```

For the first implementation, keep sessions in a locked in-memory store with a
TTL and maximum count. Add optional persistence only after the serialization and
privacy policy are settled. Never serialize API keys in session data.

Extend the existing AI Builder card rather than creating a separate tab:

- conversation history with clarification controls;
- proposed plan with per-step status;
- compact tool result and warning display;
- structure version/branch selector;
- Undo, Redo, Compare, and Reset controls;
- live viewer refresh after committed edits;
- explicit **Apply to editor** button;
- separate **Prepare calculation** and confirmation panel;
- downloadable `structure_recipe.json`;
- provider/model selection continues to use the existing controls.

The UI should never apply a partially failed revision. It may preview the latest
valid revision while the agent is still waiting for clarification.

## Training and data strategy

Do not block the initial product on fine-tuning. First ship a model-independent,
well-tested tool environment and evaluate strong tool-capable models against it.

Once the interface is stable:

1. Generate canonical valid recipes from a grammar over the registry.
2. Execute every recipe and retain only examples that pass validators.
3. Create multiple natural-language paraphrases, including scientific notation,
   shorthand, reordered constraints, and realistic omissions.
4. Include clarification, tool-error recovery, and revision requests.
5. With explicit opt-in, log user request, sanitized transcript, recipe hash,
   acceptance/rejection, and subsequent manual edits—never API keys or arbitrary
   imported file contents.
6. Curate a gold set, then supervised-fine-tune a tool-capable base model on full
   assistant/tool trajectories.
7. Use accepted versus corrected plans as preference data only after sufficient
   real examples exist.

Training and deployment must use the same tool-exposure policy. A trajectory
that exposes only the tools appearing in its reference answer is useful for
diagnosis but leaks an oracle-selected tool subset. It cannot establish
deployment readiness. Either expose the complete registry during training and
evaluation, or introduce a deterministic, versioned capability router that
returns a conservative tool superset and prove 100% router recall on the corpus.
The router itself cannot execute operations or weaken registry validation.

Split train/test by recipe family and composition, not by paraphrase, so the
same underlying structure cannot leak across splits.

Primary evaluation metrics:

- schema-valid tool-call rate;
- end-to-end execution success;
- final structural invariant satisfaction;
- correct clarification rate;
- recovery from tool errors;
- recipe replay equivalence;
- unnecessary tool calls and latency;
- forbidden-action rate (must be zero);
- human acceptance without manual geometry correction.

## Implementation phases

### Phase 0 — lock the contracts

Deliver:

- inventory existing operations and map each to reuse/wrap/rewrite;
- `schemas.py`, registry skeleton, error taxonomy, policy configuration;
- golden prompt-to-recipe evaluation cases;
- tests that capture current `nl_agent` behavior before migration.

Acceptance:

- registry exports valid OpenAI-compatible function schemas;
- no ASE import occurs when the agent feature is unused;
- existing quick builder and remote engine tests remain green.

### Phase 1 — transactional structure workspace

Deliver:

- live `ase.Atoms` workspace, immutable revisions, branching, undo/redo;
- converters between `Atoms`, UI payload, and existing structure dictionaries;
- canonical recipe replay and hashing;
- build/import/edit/inspect tools wrapping current functionality;
- validation and compact observations.

Acceptance:

- every tool has success, invalid-argument, rollback, and replay tests;
- a failed mutation leaves the structure hash unchanged;
- replaying a recipe reproduces formula, cell, PBC, positions, and constraints
  within defined tolerances;
- metadata survives tool sequences without POSCAR round-trips.

### Phase 2 — controller and compatibility adapter

Deliver:

- explicit controller state machine and budgets;
- clarification and structured error repair;
- registry-derived tool schemas and system prompt;
- `nl_agent.agent_build_from_text` adapter;
- provider-fake tests covering multi-turn tool-call protocol.

Acceptance:

- representative single- and multi-structure requests finish correctly;
- malformed JSON, unknown tools, repeated failures, and missing `finish` produce
  stable user-facing errors;
- no model response can bypass registry validation or policy.

### Phase 3 — surface chemistry quality

Deliver:

- robust layer detection and freeze-by-layer;
- named adsorption site enumeration;
- molecular orientation/anchoring;
- overlap checks and slab-specific validation;
- interface matching integrated with existing `match_supercells` and combine.

Acceptance:

- golden Pt(111), Cu(100), stepped/heterogeneous slab, atomic adsorbate,
  molecular adsorbate, vacancy, and interface cases pass invariant tests;
- bottom-layer constraints persist into saved POSCAR selective dynamics;
- site choice and minimum distances are reported to the user.

### Phase 4 — conversational UI

Deliver:

- session endpoints, TTL cleanup, thread-safe store;
- conversation/plan/version controls in the current AI Builder;
- clarification and warning UI;
- apply-to-editor and recipe download;
- local/remote import selection using existing machine-aware picker patterns.

Acceptance:

- refresh/retry cannot duplicate an already committed tool step;
- concurrent browser sessions are isolated;
- undo/redo and branch selection update the viewer deterministically;
- nothing is saved or submitted without the existing explicit user actions.

### Phase 5 — approved calculation proposals

Deliver:

- typed compute proposal tools and calculator catalog;
- confirmation endpoint and UI summary;
- adapters into `ase_engine`, `ml_tools`, workflows, and remote submission;
- job token/status linkage back to the agent session.

Acceptance:

- EMT single-point/relax flows pass locally;
- mock remote VASP/QE proposals produce the same CLI/job configuration as manual
  UI submission;
- rejected proposals perform no writes or process launches;
- calculator, step, atom, path, and machine limits are enforced server-side.

### Phase 6 — evaluation corpus and optional fine-tuning

Deliver:

- recipe generator and paraphrase ingestion format;
- offline evaluator with structural metrics;
- opt-in sanitized transcript export;
- baseline comparison across configured providers/models;
- full-registry or validated-router evaluation with identical record IDs for
  base and candidate models;
- fine-tuning dataset only after schema/tool versions stabilize.

Acceptance:

- every training target executes successfully before export;
- held-out evaluation is grouped by recipe family;
- oracle-selected tool subsets are reported as diagnostics and cannot satisfy
  the deployment promotion gate;
- model upgrades are accepted only when execution and safety metrics do not
  regress.

## Test strategy

Add focused suites:

```text
tests/test_ase_agent_registry.py
tests/test_ase_agent_workspace.py
tests/test_ase_agent_tools_build.py
tests/test_ase_agent_tools_edit.py
tests/test_ase_agent_surface.py
tests/test_ase_agent_validation.py
tests/test_ase_agent_policy.py
tests/test_ase_agent_controller.py
tests/test_ase_agent_recipe.py
tests/test_ase_agent_api.py
```

Testing layers:

1. Schema and normalization unit tests with no provider.
2. Tool tests against real ASE with small structures.
3. Property/invariant tests over generated recipes.
4. Controller protocol tests with deterministic fake model responses.
5. API tests for session isolation, confirmation, TTL, and remote path checks.
6. End-to-end UI smoke cases with recorded provider responses.
7. Existing full-suite regression, especially structure, case save, ASE engine,
   remote, and UI tests.

Do not use exact tool-sequence equality as the sole semantic metric; different
valid recipes may produce equivalent structures. Compare normalized structures
and requested invariants as well.

## Initial golden scenarios

1. fcc Cu bulk, then 2x2x2 repeat.
2. 3x3 Pt(111), four layers, 15 A vacuum, bottom two layers fixed.
3. CO on Pt(111) ontop with explicit C anchor and orientation.
4. O in fcc versus hcp hollow sites as separate branches.
5. TiO2 substitution and oxygen vacancy using typed selections.
6. Import CIF locally and from a saved remote, then center and save as POSCAR.
7. Match graphene and a metal slab, present alternatives, combine selected pair.
8. Ambiguous "iron surface" request triggers clarification.
9. Invalid atom selection rolls back and the model repairs it.
10. EMT relaxation proposal waits for approval and then appears as a normal job.
11. VASP workflow proposal inherits selected remote, cores, and concurrency rule.
12. Attempted path escape, arbitrary calculator, shell command, and 100,000-atom
    repeat are rejected before execution.

## Migration and compatibility

- Keep `/api/nl_build` and the quick JSON-command builder unchanged initially.
- Adapt `/api/nl_agent` to a one-turn session after Phase 2 so existing UI calls
  and clients keep working.
- Add session endpoints alongside it in Phase 4.
- Preserve existing structure-dictionary interfaces at module boundaries while
  the new workspace uses `Atoms` internally.
- Version tools and recipes; old recipes use a migration table rather than
  silently changing meaning.
- Do not expose experimental compute proposal tools until confirmation and
  policy tests are complete.

## Explicit non-goals

- Training a foundation model from scratch.
- Letting the model emit or execute unrestricted ASE Python.
- Replacing the existing visual editor, CLI, workflow engine, or job manager.
- Automatically accepting physically meaningful scientific choices on behalf
  of the researcher.
- Running long calculations synchronously in an HTTP request.
- Treating an LLM-generated structure as scientifically validated merely because
  it is syntactically valid.

## Definition of done

The design is complete when a user can conduct a multi-turn build/edit session,
inspect and undo every deterministic operation, replay the exact recipe without
an LLM, apply a chosen revision to the current editor, and optionally prepare an
approved ASE/VASP/QE calculation through existing job paths. Invalid or
unauthorized operations must produce clear errors without changing structures,
writing outside configured roots, or starting processes. The golden evaluation
set and full `vasp_auto` regression suite must pass before the old worker mode is
retired.
