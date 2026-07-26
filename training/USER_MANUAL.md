# ASE-agent LLM — User Manual

How to talk to the fine-tuned structure-building model and reliably get the
**correct** structure. For how the model is trained and promoted, see
`training/HANDBOOK.md`.

---

## 1. What the model does

You type a structure request in plain language. The model replies with a
sequence of **ASE tool calls**; a deterministic workspace executes them and
returns one `ase.Atoms` structure. The model does not run code or invent
numbers — it selects tools from a fixed registry and fills their arguments, and
the workspace validates and executes every call.

```
"a 4-layer 2×2 Cu(100) slab, 12 Å vacuum, bottom 2 layers fixed"
        │
        ▼  (model plans)
  build_surface → freeze_layers → finish
        │
        ▼  (workspace executes + validates)
  Cu16 slab, 8 atoms constrained, cell 5.105 × 5.105 × 30.253 Å
```

---

## 2. Quick start

The entry point is `ASE_auto_build` (`ase_auto_build.ase_agent.cli`). For a
step-by-step walkthrough — installation, activation, and feeding the result into
a DFT run — see **[docs/TUTORIAL_ASE_AGENT.md](../docs/TUTORIAL_ASE_AGENT.md)**;
this manual is the reference for *what to say* to the model.

Interactive REPL (loads the model once, then answers requests until you quit):

```bash
cd /home/tlclab/Structure_building
PYTHONPATH=src:. .venv/bin/python -m ase_auto_build.ase_agent.cli
```

One-shot, and the same command once installed with `pip install -e ".[agent]"`:

```bash
… .venv/bin/python -m ase_auto_build.ase_agent.cli \
  --prompt "Build a 4-layer 2x2 Cu(111) slab with 12 Å vacuum"

ASE_auto_build "Build a 4-layer 2x2 Cu(111) slab with 12 Å vacuum"
```

Useful flags: `--base-only` (talk to the frozen base model without the adapter,
for an A/B feel), `--adapter <dir>` (evaluate a different run), `--strict`
(§7), `--json`, `--no-write`, `--max-turns`, `--max-new-tokens`. Model load
takes a minute or two (4-bit quantization); each request then answers in
seconds. `--adapter`, `HF_HOME`, `HF_HUB_OFFLINE`, and the WSL CUDA `PATH` are
resolved for you when you run from the repo root.

The entry point prints, per request: an advisory slot-coverage hint, each
executed call with a pass/fail flag, the final structure's formula, atom count,
cell, periodicity, constrained-atom count, and content hash — and writes a
`POSCAR` plus a `structure.json` provenance sidecar to `structures/<case>/`,
which is directly usable as `vasp-auto structures/<case> --prepare`.

`training/chat_agent.py` still works as a thin deprecated shim over the same
CLI.

---

## 3. The golden rule: state every required slot

**A request maps to one correct structure only if it names every structural
determinant.** The model was trained on prompts that obey this rule; give it an
ambiguous prompt and it will pick *a* valid answer, which may not be the one you
meant. This is the same structured-request rule the corpus obeys
(`training/CORPUS_RULE.md`).

Required slots by region — include all of these:

| Region | You must state | Optional |
| --- | --- | --- |
| bulk | element, **crystal phase** | supercell repeat, cell convention (see §6) |
| surface / slab | element, facet (Miller), layers, vacuum | crystal phase, repeat |
| surface + constraint | …the above **+ which layers to fix** | crystal phase, repeat |
| atomic adsorption | element, facet, layers, vacuum, **adsorbate, site, height** | crystal phase, repeat |
| molecular adsorption | …the above **+ anchor atom** | crystal phase, repeat |
| molecule | species (formula), box size | charge, multiplicity |
| nanotube | chirality (n,m), length | vacuum |
| prototype | prototype name | — |
| vacancy | element, crystal phase, **which atom** | repeat |
| substitution | element, crystal phase, **which atom, dopant** | repeat |

If you deliberately omit exactly one required slot, the model is trained to
**ask a clarifying question** rather than guess (see §7).

---

## 4. The tool vocabulary (what the model emits)

You never call these yourself, but knowing them helps you read a trajectory and
phrase requests. Core builders and their **required** arguments:

| Tool | Required args | Builds |
| --- | --- | --- |
| `build_bulk` | name, element (+ `crystal`, `cubic`, `a`, `c`, `repeat`) | Elemental bulk crystal |
| `build_surface` | name, element, miller (+ layers, vacuum, crystal, a, repeat) | Surface slab |
| `build_molecule` | name, species (+ box, charge, multiplicity) | Molecule in a box |
| `build_nanotube` | name, n, m (+ element, length, bond, vacuum) | Single-wall nanotube |
| `build_prototype` | name, prototype (+ a, c, vacuum) | Named binary prototype |
| `add_atomic_adsorbate` | name, element (+ site, site_index, xy, height) | One adsorbed atom |
| `add_molecular_adsorbate` | name, species (+ anchor, site, height) | Adsorbed molecule |
| `freeze_layers` | name, side, layers (+ axes, tolerance) | Fix top/bottom layers |
| `make_vacancy` | name, selector | Delete selected atoms |
| `substitute` | name, selector, element | Replace selected atoms |
| `repeat` | name, repeat | Supercell |
| `ask_clarification` | question (+ choices, field) | Pause for one missing choice |
| `finish` | name | Validate + select the final structure |

Prototype names available: `graphene`, `graphite`, `hBN`, `anatase-TiO2`,
`rutile-TiO2`.

---

## 5. Worked examples

### 5a. Surface slab with frozen layers (fully worked)

**Prompt:** `Build a 2x2 Cu(100) slab with 4 layers and 12 Å vacuum, then freeze the bottom 2 layers.`

Model trajectory:

```json
{"name":"build_surface","arguments":{"crystal":"fcc","element":"Cu","layers":4,"miller":[1,0,0],"name":"slab","repeat":[2,2,1],"vacuum":12.0}}
{"name":"freeze_layers","arguments":{"axes":"xyz","layers":2,"name":"slab","side":"bottom"}}
{"name":"finish","arguments":{"name":"slab"}}
```

Result: **Cu16**, 16 atoms, cell 5.105 × 5.105 × 30.253 Å, pbc [T, T, F], **8
constrained atoms** (the bottom two layers). ✅

### 5b. Atomic adsorption

**Prompt:** `Put one O atom 1.8 Å above the ontop site of a 2x2 Cu(100) 5-layer slab with 12 Å vacuum.`
Tool sequence: `build_surface → add_atomic_adsorbate → finish`. Result: **Cu20O1**
(20-atom slab + one O on top). ✅

### 5c. Molecule, nanotube, prototype (verified atom counts)

| Prompt | Tool sequence | Result |
| --- | --- | --- |
| `Build an H2O molecule in a 12 Å box.` | `build_molecule → finish` | H2O, 3 atoms |
| `Build a (6,3) carbon nanotube, 2 unit cells long.` | `build_nanotube → finish` | C, 168 atoms |
| `Build the hBN prototype.` | `build_prototype → finish` | h-BN prototype |

### 5d. Vacancy and substitution

| Prompt | Tool sequence | Selector the model emits |
| --- | --- | --- |
| `Build a 2x1x1 bcc Fe cell with a vacancy at atom 1.` | `build_bulk → repeat → make_vacancy → finish` | `{"indices":[1]}` |
| `Build a 2x2x1 fcc Cu supercell and replace the first atom with Au.` | `build_bulk → repeat → substitute → finish` | `{"indices":[1]}`, element `Au` |

---

## 6. Disambiguation guide (getting *correct*, not just *valid*)

The most common way to get a valid-but-not-what-you-wanted structure is leaving
a determinant unstated. The flagship case:

**Bulk cell convention.** `Create a bulk bcc W crystal using a 2x2x1 repeat`
does not say *which* unit cell. The model picks the **primitive** cell:

| Cell you ask for | build_bulk | Result |
| --- | --- | --- |
| unstated → model picks primitive | `cubic` omitted | **W4**, angles 109.47° |
| conventional cubic | `cubic: true` | **W8**, angles 90° |

Both are legitimately "bcc W" with the right lattice constant — they are
*different conventions*. Fix it by saying so:

> `Create a **conventional cubic** bulk bcc W crystal with a 2x2x1 repeat.`

Other disambiguation tips:

- **Surfaces:** always give **layers** and **vacuum** — a slab with unstated
  thickness or vacuum is ambiguous. State the facet as Miller indices, e.g.
  `(100)`, `(111)`.
- **Adsorption:** name the **site** (`ontop`, `bridge`, `fcc hollow`, …) and the
  **height**; for a molecule also the **anchor** atom (e.g. "bonded through the
  carbon").
- **Molecules:** give the **box** edge; the structure's cell depends on it.
- **Supercells:** a repeat like `2x2x1` is a determinant — state it if you want
  it, omit it if you want the primitive/conventional single cell.

---

## 7. What happens at the edges

- **Out-of-scope request** → the bounded router **refuses before any tool runs**:

  ```
  request refused before execution: ValueError: unsupported structure request;
  ask for a bulk crystal, surface/slab, molecule, nanotube, prototype,
  adsorption, vacancy, substitution, or constraint
  ```

- **Exactly one missing slot** → the model calls `ask_clarification` and waits;
  answer in the REPL at the `clarify>` prompt, e.g. you say "an iron surface" and
  it asks which facet.

- **A stated number the model drops** → the entry point runs a deterministic
  post-build check comparing the numbers your request stated (`layers`,
  `vacuum`, `box`, `height`) against the calls actually executed, and warns:

  ```
  [warn] you asked for height 2.5, but the structure uses 1.8 -- the model
  omitted 'height', so add_atomic_adsorbate used its default.
  ```

  `--strict` turns that warning into exit code 7 so a batch script fails rather
  than quietly producing wrong DFT inputs. The known instance on r5: an
  adsorption height phrased as `2.5 Å above the ontop site` is dropped, while
  `at a height of 2.5 Å` is honoured. Put the number next to the word "height".

---

## 8. What "correct" means

Two bars, both reported by the entry point / evaluator:

- **Exact structure** — the output's content hash equals the canonical
  reference. The strict bar used for promotion.
- **Invariants satisfied** — formula, atom count, periodicity, cell
  lengths/angles, and constrained-atom count match within tolerance. Tolerant of
  harmless atom-ordering and floating-point differences.

A structure can satisfy invariants but differ in exact hash (e.g. a different but
equivalent atom ordering). For most purposes invariants are what you care about;
exact hash is the reproducibility guarantee.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Refused as "unsupported" | Request outside the supported regions, or a determinant so vague the router can't route it | Rephrase within a supported region (§3); state the required slots |
| Wrong atom count / cell on a bulk | Cell-convention ambiguity | Say "conventional cubic" or "primitive" (§6) |
| Model asks a question | You omitted exactly one required slot | Answer at `clarify>` |
| Slab too thin/thick, or squished | Missing `layers` or `vacuum` | State both explicitly |
| Adsorbate in the wrong place | Missing `site`/`height` (or `anchor` for a molecule) | Name the site, height, and anchor |
| Adsorbate at 1.8 Å when you asked for another height | The model dropped `height` (§7) | Phrase it `at a height of N Å`; run with `--strict` |
| Model load is slow | 4-bit base model load (~1–2 min) | One-time per session; the REPL reuses it |
