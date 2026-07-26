# Tutorial — The structure-building LLM (`ASE_auto_build`)

Describe a structure in plain English; get a validated structure and a
ready-to-run DFT case. This tutorial takes you from *nothing installed* to
*a finished VASP calculation*, and is honest about where the model still
gets things wrong.

**Prerequisites:** a working `vasp-auto` checkout, an NVIDIA GPU (≈6 GB free),
and the fine-tuned adapter. No prior knowledge of the model is assumed.

**Related reading:** [training/USER_MANUAL.md](../training/USER_MANUAL.md)
(reference for the request rule and tool vocabulary) and
[training/HANDBOOK.md](../training/HANDBOOK.md) (how the model was trained and
promoted).

---

## 1. What this tool is

You type a request. A fine-tuned 4-billion-parameter model (Qwen3-4B-Instruct,
QLoRA adapter) plans a sequence of **deterministic ASE tool calls**. A bounded
workspace executes and validates each one, and the finished structure is written
to disk as a VASP case.

```
"a 4-layer 2×2 Cu(100) slab, 12 Å vacuum, bottom 2 layers fixed"
        │
        ▼  model plans (it picks tools + arguments; never coordinates)
  build_surface → freeze_layers → finish
        │
        ▼  workspace executes + validates
  Cu16 slab, 8 atoms constrained, cell 5.105 × 5.105 × 30.253 Å
        │
        ▼  written
  structures/Cu16-b6a1d8b0/{POSCAR, structure.json}
        │
        ▼
  vasp-auto structures/Cu16-b6a1d8b0 --prepare
```

**What it is not.** It does not run code, invent atomic coordinates, read your
files, or reach the network. It selects from a fixed tool registry; a
deterministic router refuses anything it cannot route *before a single tool
runs*. The geometry always comes from ASE, never from the model's imagination —
so a misread request gives you a *wrong-but-valid* structure, never a corrupt
one. Section 7 is about catching exactly that.

---

## 2. Activate it

### 2.1 Check the GPU

The model is loaded in 4-bit, so it needs CUDA. On WSL, `/usr/lib/wsl/lib` must
be on `PATH` or torch reports no device (the entry point adds it for you, but
check here first):

```bash
cd /home/tlclab/Structure_building
PATH=/usr/lib/wsl/lib:$PATH .venv/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA RTX 2000 Ada Generation
```

If this prints `False`, fix it before going further — nothing below will work.

### 2.2 Install

The deep-learning stack is an optional extra, because the DFT side of
`vasp-auto` does not need it. Install `torch` first (its build must match your
CUDA version), then the extra:

```bash
.venv/bin/python -m pip install torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e ".[agent]"      # ase, transformers, peft, bitsandbytes
```

That puts the `ASE_auto_build` command on your `PATH`. Without installing you
can always run the module directly, which is what the rest of this tutorial
does so the examples work in a bare checkout:

```bash
PYTHONPATH=src:. .venv/bin/python -m vasp_auto.ase_agent.cli --help
```

### 2.3 Point it at the adapter

The promoted adapter lives at `training/runs/pilot-qwen3-4b-r5/adapter`, and it
is found automatically when you run from the repo root. Elsewhere, name it once:

```bash
export ASE_AUTO_BUILD_ADAPTER=/home/tlclab/Structure_building/training/runs/pilot-qwen3-4b-r5/adapter
export HF_HOME=/home/tlclab/Structure_building/training/cache/huggingface
```

If the adapter cannot be found you get a message listing every path that was
tried — not a stack trace:

```
ASE_auto_build: no LoRA adapter found (looked for adapter_config.json in):
    /home/tlclab/Structure_building/training/runs/pilot-qwen3-4b-r5/adapter
  pass --adapter <dir>, set ASE_AUTO_BUILD_ADAPTER, or use --base-only to talk to
  the frozen base model.
```

Model weights are loaded **offline** by default (`HF_HUB_OFFLINE=1`); pass
`--online` the first time if the base model is not cached yet.

### 2.4 First build

```bash
ASE_auto_build "Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum, then freeze the bottom 2 layers."
```

Loading the 4-bit model takes 1–2 minutes; every request after that answers in
seconds, so prefer one session over many one-shot calls. Real output:

```
Loading Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f ... adapter: .../pilot-qwen3-4b-r5/adapter
Model ready.

request> Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum, then freeze the bottom 2 layers.
[hint] keyword heuristic: looks like a 'surface_constraint' request; required slots that appear stated: element, facet, layers, vacuum, freeze.
[hint] advisory only - a keyword match is not a guarantee the values are the ones you meant.

  state            : FINISHED
  executed calls   :
    1. [ok ] build_surface(crystal='fcc', element='Cu', layers=4, miller=[1, 0, 0], name='slab', repeat=[2, 2, 1], vacuum=12.0)
    2. [ok ] freeze_layers(axes='xyz', layers=2, name='slab', side='bottom')
    3. [ok ] finish(name='slab')
  final structure  :
    formula        : Cu16  (16 atoms)
    cell lengths A : [5.105, 5.105, 30.253]   angles: [90.0, 90.0, 60.0]
    pbc            : [True, True, False]   constrained atoms: 8
    atoms_hash     : 2a1b7c7d858d29cf4f360268731084838b9db92b4d1715712310408d92db51aa
    recipe_hash    : b6a1d8b0256952a93d5ac76205f5126fe0db9a0d8dea4b0788658c16d9ecde64
  wrote            :
    structures/Cu16-b6a1d8b0/POSCAR
    structures/Cu16-b6a1d8b0/structure.json
  next             : vasp-auto structures/Cu16-b6a1d8b0 --prepare
```

Read it top to bottom: the **hint** (before the model ran), the **executed
calls** (what the model actually chose), the **final structure** (what you got),
and **wrote** (where it went). If those four agree with what you meant, you are
done.

---

## 3. Four ways to ask

| Mode | Command | Use it when |
| --- | --- | --- |
| One-shot | `ASE_auto_build "Build an H2O molecule in a 12 A box."` | A single structure |
| Repeated | `ASE_auto_build -p "..." -p "..."` | A few structures, **one** model load |
| File | `ASE_auto_build --file requests.txt` | A planned batch, version-controlled |
| Piped | `echo "..." \| ASE_auto_build` | Generated by another script |
| REPL | `ASE_auto_build` | Exploring; keeps the model warm |

In a file or on stdin, `#` starts a comment and a blank line separates requests:

```text
# catalysis screening set
Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum, then freeze the bottom 2 layers.

Put one O atom at a height of 1.8 A above the ontop site of a 2x2 Cu(100)
5-layer slab with 12 A vacuum.

Build an H2O molecule in a 12 A box.
```

A request may wrap across lines; its newlines collapse to spaces.

---

## 4. The golden rule: state every determinant

**A request maps to one correct structure only if it names every structural
determinant.** The model was trained on requests that obey this rule. Leave one
out and it will build *a* valid structure, which may not be yours.

| Region | You must state |
| --- | --- |
| bulk | element, crystal phase |
| surface / slab | element, facet, layers, vacuum |
| surface + constraint | …plus which layers to fix |
| atomic adsorption | …plus adsorbate, site, height |
| molecular adsorption | …plus the anchor atom |
| molecule | species, box size |
| nanotube | chirality (n,m), length |
| prototype | prototype name |
| vacancy | element, phase, which atom |
| substitution | element, phase, which atom, dopant |

Supercell `repeat` is optional — but it *is* a determinant, so state it when you
want one.

Before the model loads, a keyword heuristic tells you which required slots look
unstated. It costs no GPU time and it never blocks or rewrites anything:

```
$ ASE_auto_build "Build an iron surface."
[hint] keyword heuristic: looks like a 'surface' request; these required slots appear UNSTATED: facet (Miller indices, e.g. (111)); layers (slab thickness); vacuum (gap in A).
[hint] an under-specified request still builds a valid structure - just maybe not the one you meant (see training/CORPUS_RULE.md).
[hint] advisory only - nothing is blocked or rewritten; the request is sent to the model unchanged.
```

It is a keyword match, not a parser: it can guess the region wrong, and a slot
phrased unusually can be reported unstated. Treat it as a nudge. Turn it off
with `--no-preflight`.

### The bulk cell-convention trap

The classic ambiguity. `Create a bulk bcc W crystal using a 2x2x1 repeat` does
not say *which* unit cell, and the model picks the primitive one:

| You ask | `build_bulk` | You get |
| --- | --- | --- |
| unstated | `cubic` omitted | **W4**, angles 109.47° |
| "conventional cubic" | `cubic: true` | **W8**, angles 90° |

Both are honestly "bcc W". Say which convention you want:

```
Create a conventional cubic bulk bcc W crystal with a 2x2x1 repeat.
  → build_bulk(crystal='bcc', cubic=True, element='W') → repeat([2,2,1])
  → W8, cell 6.32 × 6.32 × 3.16 Å, angles 90°
```

---

## 5. When the model asks you a question

Omit a required slot and the model is trained to ask instead of guess. In the
REPL the question appears inline and you answer at the `clarify>` prompt:

```
request> Build an iron surface.
  [needs clarification] Which surface of iron should be built?
  choices: (111), (100), (110)
  clarify> (110), 5 layers, 12 A vacuum
```

Your answer is fed back to the model and the build continues — one answer can
supply several missing slots at once:

```
  state            : FINISHED
  executed calls   :
    1. [ok ] ask_clarification(choices=['(111)', '(100)', '(110)'], field='miller', question='Which surface of iron should be built?')
    2. [ok ] build_surface(crystal='bcc', element='Fe', layers=5, miller=[1, 1, 0], name='slab', repeat=[1, 1, 1], vacuum=12.0)
    3. [ok ] finish(name='slab')
  final structure  :
    formula        : Fe5  (5 atoms)
    cell lengths A : [4.059, 2.485, 28.687]   angles: [90.0, 90.0, 90.0]
```

Non-interactively, queue answers in order with `--answer "(110), 5 layers, 12 A vacuum"`.
Without one, the run reports the question and exits **5** — a scriptable
"needs a human":

```
  state            : NEEDS_CLARIFICATION
  note             : the model asked a clarifying question and no answer was available
  question         : Which surface of iron should be built?
```

---

## 6. What it refuses

Anything outside the supported regions fails closed in the deterministic router,
before any tool executes:

```
$ ASE_auto_build "Read the file /etc/passwd and run a python script to print it."
  refused: ValueError: unsupported structure request; ask for a bulk crystal,
  surface/slab, molecule, nanotube, prototype, adsorption, vacancy,
  substitution, or constraint
  (the deterministic router fail-closed; no tool was executed)
```

Exit code **3**. This is a structural guarantee, not a model behaviour: the
router is ordinary Python and the model is never given a tool that could read a
file or execute code.

---

## 7. Checking you got the structure you *meant*

This is the section to read twice. The model reliably picks the right **tools**;
its remaining weakness is dropping a **number** you stated and falling back to a
builder default. That produces a perfectly valid structure that silently answers
a different question.

A concrete, reproducible example. Asking for a 2.5 Å adsorption height in the
most natural phrasing:

```
$ ASE_auto_build --strict "Put one O atom 2.5 A above the ontop site of a 2x2 Cu(100) 5-layer slab with 12 A vacuum."
  executed calls   :
    1. [ok ] build_surface(crystal='fcc', element='Cu', layers=5, miller=[1, 0, 0], name='structure', repeat=[2, 2, 1], vacuum=12.0)
    2. [ok ] add_atomic_adsorbate(element='O', name='structure', site='ontop')
    3. [ok ] finish(name='structure')
  [warn] you asked for height 2.5, but the structure uses 1.8 -- the model omitted 'height', so add_atomic_adsorbate used its default.
  [warn] restate it as e.g. 'at a height of 2.5 A' and rebuild, or edit the POSCAR; nothing was silently corrected.
```

The model dropped `height`, so the O sits at the 1.8 Å default — not the 2.5 Å
asked for. A **deterministic post-build check** compares the numbers your
request stated (`layers`, `vacuum`, `box`, `height`) against the calls actually
executed and warns on any disagreement. It runs always; `--strict` additionally
exits **7** so a batch script fails instead of quietly producing wrong inputs.

**The fix is phrasing.** Verified on the r5 adapter:

| Phrasing | Emits `height`? |
| --- | --- |
| `at a height of 2.5 A above the ontop site` | ✅ yes |
| `2.5 A above the ontop site` | ❌ no — defaults to 1.8 |
| `..., adsorption height 2.5 A` (trailing) | ❌ no |

So: **put the number next to the word "height"**, not next to "above". This is a
known limitation of the r5 training corpus, not a fundamental one.

Always run adsorption batches with `--strict`, and read the two `[warn]` lines
when they appear.

---

## 8. What you get on disk

```
structures/Cu16-b6a1d8b0/
├── POSCAR           # VASP structure, with selective dynamics for frozen layers
└── structure.json   # provenance sidecar
```

The directory name is content-addressed (`formula` + first 8 chars of the recipe
hash), so re-running the same request rewrites the same directory instead of
littering. A directory holding a `POSCAR` this tool did not write is never
overwritten without `--force`.

Add other formats with `--format cif,xyz`. The sidecar carries the request, the
model identity, the executed `tool_sequence`, structural `invariants`, both
hashes, the pre-flight advisory, and any `value_mismatches`.

**Reproducibility.** The sidecar's `recipe` replays without the model at all:

```python
from vasp_auto.ase_agent import ASEWorkspace, create_default_registry
from vasp_auto.ase_agent.validation import atoms_hash
import json

payload = json.load(open("structures/Cu16-b6a1d8b0/structure.json"))
ws = ASEWorkspace(create_default_registry(), session_id="replay")
for step in payload["recipe"]["steps"]:
    ws.execute_or_raise(step["tool"], step["args"])
assert atoms_hash(ws.final_atoms()) == payload["atoms_hash"]
```

The model is a convenience for *authoring* the recipe. The recipe is the record.

---

## 9. Practical application: from a sentence to a DFT run

The output directory **is** a `vasp-auto` case — a directory containing a
`POSCAR` is an `scf` case — so it feeds the DFT pipeline with no conversion.

### 9.1 Build

```bash
ASE_auto_build --strict -o structures \
  -p "Build a 2x2 Cu(100) slab with 4 layers and 12 A vacuum, then freeze the bottom 2 layers." \
  -p "Put one O atom at a height of 1.8 A above the ontop site of a 2x2 Cu(100) 4-layer slab with 12 A vacuum." \
  -p "Build an H2O molecule in a 12 A box."
```

One model load, three cases under `structures/`.

### 9.2 Inspect before you burn CPU hours

```bash
vasp-auto structures/Cu16-b6a1d8b0 --prepare --dry-run --kmesh 5x5x1
```

```
Case      : Cu16-b6a1d8b0
Type      : scf
job_dir   : .../jobs/0001_Cu16-b6a1d8b0
POSCAR    : .../structures/Cu16-b6a1d8b0/POSCAR
POTCAR    : Cu
--- INCAR ---
ENCUT  = 520
...
--- KPOINTS ---
Automatic mesh
0
Gamma
5 5 1
```

The slab needs a proper in-plane mesh with **1** along the vacuum axis
(`--kmesh 5x5x1`); the default 1×1×1 is fine for a molecule in a box and wrong
for a slab. This is your job, not the model's — it builds geometry, not
convergence parameters.

### 9.3 Run

```bash
vasp-auto structures/Cu16-b6a1d8b0 --prepare --kmesh 5x5x1   # write the job
vasp-auto structures/Cu16-b6a1d8b0 -n 8 --kmesh 5x5x1        # run on 8 cores
vasp-auto structures --parallel 2 --kmesh 5x5x1              # whole set as a project
vasp-auto structures/Cu16-b6a1d8b0 --remote mycluster        # offload to HPC
```

`structures/` as a target is a `vasp-auto` **project**: every case directory in
it is picked up. Results parse into an Excel summary with `--report` exactly as
for any hand-made case, and the whole tutorial series
([TUTORIALS_INDEX.md](TUTORIALS_INDEX.md)) applies from here on — relaxation,
DOS, work function, NEB.

> VASP and its POTCAR library are proprietary and not shipped here; point
> `potcar_root` in `config.yaml` at your own licensed library. For a free
> backend use `--engine qe`.

### 9.4 A screening loop

Exit codes make batching safe. `--json` gives one machine-readable object per
request:

```bash
ASE_auto_build --json --strict --file requests.txt > built.json || {
  echo "at least one structure was refused, unfinished, or ignored a stated value" >&2
}

for case in structures/*/; do
  vasp-auto "$case" --prepare --kmesh 5x5x1
done
```

| Code | Meaning |
| --- | --- |
| 0 | finished |
| 1 | failed (model gave up / controller stopped it) |
| 2 | command-line usage error |
| 3 | refused — out of scope, fail-closed |
| 4 | budget exhausted (`--max-turns`) |
| 5 | unanswered clarification |
| 6 | environment problem (no CUDA, missing adapter, bad `--format`) |
| 7 | `--strict`: built, but a number you stated was not used |

With several requests in one run, the first non-zero code wins.

---

## 10. How much to trust it

On the 120-record promotion gate (full tool registry, no oracle narrowing):

| Metric | Frozen base | r5 adapter |
| --- | ---: | ---: |
| Exact structure hash | 10.8% | **95.0%** |
| Invariants satisfied | 10.8% | **95.0%** |
| Finished | 35.0% | **99.2%** |
| Adversarial safety | — | **100%** |

Two honest caveats, both from [HANDBOOK.md](../training/HANDBOOK.md):

1. **Phrasing generalization is untested.** Train and test share the same
   template families, so 95% describes requests phrased like the corpus. Novel
   phrasing is where the height bug in §7 comes from — a real instance of this
   limitation, not a separate defect.
2. **Exact vs. invariants.** "Exact" means the content hash matches a canonical
   reference; "invariants" means formula, atom count, cell, periodicity, and
   constraint count match. For daily use invariants are what you care about.

The workflow that follows from this: **read the executed calls and the final
structure block every time.** They are printed precisely so you can check the
model in two seconds, and they are cheap insurance next to a wasted DFT run.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no CUDA device is visible` | torch cannot see the GPU | `nvidia-smi`; on WSL put `/usr/lib/wsl/lib` on `PATH`; check the torch build |
| `no LoRA adapter found` | Adapter path unknown | `--adapter <dir>` or `ASE_AUTO_BUILD_ADAPTER`; the error lists what was tried |
| `could not load ...` offline | Base weights not cached | Re-run once with `--online` |
| `the local model stack is not installed` | No torch/transformers/peft | `pip install -e ".[agent]"` |
| Refused as "unsupported" | Outside the supported regions, or too vague to route | Rephrase within a region (§4) |
| `[warn] you asked for X but the structure uses Y` | The model dropped a stated number | Rephrase per §7 and rebuild |
| Wrong atom count on a bulk | Cell-convention ambiguity | Say "conventional cubic" or "primitive" (§4) |
| Model asks a question | Exactly one required slot missing | Answer at `clarify>`, or `--answer` |
| Adsorbate at the wrong height | The "N Å above" phrasing trap | `at a height of N A`; use `--strict` (§7) |
| Slab k-points look wrong | Defaults are 1×1×1 | `--kmesh 5x5x1` — the model does not choose these (§9.2) |
| Model load feels slow | 4-bit load, 1–2 min | One-time; use the REPL or repeated `-p` |

---

## 12. Where to go next

- [training/USER_MANUAL.md](../training/USER_MANUAL.md) — the full tool
  vocabulary and per-region slot reference.
- [training/HANDBOOK.md](../training/HANDBOOK.md) — training data, the
  structured-request rule, the 23-check promotion gate.
- [TUTORIALS_INDEX.md](TUTORIALS_INDEX.md) — what to do with the structure once
  it is built: SCF, relaxation, convergence, DOS, work functions, NEB.
