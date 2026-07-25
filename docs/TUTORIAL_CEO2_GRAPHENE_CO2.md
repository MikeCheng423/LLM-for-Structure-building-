# Tutorial — CO₂ adsorption on a CeO₂/graphene heterostructure (in the Web UI)

This walkthrough builds a **CeO₂ crystal supported on a graphene sheet**, puts a
**CO₂ molecule** on the ceria surface, and computes the **CO₂ adsorption energy**
— entirely from the `vasp-auto-ui` web interface, starting from an empty editor.

It uses three engine capabilities you'll click through in the UI:

- the **space-group crystal** + **surface slab** builders (ASE) for CeO₂,
- the **prototype** builder + **cell match** + **combine/stack** for the
  CeO₂-on-graphene interface,
- the **molecule builder** + **combine/insert** to drop CO₂ onto the surface,
- the **Adsorption energy** analysis card for the final number.

> **The quantity we're after**
>
> ```
> E_ads = E(CeO2/graphene + CO2) − E(CeO2/graphene) − E(CO2, gas)
> ```
>
> A **negative** `E_ads` means CO₂ binds (exothermic adsorption). You therefore
> run **three** calculations — the bare support, the support+CO₂, and an
> isolated CO₂ molecule — and the UI subtracts them for you.

---

## 0. Start the UI

From the repo root:

```bash
vasp-auto-ui            # serves http://127.0.0.1:8800 (localhost only)
```

Open the page. You'll see four tabs: **Build · Calculate · Workflow · Results**.
Everything in Part A happens on the **Build** tab, which has a searchable
*Build function* catalog on the left — clicking a tile reveals **one** builder
form at a time and loads its result into the central **3D editor**. Builders
write *nothing* to disk until you press the **💾 Save** button, which turns the
editor's current structure into a *case* under `inputs/`.

Keep this rule in mind for every combine step:

> **Host = whatever structure is currently in the editor. Guest = a saved case
> you point the form at.** So you "load the host into the editor, then pick the
> guest."

---

## Part A — Build the structures

### A1. CeO₂ bulk (fluorite, space group 225)

CeO₂ is cubic fluorite (`Fm-3m`, a ≈ 5.41 Å) with Ce at `(0,0,0)` and O at
`(¼,¼,¼)`.

1. Build tab → search **"crystal"** → open **🔷 Space-group crystal**.
2. Fill in:
   - **symbols**: `Ce O`
   - **basis** (one representative site per symbol):
     `0 0 0` for Ce, `0.25 0.25 0.25` for O
   - **spacegroup**: `225`
   - **a**: `5.411` (b, c blank → cubic; angles 90°)
3. Click build → the fluorite cell appears in the editor.
4. **💾 Save** as `CeO2_bulk`. (You now have `inputs/CeO2_bulk/POSCAR`.)

### A2. Cleave the CeO₂(111) surface

The (111) facet is the most stable, charge-neutral CeO₂ surface and the natural
one to support on graphene.

1. Build tab → search **"slab"** → open **🪨 Surface slab**.
2. Set:
   - **source**: the path to the file you just saved —
     `inputs/CeO2_bulk/POSCAR` (use the 📁 browse button). The slab builder reads
     a structure file *or* an element symbol; here we feed it the ceria cell.
   - **miller**: `1 1 1`
   - **layers**: `3` (a thin film — enough for a demo, raise later)
   - **vacuum**: `12` (Å above the slab)
3. Build → a CeO₂(111) slab loads into the editor.
4. **💾 Save** as `CeO2_111`.

> Keep this slab **small** in-plane for now (don't repeat it). The next step
> finds a graphene supercell that matches *this* cell.

### A3. Graphene support

1. Build tab → search **"prototype"** → open **💠 Prototype crystal**.
2. Choose **graphene** (a ≈ 2.468 Å, vacuum along c). Build.
3. **💾 Save** as `graphene`.

### A4. Match cells and stack CeO₂(111) on graphene

Graphene and CeO₂(111) have different in-plane lattices, so you need a
**commensurate supercell pair** that minimises strain before stacking.

1. Load the **host** into the editor: in the Build catalog open **📥 Import a
   file** (or the editor's open control) and load `inputs/graphene/POSCAR`, *or*
   simply re-build graphene so it's the live structure. Graphene is the host
   (it keeps its flat in-plane vectors; the ceria slab rides on top).
2. Build tab → search **"combine"** → open **🧬 Combine structures**.
3. Set **guest case** (or guest path) to `inputs/CeO2_111`.
4. Click **🔍 suggest cell match**. The panel lists host×guest supercell pairs
   sorted by strain (e.g. *graphene 4×4 under CeO₂ 1×1*, with the % strain and
   angle mismatch). Pick a row with **strain ≲ 5 %** — clicking it fills in the
   host/guest repeats.
5. Combine settings:
   - **mode**: `stack on top`
   - **gap**: `2.5` Å (ceria–graphene separation; this is a van-der-Waals-ish
     contact, refine during relaxation)
   - **vacuum**: `15` Å left above
   - **strain guest to host lattice**: ✅ **on** (epitaxial match — the ceria
     slab is strained onto the chosen graphene supercell)
6. **Combine → editor**. Inspect: a CeO₂(111) film sitting `gap` Å above a
   graphene sheet, with vacuum on top.
7. **💾 Save** as `CeO2_graphene`. **This is your bare-support structure
   (the "slab" reference).**

> **Tip — freeze the support, free the surface.** Optionally, in the editor
> select the graphene atoms and the bottom ceria layer and mark them fixed
> (selective dynamics `F F F`) so only the top of the ceria relaxes. This makes
> all three jobs cheaper and keeps the support geometry consistent between the
> "slab" and "slab+CO₂" runs.

### A5. CO₂ molecule reference

1. Build tab → search **"molecule"** → open **💧 Molecule in a box**.
2. **name**: `CO2` (ASE's molecule database; linear O=C=O), **box**: `12` Å.
3. Build → **💾 Save** as `CO2`. **This is the gas-phase reference.**

### A6. Put CO₂ on the ceria surface

The **🧲 Adsorption quick build** card only places a *single* atom over a surface
atom — fine for atomic O or H, but CO₂ is a 3-atom molecule. So we instead
**insert** the saved CO₂ structure into the support cell:

1. Load **`CeO2_graphene`** into the editor (Import → `inputs/CeO2_graphene/POSCAR`)
   so it is the host.
2. (Optional, to aim the molecule) click a surface **Ce** atom in the editor and
   read off its x,y — Ce sites are the Lewis-acidic adsorption centres for CO₂.
3. Build tab → **🧬 Combine structures**:
   - **guest case**: `inputs/CO2`
   - **mode**: `insert into cell` (keeps the support cell unchanged and drops
     CO₂ in)
   - **gap**: `2.2` Å (height of CO₂ above the surface top atom)
   - **shift x,y**: fractions of the support a/b that move CO₂ over your chosen
     Ce site (e.g. `0.5,0.5` to centre it; tune to sit above Ce).
   - leave **strain** *off* (the molecule must keep its own geometry).
4. **Combine → editor**. You should see CO₂ hovering above the ceria surface.
   Nudge it with the arrow keys / drag if needed so the C or an O points at Ce.
5. **💾 Save** as `CeO2_graphene_CO2`. **This is your slab+adsorbate structure.**

You now have three cases:

| Case | Role in `E_ads` |
|------|-----------------|
| `inputs/CeO2_graphene`      | slab (bare support)        |
| `inputs/CeO2_graphene_CO2`  | slab + adsorbate (total)   |
| `inputs/CO2`                | gas-phase molecule         |

---

## Part B — Run the three calculations

Go to the **Calculate** tab. Run each case as a **structure optimisation**
(`relax`) so the geometry settles, then the energies are comparable.

For **each** of the three cases:

1. Pick the case (`CeO2_graphene`, then `CeO2_graphene_CO2`, then `CO2`).
2. **Calculation type**: `relax`.
3. K-points: for the two **slab** cases use a Γ-centred in-plane mesh
   (e.g. spacing ≈ 0.25 Å⁻¹, **1** k-point along the vacuum direction). For the
   **CO2** box, **Γ-only** is correct (isolated molecule).
4. Run. Logs stream into `ui_logs/`; results land under `jobs/`.

> **Physics caveats worth setting before production runs** (edit the case INCAR
> in the editor's POSCAR/INCAR side panel, or drop a `config.yaml`/INCAR in the
> case folder):
>
> - **DFT+U on cerium.** Ceria's Ce-4f states are badly described by plain GGA.
>   For meaningful energetics add Hubbard U, e.g. `LDAU = .TRUE.`,
>   `LDAUTYPE = 2`, `LDAUL` = `3 -1 -1` (f on Ce, none on O/C), `LDAUU` =
>   `4.5 0 0`, `LDAUJ = 0 0 0`, `LMAXMIX = 6`. Order the values to match the
>   POTCAR element order (Ce, O, C).
> - **Dipole correction.** An adsorbate on one face breaks z-symmetry — set
>   `LDIPOL = .TRUE.` and `IDIPOL = 3` for the slab cases.
> - **Spin.** Reduced Ce³⁺ can be magnetic; enable spin (`--spin`/`ISPIN=2`) if
>   you expect charge transfer to the f-states.
> - **Converge first.** Consider a `converge → relax` workflow (Workflow tab)
>   so ENCUT/k-points are validated before the production relaxations.

Two Calculate-tab conveniences for this part:

- **Reuse an INCAR across the three cases.** Once one case's INCAR carries the
  DFT+U / dipole settings above, switch to the next case and press **Load from
  another work…** in the INCAR editor — pick the first case's folder (or its
  finished job) and its INCAR is copied into the editor; *Save to case* applies
  it.
- **A relaxation stopped early?** (hit `NSW`, was terminated, or the machine
  went down) — press **Resume**. It asks for the CPU count (blank = same as
  before), then opens that job's INCAR as an editable spreadsheet so you can
  raise `NSW` or adjust tags before it restarts in place from the newest
  CONTCAR. The Results tab's checkboxes offer the same as a bulk action.

Wait for all three to finish (the **Results** tab shows convergence status).

---

## Part C — Compute the adsorption energy

1. Go to the card **🧗 Adsorption energy** (Results/Analysis area).
2. Fill the three finished **job** directories (use 📁 to pick from `jobs/`):
   - **slab + adsorbate job** → the `CeO2_graphene_CO2` job
   - **slab job** → the `CeO2_graphene` job
   - **molecule job** → the `CO2` job
   - **scale** → `1` (one whole CO₂ molecule adsorbs)
3. **Compute E_ads**.

The card prints `E(slab+ads)`, `E(slab)`, `E(molecule) × scale`, and

```
E_ads = E(slab+ads) − E(slab) − 1·E(CO2)     [eV]
```

A negative value = CO₂ is bound to the CeO₂/graphene surface. You can re-run
Part A6 with CO₂ over different sites (top-Ce, bridge, O-vacancy) and compare
`E_ads` to find the preferred binding mode.

---

## Appendix — the same thing from the CLI

Every UI action maps to a `vasp-auto` flag, so the build is fully scriptable:

```bash
# A1  CeO2 fluorite bulk
vasp-auto --ase-build-crystal --ase-spacegroup 225 \
          --ase-basis "Ce 0 0 0; O 0.25 0.25 0.25" --ase-a 5.411 \
          --ase-output CeO2_bulk --ase-only

# A2  CeO2(111) slab from that file
vasp-auto --ase-build-slab inputs/CeO2_bulk/POSCAR --ase-miller 1 1 1 \
          --ase-layers 3 --ase-vacuum 12 --ase-output CeO2_111 --ase-only

# A3  graphene support
vasp-auto --build-prototype graphene --ase-output graphene --ase-only

# A4  find a commensurate supercell pair (graphene host, CeO2 guest)
vasp-auto --target inputs/graphene --match-cells inputs/CeO2_111 \
          --match-strain 0.05
# A4  stack the ceria slab on graphene (use repeats from the match output;
#     here illustrative --supercell on the target if a repeat is suggested)
vasp-auto --target inputs/graphene --combine inputs/CeO2_111 \
          --combine-mode stack --combine-gap 2.5 --combine-vacuum 15 \
          --combine-strain --ase-only      # -> inputs/graphene_plus_CeO2_111

# A5  CO2 reference molecule
vasp-auto --ase-build-molecule CO2 --ase-box 12 --ase-output CO2 --ase-only

# A6  insert CO2 onto the support (host = the heterostructure case)
vasp-auto --target inputs/graphene_plus_CeO2_111 --combine inputs/CO2 \
          --combine-mode insert --combine-gap 2.2 --combine-shift "0.5,0.5" \
          --ase-only

# B   run the three relaxations
vasp-auto --target inputs/graphene_plus_CeO2_111      --calc-type relax
vasp-auto --target inputs/graphene_plus_CeO2_111_plus_CO2 --calc-type relax
vasp-auto --target inputs/CO2                         --calc-type relax

# C   adsorption energy from the three finished jobs
vasp-auto --adsorption-energy \
          jobs/<total_job> jobs/<slab_job> jobs/<co2_job> \
          --molecule-scale 1
```

(Case/job folder names depend on your `inputs/`/`jobs_root`; the `--match-cells`
output tells you the exact host/guest repeats to use for a low-strain stack.)

---

## Recap

1. **Build** CeO₂ bulk (space-group 225) → cleave **(111) slab**.
2. **Stack** it on **graphene** using **cell match** + **combine/stack** → the
   support, `CeO2_graphene`.
3. **Insert** a box-built **CO₂** onto the surface → `CeO2_graphene_CO2`.
4. **Relax** all three (support, support+CO₂, CO₂) — with DFT+U on Ce and a
   dipole correction for real numbers.
5. **Adsorption energy** card subtracts them: `E_ads < 0` ⇒ CO₂ binds.

See also `docs/TUTORIAL_HETEROSTRUCTURE.md` (mismatched-cell strategies) and
`docs/TUTORIAL_CATALYSIS.md` (free energies, d-band, Bader, work function) for
deeper analysis once the geometry is in place.
