# Structure_building workspace rules

## Repository

- Git remote: `git@github.com:MikeCheng423/LLM-for-Structure-building-.git`

## Isolation boundary

All implementation, training, evaluation, testing, generated data, configuration,
and documentation changes for this project must stay inside:

```text
/home/vv/Structure_building
```

The original project at `/home/vv/vasp_auto` is read-only reference material.
Do not edit, patch, format, delete, copy generated files into, or otherwise
mutate anything under `/home/vv/vasp_auto` while working on this project.

Do not install this workspace by modifying the original environment or original
checkout. Prefer a virtual environment inside `/home/vv/Structure_building`.
An existing external interpreter may be used to run tests only when imports are
confirmed to resolve from this workspace's `src/` directory.

Integration back into `vasp_auto` is outside the scope of training work. It may
only happen after the user explicitly requests integration and approves the
reviewed diff or patch in a separate task.

## Project layout

- Runtime implementation: `src/ase_auto_build/ase_agent/`
- Runtime integration adapters: copied modules under `src/ase_auto_build/` and
  `src/vasp_auto_ui/`
- Tests: `tests/`
- Training/evaluation utilities and data: `training/`
- Architecture and milestones: `plan.md`
- Snapshot provenance and setup: `SNAPSHOT.md`

## Safety

- Never place API keys, remote credentials, proprietary POTCAR files, private
  structures, or unredacted user transcripts in training data.
- Do not give the model arbitrary Python, shell, filesystem, network, or
  calculator-constructor execution.
- Keep expensive calculations behind explicit confirmation and existing job
  dispatch boundaries.
- Preserve deterministic recipes and validation for every accepted trajectory.
