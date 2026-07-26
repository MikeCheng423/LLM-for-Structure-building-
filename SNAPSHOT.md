# Isolated development snapshot

This directory is an isolated working copy for developing and training the
LLM-controlled ASE structure builder described in `plan.md`.

## Isolation rule

Treat `/home/vv/ase_auto_build` as read-only reference material. All new agent code,
training utilities, datasets, experiments, tests, generated artifacts, and
configuration belong under `/home/vv/Structure_building`.

When the design is complete and validated here, integration back into
`ase_auto_build` should happen only as a separate, explicit task after reviewing a
diff or patch. Training must never write into the original repository.

## Snapshot source

- Source: `/home/vv/ase_auto_build`
- Snapshot date: 2026-07-24
- Package source: `src/ase_auto_build`, `src/vasp_auto_ui`
- Supporting material: tests, docs, examples, packaging metadata, README, and
  license

The snapshot intentionally has no Git history. It is a development seed, not a
second authoritative upstream repository.

## Excluded on purpose

- `.git` and repository metadata
- `__pycache__`, `.pyc`, pytest/build caches, and graph visualizer output
- `jobs`, `inputs`, calculation results, and temporary UI logs
- POTCAR/pseudopotential libraries
- `config.yaml`, `remotes.json`, and other machine-specific or potentially
  sensitive runtime configuration
- virtual environments and installed dependencies

Only example configuration files are copied. Create local experimental settings
inside this directory and never reuse secrets in training records.

## Starting the isolated environment

From `/home/vv/Structure_building`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The copied package still imports as `ase_auto_build` so its existing relative imports
and tests work. Run commands from this directory or its virtual environment to
avoid accidentally importing the original checkout.

## Development boundary

New implementation should begin under:

```text
src/ase_auto_build/ase_agent/
```

Tests should be added under:

```text
tests/test_ase_agent_*.py
```

Training/evaluation code and data should stay separate from runtime code:

```text
training/
    generators/
    schemas/
    datasets/       # ignored if it contains generated/private data
    evaluations/
```

Do not add training frameworks or large model dependencies to the core
`ase_auto_build` runtime dependency list. Keep them in an isolated training extra or
training-specific environment.
