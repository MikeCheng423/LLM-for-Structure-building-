"""Base-model identity for the promoted adapter, importable without torch.

Kept in its own module so the CLI can print defaults in ``--help`` and tests can
assert them on a machine with no deep-learning stack. ``llm_local`` re-exports
these; ``tests/test_ase_agent_cli.py`` asserts they still match
``training/train_qlora.py``, which is the training-side authority.
"""

from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
