#!/usr/bin/env python3
"""Interactive entry point to the fine-tuned ASE-agent LLM.

Loads the frozen 4-bit base model plus a LoRA adapter once, then drives the
bounded sequential tool-call controller against natural-language
structure-building requests. Use it as a REPL (default) or one-shot with
``--prompt``.

This is the real deployment path: the model must select from the *full*
runtime tool registry (no oracle narrowing), the deterministic workspace
executes every emitted tool call, and unroutable or forbidden requests fail
closed -- the bounded router raises before any tool runs.

Reuses the exact model loader and chat wrapper from the evaluation harness so
quantization, decoding, and one-tool-call-per-turn parsing match how the
adapter was gated.

Example (one-shot):

    PYTHONPATH=src:. PATH=/usr/lib/wsl/lib:$PATH \\
      HF_HOME=training/cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 \\
      .venv/bin/python training/chat_agent.py \\
      --prompt "Build a 4-layer 2x2 Cu(111) slab with 12 A vacuum"

Omit ``--prompt`` for an interactive REPL. Pass ``--base-only`` to talk to the
frozen base model without the adapter (useful for an A/B feel).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from vasp_auto.ase_agent import (
    AgentController,
    AgentPolicy,
    ASEWorkspace,
    ControllerState,
    create_default_registry,
)
from vasp_auto.ase_agent.validation import atoms_hash

# Reuse the gated harness so decoding/parsing/quantization are identical.
from training.evaluations.evaluate_model import (
    LocalModelChat,
    _load_model,
    structure_invariants,
)
from training.train_qlora import DEFAULT_MODEL, DEFAULT_REVISION

DEFAULT_ADAPTER = Path("training/runs/pilot-qwen3-4b-r5/adapter")


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _fmt_formula(formula: dict) -> str:
    return "".join(f"{element}{count}" for element, count in formula.items())


def _print_trajectory(result, workspace, chat) -> None:
    print()
    print(f"  controller state : {result.state.value}")
    if result.error:
        print(f"  controller note  : {result.error}")

    for turn, text in enumerate(chat.generated_texts, start=1):
        snippet = " ".join(text.split())
        if len(snippet) > 220:
            snippet = snippet[:220] + " ..."
        print(f"  model turn {turn:>2}    : {snippet or '(empty)'}")

    if result.transcript:
        print("  executed calls   :")
        for number, item in enumerate(result.transcript, start=1):
            status = "ok " if item["success"] else "ERR"
            print(f"    {number}. [{status}] {item['tool']}({_fmt_args(item['args'])})")
            if not item["success"]:
                print(f"          -> {item['result']}")
    else:
        print("  executed calls   : (none)")

    if result.state is ControllerState.FINISHED:
        atoms = workspace.final_atoms()
        inv = structure_invariants(atoms)
        print("  final structure  :")
        print(f"    formula        : {_fmt_formula(inv['formula'])}  ({inv['natoms']} atoms)")
        print(f"    cell lengths A : {inv['cell_lengths']}   angles: {inv['cell_angles']}")
        print(f"    pbc            : {inv['pbc']}   constrained atoms: {inv['constrained_atoms']}")
        print(f"    atoms_hash     : {atoms_hash(atoms)}")
    print()


def run_request(model, tokenizer, prompt, *, max_turns, max_new_tokens, clarify) -> None:
    """Drive one request through a fresh workspace/controller and print the result."""
    registry = create_default_registry()
    policy = AgentPolicy(max_model_turns=max_turns, max_tool_calls=max_turns)
    workspace = ASEWorkspace(registry, policy=policy, session_id="chat-agent")
    chat = LocalModelChat(model, tokenizer, tool_override=None, max_new_tokens=max_new_tokens)
    controller = AgentController(workspace, chat)
    try:
        result = controller.run(prompt)
        while result.state is ControllerState.NEEDS_CLARIFICATION:
            follow_up = clarify(result)
            if follow_up is None:
                break
            result = controller.resume(follow_up)
    except Exception as exc:  # noqa: BLE001 - a fail-closed refusal is expected, not a crash
        print()
        print(f"  request refused before execution: {type(exc).__name__}: {exc}")
        print()
        return
    _print_trajectory(result, workspace, chat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive ASE-agent LLM entry point")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-only", action="store_true",
                        help="Load the frozen base model without the LoRA adapter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", help="One-shot request; omit for an interactive REPL")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    load_args = SimpleNamespace(
        model=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        adapter=None if args.base_only else args.adapter,
    )
    label = "base (no adapter)" if args.base_only else str(args.adapter)
    print(f"Loading {args.model} @ {args.revision[:8]} ... adapter: {label}")
    model, tokenizer = _load_model(load_args)
    print("Model ready.")

    def interactive_clarify(result):
        print(f"  [needs clarification] {result.error or ''}")
        try:
            return input("  clarify> ").strip() or None
        except EOFError:
            return None

    def oneshot_clarify(result):
        print(f"  [model asked for clarification; one-shot mode cannot answer] {result.error or ''}")
        return None

    if args.prompt:
        run_request(model, tokenizer, args.prompt,
                    max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                    clarify=oneshot_clarify)
        return 0

    print("Interactive ASE-agent. Enter a structure-building request; 'quit' or Ctrl-D to exit.")
    while True:
        try:
            prompt = input("\nrequest> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            break
        run_request(model, tokenizer, prompt,
                    max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                    clarify=interactive_clarify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
