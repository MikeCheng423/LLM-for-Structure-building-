"""Local 4-bit model loading and one-tool-call-per-turn chat wrapper.

This is the single source of truth for *how the fine-tuned adapter is driven*:
quantization, greedy decoding, and the Qwen ``<tool_call>`` parsing that turns a
generated turn into the controller's ``tool_calls``. Both the promotion gate
(``training/evaluations/evaluate_model.py``) and the user entry point
(``vasp_auto.ase_agent.cli``) import from here, so the numbers measured during
promotion describe what a user actually gets.

Importing this module pulls in torch/transformers/peft. Keep it out of
``vasp_auto.ase_agent.__init__`` and import it lazily, so the GPU-free parts of
the runtime (workspace, controller, CLI argument handling) stay importable on a
machine with no deep-learning stack installed.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Re-exported: the evaluation harness has always imported these from its own
# namespace, and they are pure-ASE/pure-data, so they live in torch-free modules.
from .llm_defaults import DEFAULT_MODEL, DEFAULT_REVISION  # noqa: F401
from .validation import _constrained_count, structure_invariants  # noqa: F401


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for number, match in enumerate(TOOL_CALL_RE.finditer(text), start=1):
        value = json.loads(match.group(1))
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("tool call must contain a string name")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be an object")
        calls.append({
            "id": f"generated_{number}",
            "type": "function",
            "function": {"name": value["name"], "arguments": arguments},
        })
    return calls


def first_tool_call_turn(text: str) -> str:
    """Enforce the controller's one sequential tool call per model turn."""
    match = TOOL_CALL_RE.search(text)
    return text[: match.end()] if match is not None else text


class LocalModelChat:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        tool_override: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tool_override = tool_override
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.generated_texts: list[str] = []

    def __call__(self, messages, tools):
        selected_tools = self.tool_override if self.tool_override is not None else tools
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=selected_tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        self.generated_tokens += int(generated.shape[0])
        text = first_tool_call_turn(
            self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        self.generated_texts.append(text)
        try:
            calls = parse_tool_calls(text)
        except Exception:
            calls = []
        return {"role": "assistant", "content": text, "tool_calls": calls}


def load_model(
    *,
    model: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    cache_dir=None,
    adapter=None,
):
    """Load the frozen 4-bit base model, optionally with a LoRA adapter on top."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded = AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        cache_dir=cache_dir,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if adapter:
        loaded = PeftModel.from_pretrained(loaded, adapter)
    loaded.eval()
    return loaded, tokenizer


def _load_model(args):
    """Argparse-namespace shim kept for the evaluation harness's call site."""
    return load_model(
        model=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        adapter=args.adapter,
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "LocalModelChat",
    "TOOL_CALL_RE",
    "first_tool_call_turn",
    "load_model",
    "parse_tool_calls",
    "structure_invariants",
]
