#!/usr/bin/env python3
"""QLoRA SFT for validated ASE-agent tool trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import bitsandbytes
import datasets
import peft
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from training.dataset_contract import load_journal_jsonl, load_jsonl, render_assistant_only
from training.evaluations.evaluate_corpus import evaluate_dataset
from training.evaluations.evaluate_journal_corpus import evaluate_dataset as evaluate_journal_dataset


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def adapter_integrity(path: Path) -> dict[str, int | str | float]:
    """Fail closed if the saved adapter is missing or contains untouched LoRA-B weights."""
    if not path.is_file():
        raise RuntimeError(f"saved adapter tensor file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    tensor_count = 0
    value_count = 0
    nonzero_count = 0
    maximum = 0.0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if "lora_B" not in name:
                continue
            values = tensors.get_tensor(name)
            tensor_count += 1
            value_count += values.numel()
            nonzero_count += int((values != 0).sum())
            maximum = max(maximum, float(values.abs().max()))
    if tensor_count == 0 or value_count == 0 or nonzero_count != value_count:
        raise RuntimeError("saved adapter contains missing or zero LoRA-B weights")
    return {
        "file": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "lora_b_tensors": tensor_count,
        "lora_b_values": value_count,
        "lora_b_nonzero_values": nonzero_count,
        "lora_b_max_abs": maximum,
    }


def initial_adapter_metadata(
    adapter_dir: Path,
    *,
    model: str,
    revision: str,
    lora_rank: int,
    require_promoted: bool,
) -> dict[str, object]:
    """Validate a warm-start adapter and return immutable provenance."""
    adapter_dir = adapter_dir.resolve()
    config_path = adapter_dir / "adapter_config.json"
    manifest_path = adapter_dir.parent / "manifest.json"
    tensor_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("initial adapter requires adapter_config.json and parent manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != model or manifest.get("base_model") != model:
        raise RuntimeError("initial adapter base model does not match --model")
    if manifest.get("base_revision") != revision:
        raise RuntimeError("initial adapter base revision does not match --revision")
    if config.get("r") != lora_rank or manifest.get("lora_rank") != lora_rank:
        raise RuntimeError("initial adapter rank does not match --lora-rank")
    if require_promoted and manifest.get("production_ready") is not True:
        raise RuntimeError("production warm-start requires a promoted initial adapter")
    integrity = adapter_integrity(tensor_path)
    expected_sha256 = manifest.get("adapter_integrity", {}).get("sha256")
    if integrity["sha256"] != expected_sha256:
        raise RuntimeError("initial adapter hash does not match its manifest")
    return {
        "path": str(adapter_dir),
        "source_manifest": str(manifest_path),
        "production_ready": manifest.get("production_ready") is True,
        "integrity": integrity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("training/cache/huggingface"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Continue training a compatible, promoted LoRA adapter.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-records", type=int, default=1_000)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument(
        "--record-id",
        help="Train one named record for memory smoke testing; requires --allow-smoke-data.",
    )
    parser.add_argument(
        "--allow-smoke-data",
        action="store_true",
        help="Accept explicitly marked synthetic smoke records; output remains non-production.",
    )
    parser.add_argument(
        "--corpus-contract", choices=("ase", "journal"), default="ase",
        help="Select the fail-closed corpus auditor; defaults to ASE trajectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this QLoRA configuration")
    if not args.allow_smoke_data and args.eval_dataset is None:
        raise SystemExit("production training requires --eval-dataset")

    initial_adapter = None
    if args.initial_adapter is not None:
        try:
            initial_adapter = initial_adapter_metadata(
                args.initial_adapter,
                model=args.model,
                revision=args.revision,
                lora_rank=args.lora_rank,
                require_promoted=not args.allow_smoke_data,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"invalid initial adapter: {exc}") from exc

    corpus_audit = None
    if not args.allow_smoke_data:
        assert args.eval_dataset is not None
        if args.dataset.parent.resolve() != args.eval_dataset.parent.resolve():
            raise SystemExit("training and evaluation splits must share one corpus directory")
        if args.dataset.name != "train.jsonl" or args.eval_dataset.name != "validation.jsonl":
            raise SystemExit("production training requires train.jsonl and validation.jsonl")
        try:
            audit = evaluate_journal_dataset if args.corpus_contract == "journal" else evaluate_dataset
            corpus_audit = audit(args.dataset.parent)
        except Exception as exc:
            raise SystemExit(f"production corpus replay failed: {exc}") from exc
        if (
            corpus_audit.get("passed") is not True
            or corpus_audit.get("execution_success_rate") != 1.0
            or corpus_audit.get("forbidden_action_rate") != 0.0
        ):
            raise SystemExit("production corpus did not pass replay and safety gates")

    if args.corpus_contract == "journal":
        records, dataset_sha256 = load_journal_jsonl(args.dataset)
    else:
        records, dataset_sha256 = load_jsonl(args.dataset, allow_smoke=args.allow_smoke_data)
    if args.record_id:
        if not args.allow_smoke_data:
            raise SystemExit("--record-id is restricted to explicit smoke runs")
        records = [record for record in records if record["id"] == args.record_id]
        if len(records) != 1:
            raise SystemExit(f"record id not found or not unique: {args.record_id}")
    registry_versions = {record["validation"]["registry_version"] for record in records}
    if len(registry_versions) != 1:
        raise SystemExit("training records contain mixed registry versions")
    dataset_registry_version = next(iter(registry_versions))
    if not args.allow_smoke_data and len(records) < args.minimum_records:
        raise SystemExit(
            f"production dataset has {len(records)} records; minimum is {args.minimum_records}"
        )
    eval_records = None
    eval_dataset_sha256 = None
    if args.eval_dataset is not None:
        if args.corpus_contract == "journal":
            eval_records, eval_dataset_sha256 = load_journal_jsonl(args.eval_dataset)
        else:
            eval_records, eval_dataset_sha256 = load_jsonl(
                args.eval_dataset, allow_smoke=args.allow_smoke_data,
            )
        eval_registry_versions = {
            record["validation"]["registry_version"] for record in eval_records
        }
        if eval_registry_versions != {dataset_registry_version}:
            raise SystemExit("evaluation records do not match the training registry version")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rendered = [
        render_assistant_only(tokenizer, record, max_length=args.max_length)
        for record in records
    ]
    train_dataset = Dataset.from_list(rendered)
    eval_dataset = None
    if eval_records is not None:
        eval_dataset = Dataset.from_list([
            render_assistant_only(tokenizer, record, max_length=args.max_length)
            for record in eval_records
        ])

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.initial_adapter is not None:
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant" if args.allow_smoke_data else "cosine",
        warmup_ratio=0.0 if args.allow_smoke_data else 0.03,
        logging_steps=1,
        eval_strategy="no" if eval_dataset is None else "steps",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="no" if args.allow_smoke_data else "steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not args.allow_smoke_data,
        metric_for_best_model="eval_loss" if not args.allow_smoke_data else None,
        greater_is_better=False if not args.allow_smoke_data else None,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        tf32=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")
    saved_adapter_integrity = adapter_integrity(
        args.output_dir / "adapter" / "adapter_model.safetensors"
    )

    manifest = {
        "base_model": args.model,
        "base_revision": args.revision,
        "corpus_contract": args.corpus_contract,
        "dataset_registry_version": dataset_registry_version,
        "dataset_sha256": dataset_sha256,
        "eval_dataset_sha256": eval_dataset_sha256,
        "record_count": len(records),
        "eval_record_count": len(eval_records or []),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "lora_rank": args.lora_rank,
        "initial_adapter": initial_adapter,
        "lr_scheduler": "constant" if args.allow_smoke_data else "cosine",
        "seed": args.seed,
        "smoke_run": args.allow_smoke_data,
        "training_data_gate_passed": not args.allow_smoke_data,
        "corpus_audit": None if corpus_audit is None else {
            "passed": corpus_audit["passed"],
            "record_count": corpus_audit["record_count"],
            "execution_success_rate": corpus_audit["execution_success_rate"],
            "forbidden_action_rate": corpus_audit["forbidden_action_rate"],
            "dataset_manifest_sha256": corpus_audit["dataset_manifest_sha256"],
        },
        # Promotion requires the separate held-out structural and safety
        # evaluation suite; a successful optimizer run is never sufficient.
        "production_ready": False,
        "train_metrics": train_result.metrics,
        "peak_gpu_memory_allocated_gib": round(peak_allocated_gib, 4),
        "peak_gpu_memory_reserved_gib": round(peak_reserved_gib, 4),
        "adapter_integrity": saved_adapter_integrity,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    if args.corpus_contract == "journal":
        manifest["journal_role_ready"] = False
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
