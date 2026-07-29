from __future__ import annotations

import hashlib
import json

import pytest


def test_initial_adapter_requires_matching_promoted_manifest(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    train = pytest.importorskip("training.train_qlora")
    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model", "r": 16}), encoding="utf-8"
    )
    tensor_path = adapter / "adapter_model.safetensors"
    save_file({"layer.lora_B.weight": torch.ones(2, 2)}, tensor_path)
    sha256 = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = {
        "base_model": "model",
        "base_revision": "revision",
        "lora_rank": 16,
        "production_ready": True,
        "adapter_integrity": {"sha256": sha256},
    }
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metadata = train.initial_adapter_metadata(
        adapter,
        model="model",
        revision="revision",
        lora_rank=16,
        require_promoted=True,
    )
    assert metadata["integrity"]["sha256"] == sha256

    manifest["production_ready"] = False
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="promoted"):
        train.initial_adapter_metadata(
            adapter,
            model="model",
            revision="revision",
            lora_rank=16,
            require_promoted=True,
        )
