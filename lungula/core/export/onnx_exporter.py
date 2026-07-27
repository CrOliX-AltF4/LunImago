"""Export a trained model to ONNX for cross-platform inference."""

from __future__ import annotations

import json
import time

import torch

from ..models.base_model import BaseImitatonModel


def export_onnx(
    model: BaseImitatonModel,
    output_path: str,
    window: int = 32,
    opset: int = 17,
    source_checkpoint: str | None = None,
    epoch: int | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    """Export model to ONNX, plus a `<output_path>.manifest.json` recording where the
    weights actually came from — which checkpoint, which epoch, what loss it reached,
    and when. Consumers (e.g. Natsume's panel) can show real provenance instead of a
    bare file with no history, which is what let an unconverged, interrupted training
    run silently end up in production before (CLAUDE.md §7, C-009).

    The exported model takes a single input:
      context: float32[1, window, feature_dim]
    and produces:
      action: float32[1, action_dim]
    """
    model.eval()
    dummy = torch.zeros(1, window, model.feature_dim)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["context"],
        output_names=["action"],
        dynamic_axes={"context": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported to {output_path}")

    manifest = {
        "sourceCheckpoint": source_checkpoint,
        "epoch": epoch,
        "trainLoss": train_loss,
        "valLoss": val_loss,
        "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window": window,
        "featureDim": model.feature_dim,
        "actionDim": model.action_dim,
    }
    manifest_path = f"{output_path}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")
