"""Tests for core.export.onnx_exporter — ONNX export and inference."""

import json
import os
import pathlib

import pytest
import torch

from lungula.core.export.onnx_exporter import export_onnx
from lungula.core.models.lstm_agent import LSTMAgent


@pytest.fixture()
def model() -> LSTMAgent:
    return LSTMAgent(feature_dim=10, action_dim=4, hidden_size=32)


class TestOnnxExporter:
    def test_export_creates_file(self, model: LSTMAgent, tmp_path: pathlib.Path) -> None:
        path = str(tmp_path / "model.onnx")
        export_onnx(model, path, window=16)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_exported_model_is_loadable(self, model: LSTMAgent, tmp_path: pathlib.Path) -> None:
        import onnx

        path = str(tmp_path / "model.onnx")
        export_onnx(model, path, window=16)
        loaded = onnx.load(path)
        onnx.checker.check_model(loaded)

    def test_inference_with_onnxruntime(self, model: LSTMAgent, tmp_path: pathlib.Path) -> None:
        import numpy as np
        import onnxruntime as ort

        path = str(tmp_path / "model.onnx")
        export_onnx(model, path, window=16)

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        context = np.zeros((1, 16, 10), dtype=np.float32)
        outputs = sess.run(None, {"context": context})
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 4)

    def test_output_matches_torch_forward(self, model: LSTMAgent, tmp_path: pathlib.Path) -> None:
        import numpy as np
        import onnxruntime as ort

        path = str(tmp_path / "model.onnx")
        export_onnx(model, path, window=8)

        x = torch.randn(1, 8, 10)
        model.eval()
        with torch.no_grad():
            torch_out = model(x).numpy()

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"context": x.numpy()})[0]

        assert np.allclose(torch_out, ort_out, atol=1e-5)

    def test_writes_a_provenance_manifest_alongside_the_export(
        self, model: LSTMAgent, tmp_path: pathlib.Path
    ) -> None:
        # This is the actual fix for CLAUDE.md's C-009: a model with no recorded
        # provenance (which checkpoint, which epoch, what loss it reached) is how an
        # interrupted, never-converged run silently ended up wired into production.
        path = str(tmp_path / "model.onnx")
        export_onnx(
            model,
            path,
            window=16,
            source_checkpoint="checkpoints/epoch_041.pt",
            epoch=41,
            train_loss=0.05,
            val_loss=0.0475,
        )
        manifest_path = pathlib.Path(f"{path}.manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["sourceCheckpoint"] == "checkpoints/epoch_041.pt"
        assert manifest["epoch"] == 41
        assert manifest["trainLoss"] == 0.05
        assert manifest["valLoss"] == 0.0475
        assert manifest["window"] == 16
        assert manifest["featureDim"] == 10
        assert manifest["actionDim"] == 4
        assert "exportedAt" in manifest

    def test_manifest_fields_default_to_none_when_provenance_not_given(
        self, model: LSTMAgent, tmp_path: pathlib.Path
    ) -> None:
        path = str(tmp_path / "model.onnx")
        export_onnx(model, path, window=16)
        manifest = json.loads(pathlib.Path(f"{path}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["sourceCheckpoint"] is None
        assert manifest["epoch"] is None
