"""Tests for core.trainer — Trainer fit loop."""

import os
import pathlib
from typing import Any

import pytest
import torch
from torch.utils.data import TensorDataset

from lungula.core.models.lstm_agent import LSTMAgent
from lungula.core.trainer import Trainer

_FEATURE_DIM = 4
_ACTION_DIM = 2
_WINDOW = 8
_N_SAMPLES = 60


@pytest.fixture()
def tiny_dataset() -> TensorDataset:
    x = torch.randn(_N_SAMPLES, _WINDOW, _FEATURE_DIM)
    y = torch.randn(_N_SAMPLES, _ACTION_DIM)
    return TensorDataset(x, y)


@pytest.fixture()
def trainer() -> Trainer:
    model = LSTMAgent(feature_dim=_FEATURE_DIM, action_dim=_ACTION_DIM, hidden_size=32)
    device = torch.device("cpu")
    return Trainer(model, device, lr=1e-3)


class TestTrainer:
    def test_fit_returns_history_with_correct_length(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=3, batch_size=16)
        assert len(history) == 3

    def test_history_has_required_keys(self, trainer: Trainer, tiny_dataset: TensorDataset) -> None:
        history = trainer.fit(tiny_dataset, epochs=1, batch_size=16)
        assert "epoch" in history[0]
        assert "train" in history[0]
        assert "val" in history[0]
        assert "lr" in history[0]

    def test_lr_in_history_is_positive(self, trainer: Trainer, tiny_dataset: TensorDataset) -> None:
        history = trainer.fit(tiny_dataset, epochs=2, batch_size=16)
        for entry in history:
            assert isinstance(entry["lr"], float)
            assert entry["lr"] > 0

    def test_grad_clip_param_accepted(self, tiny_dataset: TensorDataset) -> None:
        model = LSTMAgent(feature_dim=_FEATURE_DIM, action_dim=_ACTION_DIM, hidden_size=32)
        trainer = Trainer(model, torch.device("cpu"), lr=1e-3, grad_clip=0.5)
        history = trainer.fit(tiny_dataset, epochs=1, batch_size=16)
        assert len(history) == 1

    def test_null_baseline_is_positive(self, trainer: Trainer, tiny_dataset: TensorDataset) -> None:
        from torch.utils.data import DataLoader

        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            tiny_dataset, batch_size=16
        )
        baseline = trainer._null_baseline(loader)
        assert baseline > 0

    def test_epoch_numbers_are_sequential(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=4, batch_size=16)
        for i, entry in enumerate(history, start=1):
            assert entry["epoch"] == i

    def test_loss_values_are_finite(self, trainer: Trainer, tiny_dataset: TensorDataset) -> None:
        history = trainer.fit(tiny_dataset, epochs=2, batch_size=16)
        for entry in history:
            assert isinstance(entry["train"], float)
            assert isinstance(entry["val"], float)
            assert entry["train"] >= 0
            assert entry["val"] >= 0

    def test_loss_does_not_increase_wildly(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        # Training loss after 5 epochs must stay below 100 (sanity guard only)
        history = trainer.fit(tiny_dataset, epochs=5, batch_size=16)
        assert history[-1]["train"] < 100.0

    def test_checkpoint_dir_creates_files(
        self, trainer: Trainer, tiny_dataset: TensorDataset, tmp_path: pathlib.Path
    ) -> None:
        ckpt_dir = str(tmp_path / "ckpts")
        trainer.fit(tiny_dataset, epochs=2, batch_size=16, checkpoint_dir=ckpt_dir)
        files = os.listdir(ckpt_dir)
        assert len(files) == 2
        assert all(f.endswith(".pt") for f in files)

    def test_on_epoch_called_once_per_epoch(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        calls: list[dict[str, Any]] = []

        def record(entry: dict[str, Any]) -> None:
            calls.append(entry)

        trainer.fit(tiny_dataset, epochs=3, batch_size=16, on_epoch=record)
        assert len(calls) == 3
        assert [c["epoch"] for c in calls] == [1, 2, 3]

    def test_on_epoch_returning_false_stops_training_early(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(
            tiny_dataset, epochs=5, batch_size=16, on_epoch=lambda entry: entry["epoch"] < 2
        )
        # Runs epoch 1 (on_epoch returns True, continue), epoch 2 (on_epoch returns
        # False, stop) — never reaches epoch 3, unlike a full 5-epoch run.
        assert len(history) == 2
        assert history[-1]["epoch"] == 2

    def test_stopping_early_still_writes_the_last_epoch_checkpoint(
        self, trainer: Trainer, tiny_dataset: TensorDataset, tmp_path: pathlib.Path
    ) -> None:
        # A graceful stop must never lose the checkpoint for the epoch it stopped on —
        # that's the entire point of checking on_epoch after the checkpoint write, not
        # before it (see the comment in Trainer.fit's docstring).
        ckpt_dir = str(tmp_path / "ckpts")
        trainer.fit(
            tiny_dataset,
            epochs=5,
            batch_size=16,
            checkpoint_dir=ckpt_dir,
            on_epoch=lambda entry: entry["epoch"] < 2,
        )
        assert os.path.exists(f"{ckpt_dir}/epoch_002.pt")
        assert not os.path.exists(f"{ckpt_dir}/epoch_003.pt")
