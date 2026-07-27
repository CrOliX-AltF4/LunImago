"""Tests for cli.py — the `games` subcommand and the NDJSON event emitter.

`train` itself isn't exercised end-to-end here (it needs real replay data — covered
indirectly via core.trainer's own test suite for the training-loop behavior this CLI
wraps); these tests cover what's genuinely new and standalone: game-registry listing
and the --json event-emission helper `train` relies on.
"""

import argparse
import json

import pytest

from lungula.cli import GAMES, _cmd_games, _emit


class TestCmdGames:
    def test_prints_plain_game_ids_by_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        _cmd_games(argparse.Namespace(json=False))
        out = capsys.readouterr().out.strip().splitlines()
        assert out == list(GAMES)

    def test_prints_a_json_array_with_id_and_module(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_games(argparse.Namespace(json=True))
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed == [{"id": game_id, "module": module} for game_id, module in GAMES.items()]


class TestEmit:
    def test_prints_nothing_when_json_flag_is_off(self, capsys: pytest.CaptureFixture[str]) -> None:
        _emit(argparse.Namespace(json=False), {"type": "started"})
        assert capsys.readouterr().out == ""

    def test_prints_the_event_as_a_single_json_line_when_json_flag_is_on(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _emit(argparse.Namespace(json=True), {"type": "epoch_completed", "epoch": 3})
        out = capsys.readouterr().out.strip()
        assert json.loads(out) == {"type": "epoch_completed", "epoch": 3}
