"""lun'gula CLI — train --game osu --data ./replays"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from types import FrameType
from typing import Any, cast

GAMES: dict[str, str] = {
    "osu": "lungula.games.osu.plugin",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lungula",
        description="lun'gula — game imitation learning framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model on replay data")
    train.add_argument("--game", required=True, choices=list(GAMES), help="Game plugin to use")
    train.add_argument("--data", required=True, help="Directory containing replay + beatmap files")
    train.add_argument("--out", default="checkpoints", help="Output directory for checkpoints")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch", type=int, default=128)
    train.add_argument("--window", type=int, default=32, help="Context window size (frames)")
    train.add_argument("--device", default="auto", help="auto | cuda | directml | mps | cpu")
    train.add_argument("--export", default=None, help="Export final model to this .onnx path")
    train.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)"
    )
    train.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        dest="grad_clip",
        help="Gradient clipping max norm (default: 1.0, 0 = disabled)",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in --out",
    )
    train.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        dest="scheduler_patience",
        help="ReduceLROnPlateau patience in epochs before halving LR (default: 5)",
    )
    train.add_argument(
        "--json",
        action="store_true",
        help="Also emit one NDJSON event per line on stdout (started/epoch_completed/"
        "stopped/completed/error) for programmatic consumption, alongside the normal "
        "human-readable output.",
    )
    train.add_argument(
        "--stop-file",
        default=None,
        dest="stop_file",
        help="Path checked for existence after each epoch; if present, stop gracefully "
        "at the next checkpoint boundary. This is the mechanism a remote controller "
        "(Natsume's lungula_runner.ts) should use instead of killing the process:  on "
        "Windows, Node's ChildProcess.kill() cannot deliver a real POSIX signal to a "
        "child process — it force-terminates instead, which is the exact "
        "un-graceful-stop problem this whole feature exists to fix. SIGTERM/SIGINT "
        "still work for direct interactive use (a human's own Ctrl-C).",
    )

    games_cmd = sub.add_parser("games", help="List available game plugins")
    games_cmd.add_argument("--json", action="store_true", help="Output as a JSON array")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "games":
        _cmd_games(args)


def _cmd_games(args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps([{"id": game_id, "module": module} for game_id, module in GAMES.items()]))
    else:
        for game_id in GAMES:
            print(game_id)


def _emit(args: argparse.Namespace, event: dict[str, Any]) -> None:
    if args.json:
        print(json.dumps(event), flush=True)


def _cmd_train(args: argparse.Namespace) -> None:
    import importlib

    from lungula.core.dataset import ReplayDataset
    from lungula.core.device import resolve_device
    from lungula.core.export.onnx_exporter import export_onnx
    from lungula.core.trainer import Trainer

    plugin = importlib.import_module(GAMES[args.game])
    parser = plugin.make_parser()
    model = plugin.make_model()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(
        f"Game:   {args.game}  |  feature_dim={parser.feature_dim}  action_dim={parser.action_dim}"
    )

    pairs = plugin.collect_pairs(args.data)
    if not pairs:
        print(f"No replay pairs found in {args.data}", file=sys.stderr)
        _emit(args, {"type": "error", "message": f"No replay pairs found in {args.data}"})
        sys.exit(1)

    print(f"Replays: {len(pairs)}")
    flip_feat = getattr(plugin, "FLIP_FEAT_Y", None)
    flip_act = getattr(plugin, "FLIP_ACT_Y", None)
    dataset = ReplayDataset(
        pairs,
        parser,
        window=args.window,
        flip_feat_indices=flip_feat,
        flip_act_indices=flip_act,
    )
    print(f"Samples: {len(dataset)}")

    grad_clip = args.grad_clip if args.grad_clip > 0 else float("inf")
    trainer = Trainer(
        model, device, lr=args.lr, grad_clip=grad_clip, scheduler_patience=args.scheduler_patience
    )

    # Cooperative stop: a caller controlling this process asks for a graceful stop
    # rather than a hard kill. Trainer.fit() only checks this between epochs — after
    # that epoch's checkpoint is already on disk — so a stop never loses progress or
    # leaves a half-written checkpoint, unlike the raw Ctrl-C that produced the
    # interrupted, never-converged run behind CLAUDE.md's C-009.
    #
    # Two independent triggers, because neither alone covers every real caller:
    #   - SIGTERM/SIGINT: works for a human's own Ctrl-C in an interactive terminal.
    #   - --stop-file: what a remote controller (Natsume's lungula_runner.ts) must use
    #     instead — on Windows, Node's ChildProcess.kill() cannot deliver a real POSIX
    #     signal to a child process, it force-terminates instead, silently reproducing
    #     the exact ungraceful-stop problem this feature exists to fix.
    stop_requested = False

    def _request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def on_epoch(entry: dict[str, Any]) -> bool:
        _emit(args, {"type": "epoch_completed", **entry})
        stop_file_present = bool(args.stop_file) and os.path.exists(args.stop_file)
        return not (stop_requested or stop_file_present)

    _emit(args, {"type": "started", "totalEpochs": args.epochs, "device": str(device)})

    try:
        history = trainer.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch,
            checkpoint_dir=args.out,
            resume=args.resume,
            on_epoch=on_epoch,
        )
    except Exception as exc:
        _emit(args, {"type": "error", "message": str(exc)})
        raise

    last_epoch = history[-1]["epoch"] if history else None
    completed_fully = last_epoch == args.epochs
    if completed_fully:
        _emit(args, {"type": "completed", "history": history})
    else:
        _emit(args, {"type": "stopped", "epoch": last_epoch})

    if args.export:
        last = history[-1] if history else None
        export_onnx(
            model,
            args.export,
            window=args.window,
            source_checkpoint=f"{args.out}/epoch_{last['epoch']:03d}.pt"
            if last and args.out
            else None,
            epoch=cast(int, last["epoch"]) if last else None,
            train_loss=cast(float, last["train"]) if last else None,
            val_loss=cast(float, last["val"]) if last else None,
        )


if __name__ == "__main__":
    main()
