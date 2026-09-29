from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.polyglot

from polyglot_utils import encode_polyglot_move, save_updates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omega Chess960 self-play from a Polyglot seed book")
    p.add_argument("--engine", required=True)
    p.add_argument("--book", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--nodes", type=int, default=30000)
    p.add_argument("--book-plies", type=int, default=12)
    p.add_argument("--train-plies", type=int, default=40)
    p.add_argument("--max-plies", type=int, default=220)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--engine-id", default="Omega-main")
    p.add_argument("--resign-cp", type=int, default=900)
    p.add_argument("--resign-plies", type=int, default=6)
    p.add_argument("--draw-cp", type=int, default=20)
    p.add_argument("--draw-plies", type=int, default=12)
    p.add_argument("--draw-after-ply", type=int, default=80)
    p.add_argument("--book-weight-power", type=float, default=0.55,
                   help="<1 flattens seed-book weights to increase line diversity")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_weighted(entries: list[chess.polyglot.Entry], rng: random.Random, power: float) -> chess.polyglot.Entry:
    weights = [max(1.0, float(e.weight)) ** power for e in entries]
    return rng.choices(entries, weights=weights, k=1)[0]


def start_with_book(reader: chess.polyglot.MemoryMappedReader, rng: random.Random, book_plies: int, power: float):
    for _ in range(4000):
        index = rng.randrange(960)
        board = chess.Board.from_chess960_pos(index)
        try:
            root_entries = list(reader.find_all(board))
        except Exception:
            root_entries = []
        if not root_entries:
            continue

        initial = board.copy(stack=False)
        seed_moves: list[tuple[int, chess.Move, int]] = []
        for _ply in range(book_plies):
            entries = list(reader.find_all(board))
            if not entries:
                break
            entry = choose_weighted(entries, rng, power)
            move = entry.move
            if move not in board.legal_moves:
                break
            seed_moves.append((chess.polyglot.zobrist_hash(board), move, entry.weight))
            board.push(move)
        return index, initial, board, seed_moves
    raise RuntimeError("Could not find a Chess960 start position covered by the seed book.")


def result_for_color(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if color == chess.BLACK else "loss"
    return "draw"


def weight_for_result(result: str, color: chess.Color) -> int:
    perspective = result_for_color(result, color)
    return 8 if perspective == "win" else 3 if perspective == "draw" else 0


def configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if "Ponder" in engine.options:
        options["Ponder"] = False
    if options:
        engine.configure(options)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / "selfplay.pgn"
    updates_path = out_dir / "run_updates.json.gz"
    summary_path = out_dir / "summary.json"

    book_path = Path(args.book)
    book_hash = sha256(book_path)
    updates: dict[tuple[int, int], int] = defaultdict(int)
    stats = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    total_plies = 0
    started = time.time()

    white_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    black_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    configure_engine(white_engine, args.threads, args.hash_mb)
    configure_engine(black_engine, args.threads, args.hash_mb)

    try:
        with chess.polyglot.open_reader(book_path) as reader, open(pgn_path, "w", encoding="utf-8") as pgn_out:
            for game_no in range(1, args.games + 1):
                index, initial, board, seed_moves = start_with_book(
                    reader, rng, args.book_plies, args.book_weight_power
                )

                game = chess.pgn.Game()
                game.setup(initial)
                game.headers["Event"] = "Omega Chess960 Selfplay"
                game.headers["Site"] = "GitHub Actions"
                game.headers["Round"] = str(game_no)
                game.headers["White"] = args.engine_id
                game.headers["Black"] = args.engine_id
                game.headers["Variant"] = "Chess960"
                game.headers["Scharnagl"] = str(index)
                game.headers["SeedBookSHA256"] = book_hash
                node = game

                training: list[tuple[int, int, chess.Color, int]] = []
                replay = initial.copy(stack=False)
                for ply, (key, move, weight) in enumerate(seed_moves):
                    training.append((key, encode_polyglot_move(replay, move), replay.turn, ply))
                    node = node.add_variation(move)
                    node.comment = f"seed-book weight={weight}"
                    replay.push(move)

                bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
                equal_streak = 0
                result = "*"
                game_token = (args.seed, game_no)

                while board.ply() < args.max_plies:
                    outcome = board.outcome(claim_draw=True)
                    if outcome:
                        result = outcome.result()
                        break

                    mover = board.turn
                    engine = white_engine if mover == chess.WHITE else black_engine
                    key = chess.polyglot.zobrist_hash(board)
                    play = engine.play(
                        board,
                        chess.engine.Limit(nodes=args.nodes),
                        game=game_token,
                        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                    )
                    move = play.move
                    score_obj = play.info.get("score")
                    cp = None
                    if score_obj is not None:
                        cp = score_obj.pov(mover).score(mate_score=40000)

                    if board.ply() < args.train_plies:
                        training.append((key, encode_polyglot_move(board, move), mover, board.ply()))

                    node = node.add_variation(move)
                    if cp is not None:
                        node.comment = f"eval={cp/100:.2f} nodes={args.nodes}"
                    board.push(move)

                    if cp is not None and cp <= -args.resign_cp:
                        bad_streak[mover] += 1
                    else:
                        bad_streak[mover] = 0
                    if bad_streak[mover] >= args.resign_plies:
                        result = "0-1" if mover == chess.WHITE else "1-0"
                        break

                    if board.ply() >= args.draw_after_ply and cp is not None and abs(cp) <= args.draw_cp:
                        equal_streak += 1
                    else:
                        equal_streak = 0
                    if equal_streak >= args.draw_plies:
                        result = "1/2-1/2"
                        break
                else:
                    result = "1/2-1/2"

                if result == "*":
                    result = "1/2-1/2"
                game.headers["Result"] = result
                stats[result] += 1
                total_plies += board.ply()

                for key, raw_move, color, _ply in training:
                    delta = weight_for_result(result, color)
                    if delta:
                        updates[(key, raw_move)] += delta

                print(game, file=pgn_out, end="\n\n")
                pgn_out.flush()
                print(
                    f"game {game_no}/{args.games} start={index:03d} result={result} "
                    f"plies={board.ply()} learned={len(updates)}"
                )
    finally:
        white_engine.quit()
        black_engine.quit()

    elapsed = time.time() - started
    metadata = {
        "engine": args.engine_id,
        "seed_book_sha256": book_hash,
        "games": args.games,
        "nodes_per_move": args.nodes,
        "book_plies": args.book_plies,
        "train_plies": args.train_plies,
        "seed": args.seed,
        "result_weights": {"win": 8, "draw": 3, "loss": 0},
    }
    save_updates(updates_path, dict(updates), metadata)
    summary = {
        **metadata,
        "results": stats,
        "unique_learned_moves": len(updates),
        "total_plies": total_plies,
        "elapsed_seconds": round(elapsed, 3),
        "games_per_hour": round(args.games / elapsed * 3600, 2) if elapsed else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
