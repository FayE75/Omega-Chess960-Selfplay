from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    p = argparse.ArgumentParser(description="Timed Omega Chess960 self-play with quality-weighted Polyglot learning")
    p.add_argument("--engine", required=True)
    p.add_argument("--book", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--games", type=int, default=0, help="0 = unlimited games until duration deadline")
    p.add_argument("--duration-minutes", type=float, default=330.0)
    p.add_argument("--nodes", type=int, default=15000)
    p.add_argument("--multipv", type=int, default=3)
    p.add_argument("--good-move-cp", type=int, default=35,
                   help="Only candidate moves within this centipawn gap from the best move may be selected")
    p.add_argument("--choice-temperature-cp", type=float, default=14.0,
                   help="Softmax temperature for choosing among good candidates")
    p.add_argument("--book-plies", type=int, default=0,
                   help="0 = follow the active enriched book until its line ends; positive value caps book plies")
    p.add_argument("--max-book-plies-safety", type=int, default=512,
                   help="Safety cap when --book-plies=0")
    p.add_argument("--max-plies", type=int, default=260)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--engine-id", default="Omega-main")
    p.add_argument("--resign-cp", type=int, default=900)
    p.add_argument("--resign-plies", type=int, default=6)
    p.add_argument("--draw-cp", type=int, default=20)
    p.add_argument("--draw-plies", type=int, default=12)
    p.add_argument("--draw-after-ply", type=int, default=90)
    p.add_argument("--book-weight-power", type=float, default=0.55,
                   help="<1 flattens active-book weights to increase line diversity")
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


def start_with_book(
    reader: chess.polyglot.MemoryMappedReader,
    rng: random.Random,
    book_plies: int,
    max_book_plies_safety: int,
    power: float,
):
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
        cap = book_plies if book_plies > 0 else max_book_plies_safety

        for _ply in range(cap):
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
    raise RuntimeError("Could not find a Chess960 start position covered by the active book.")


def result_for_color(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if color == chess.BLACK else "loss"
    return "draw"


def quality_points(gap_cp: int) -> int:
    if gap_cp <= 2:
        return 20
    if gap_cp <= 5:
        return 16
    if gap_cp <= 10:
        return 12
    if gap_cp <= 20:
        return 8
    if gap_cp <= 35:
        return 4
    if gap_cp <= 50:
        return 2
    return 1


def learned_weight(result: str, color: chess.Color, gap_cp: int, ply: int) -> int:
    result_factor = {"win": 1.35, "draw": 1.0, "loss": 0.55}[result_for_color(result, color)]
    depth_factor = max(0.30, math.exp(-max(0, ply - 40) / 140.0))
    return max(1, int(round(quality_points(gap_cp) * result_factor * depth_factor)))


def configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    # python-chess automatically manages protocol-sensitive UCI options such as
    # Ponder and UCI_Chess960. Configuring Ponder manually raises EngineError.
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)


def analyse_good_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
    max_gap_cp: int,
    temperature_cp: float,
    rng: random.Random,
    game_token: object,
) -> tuple[chess.Move, int, int, list[tuple[str, int, int]]]:
    mover = board.turn
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=max(1, multipv),
        game=game_token,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    infos = info if isinstance(info, list) else [info]

    candidates: list[tuple[chess.Move, int]] = []
    for item in infos:
        pv = item.get("pv") or []
        score_obj = item.get("score")
        if not pv or score_obj is None:
            continue
        cp = score_obj.pov(mover).score(mate_score=40000)
        if cp is None:
            continue
        candidates.append((pv[0], int(cp)))

    if not candidates:
        raise RuntimeError("Omega returned no scored PV candidate.")

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_cp = candidates[0][1]
    good: list[tuple[chess.Move, int, int]] = []
    seen: set[chess.Move] = set()
    for move, cp in candidates:
        if move in seen or move not in board.legal_moves:
            continue
        seen.add(move)
        gap = max(0, best_cp - cp)
        if gap <= max_gap_cp:
            good.append((move, cp, gap))

    if not good:
        move, cp = candidates[0]
        good = [(move, cp, 0)]

    temp = max(1.0, temperature_cp)
    weights = [math.exp(-gap / temp) for _move, _cp, gap in good]
    chosen = rng.choices(good, weights=weights, k=1)[0]
    preview = [(move.uci(), cp, gap) for move, cp, gap in good]
    return chosen[0], chosen[1], chosen[2], preview


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
    deadline = started + max(1.0, args.duration_minutes) * 60.0
    completed_games = 0

    white_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    black_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    configure_engine(white_engine, args.threads, args.hash_mb)
    configure_engine(black_engine, args.threads, args.hash_mb)

    try:
        with chess.polyglot.open_reader(book_path) as reader, open(pgn_path, "w", encoding="utf-8") as pgn_out:
            game_no = 0
            while True:
                if time.time() >= deadline:
                    break
                if args.games > 0 and game_no >= args.games:
                    break
                game_no += 1

                index, initial, board, seed_moves = start_with_book(
                    reader,
                    rng,
                    args.book_plies,
                    args.max_book_plies_safety,
                    args.book_weight_power,
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
                game.headers["ActiveBookSHA256"] = book_hash
                game.headers["BookPlies"] = str(len(seed_moves))
                node = game

                replay = initial.copy(stack=False)
                for _ply, (_key, move, weight) in enumerate(seed_moves):
                    node = node.add_variation(move)
                    node.comment = f"active-book weight={weight}"
                    replay.push(move)

                training: list[tuple[int, int, chess.Color, int, int]] = []
                bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
                equal_streak = 0
                result = "*"
                game_token = (args.seed, game_no)

                while board.ply() < args.max_plies:
                    if time.time() >= deadline:
                        result = "1/2-1/2"
                        break

                    outcome = board.outcome(claim_draw=True)
                    if outcome:
                        result = outcome.result()
                        break

                    mover = board.turn
                    engine = white_engine if mover == chess.WHITE else black_engine
                    key = chess.polyglot.zobrist_hash(board)
                    move, cp, gap_cp, preview = analyse_good_moves(
                        engine,
                        board,
                        args.nodes,
                        args.multipv,
                        args.good_move_cp,
                        args.choice_temperature_cp,
                        rng,
                        game_token,
                    )

                    training.append((key, encode_polyglot_move(board, move), mover, board.ply(), gap_cp))
                    node = node.add_variation(move)
                    candidates_txt = ",".join(f"{uci}:{score/100:.2f}/{gap}" for uci, score, gap in preview)
                    node.comment = (
                        f"eval={cp/100:.2f} gap={gap_cp}cp nodes={args.nodes} "
                        f"good=[{candidates_txt}]"
                    )
                    board.push(move)

                    if cp <= -args.resign_cp:
                        bad_streak[mover] += 1
                    else:
                        bad_streak[mover] = 0
                    if bad_streak[mover] >= args.resign_plies:
                        result = "0-1" if mover == chess.WHITE else "1-0"
                        break

                    if board.ply() >= args.draw_after_ply and abs(cp) <= args.draw_cp:
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
                completed_games += 1

                for key, raw_move, color, ply, gap_cp in training:
                    updates[(key, raw_move)] += learned_weight(result, color, gap_cp, ply)

                print(game, file=pgn_out, end="\n\n")
                pgn_out.flush()
                elapsed_min = (time.time() - started) / 60.0
                print(
                    f"game {game_no} start={index:03d} result={result} plies={board.ply()} "
                    f"book_plies={len(seed_moves)} learned={len(updates)} elapsed={elapsed_min:.1f}m"
                )
    finally:
        white_engine.quit()
        black_engine.quit()

    elapsed = time.time() - started
    metadata = {
        "engine": args.engine_id,
        "active_book_sha256": book_hash,
        "games": completed_games,
        "duration_minutes_requested": args.duration_minutes,
        "nodes_per_move": args.nodes,
        "multipv": args.multipv,
        "good_move_cp": args.good_move_cp,
        "choice_temperature_cp": args.choice_temperature_cp,
        "book_plies": "until_active_book_line_ends" if args.book_plies == 0 else args.book_plies,
        "training_depth": "unlimited_until_game_end",
        "max_plies_safety": args.max_plies,
        "seed": args.seed,
        "weighting": {
            "quality_gap_points": {"<=2": 20, "<=5": 16, "<=10": 12, "<=20": 8, "<=35": 4, "<=50": 2, "other": 1},
            "result_factor": {"win": 1.35, "draw": 1.0, "loss": 0.55},
            "depth_decay": "max(0.30, exp(-max(0, ply-40)/140))",
        },
    }
    save_updates(updates_path, dict(updates), metadata)
    summary = {
        **metadata,
        "results": stats,
        "unique_learned_moves": len(updates),
        "total_plies": total_plies,
        "elapsed_seconds": round(elapsed, 3),
        "games_per_hour": round(completed_games / elapsed * 3600, 2) if elapsed else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
