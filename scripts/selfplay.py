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
    p = argparse.ArgumentParser(
        description="Timed Omega Chess960 self-play using Fishtest-style LTC and quality-filtered Polyglot learning"
    )
    p.add_argument("--engine", required=True)
    p.add_argument("--book", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--games", type=int, default=0, help="0 = unlimited games until duration deadline")
    p.add_argument("--duration-minutes", type=float, default=330.0)
    p.add_argument("--tc-base-seconds", type=float, default=60.0, help="LTC base clock per side")
    p.add_argument("--tc-increment-seconds", type=float, default=0.6, help="LTC increment per move")
    p.add_argument("--multipv", type=int, default=3)
    p.add_argument(
        "--good-move-cp",
        type=int,
        default=35,
        help="Only candidate moves within this centipawn gap from the best move may be selected",
    )
    p.add_argument(
        "--choice-temperature-cp",
        type=float,
        default=14.0,
        help="Softmax temperature for choosing among good candidates",
    )
    p.add_argument(
        "--book-plies",
        type=int,
        default=12,
        help="Maximum opening-book prefix before Omega takes over; does not limit learned book depth",
    )
    p.add_argument(
        "--max-plies",
        type=int,
        default=0,
        help="0 = no artificial game/book-depth cap; positive values are emergency safety caps only",
    )
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--engine-id", default="Omega-main")
    p.add_argument("--resign-cp", type=int, default=900)
    p.add_argument("--resign-plies", type=int, default=6)
    p.add_argument("--draw-cp", type=int, default=20)
    p.add_argument("--draw-plies", type=int, default=12)
    p.add_argument("--draw-after-ply", type=int, default=90)
    p.add_argument(
        "--book-weight-power",
        type=float,
        default=0.55,
        help="<1 flattens existing book weights to increase line diversity",
    )
    p.add_argument("--draw-learn-gap-cp", type=int, default=10)
    p.add_argument("--win-learn-gap-cp", type=int, default=20)
    p.add_argument(
        "--learn-min-cp",
        type=int,
        default=-100,
        help="Do not learn a continuation if its evaluation is worse than this for the mover",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_weighted(
    entries: list[chess.polyglot.Entry], rng: random.Random, power: float
) -> chess.polyglot.Entry:
    weights = [max(1.0, float(e.weight)) ** power for e in entries]
    return rng.choices(entries, weights=weights, k=1)[0]


def start_with_book(
    reader: chess.polyglot.MemoryMappedReader,
    rng: random.Random,
    book_plies: int,
    power: float,
):
    if book_plies < 0:
        raise ValueError("--book-plies must be >= 0")

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
            if board.is_game_over(claim_draw=True):
                break

        return index, initial, board, seed_moves

    raise RuntimeError("Could not find a Chess960 start position covered by the active book.")


def result_for_color(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if color == chess.BLACK else "loss"
    return "unknown"


def quality_points(gap_cp: int) -> int:
    if gap_cp <= 2:
        return 20
    if gap_cp <= 5:
        return 16
    if gap_cp <= 10:
        return 12
    if gap_cp <= 20:
        return 8
    return 0


def learning_decision(
    result: str,
    color: chess.Color,
    gap_cp: int,
    eval_cp: int,
    ply: int,
    draw_gap_cp: int,
    win_gap_cp: int,
    min_eval_cp: int,
) -> tuple[int, str]:
    """Return (weight, reason) for one engine-generated continuation move."""
    perspective = result_for_color(result, color)

    if perspective == "unknown":
        return 0, "reject_unknown_result"
    if perspective == "loss":
        return 0, "reject_loss"
    if eval_cp < min_eval_cp:
        return 0, "reject_bad_eval"
    if perspective == "draw" and gap_cp > draw_gap_cp:
        return 0, "reject_draw_gap"
    if perspective == "win" and gap_cp > win_gap_cp:
        return 0, "reject_win_gap"

    base = quality_points(gap_cp)
    if base <= 0:
        return 0, "reject_quality"

    result_factor = 1.50 if perspective == "win" else 0.60
    depth_factor = max(0.30, math.exp(-max(0, ply - 40) / 140.0))
    weight = max(1, int(round(base * result_factor * depth_factor)))
    return weight, f"accept_{perspective}"


def configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    # python-chess automatically manages protocol-sensitive options such as
    # Ponder and UCI_Chess960.
    if options:
        engine.configure(options)


def analyse_good_moves_ltc(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    clocks: dict[chess.Color, float],
    increment_seconds: float,
    multipv: int,
    max_gap_cp: int,
    temperature_cp: float,
    rng: random.Random,
    game_token: object,
) -> tuple[chess.Move, int, int, list[tuple[str, int, int]], float, int]:
    mover = board.turn
    limit = chess.engine.Limit(
        white_clock=max(0.001, clocks[chess.WHITE]),
        black_clock=max(0.001, clocks[chess.BLACK]),
        white_inc=max(0.0, increment_seconds),
        black_inc=max(0.0, increment_seconds),
    )

    wall_started = time.perf_counter()
    info = engine.analyse(
        board,
        limit,
        multipv=max(1, multipv),
        game=game_token,
        info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    wall_seconds = time.perf_counter() - wall_started
    infos = info if isinstance(info, list) else [info]

    candidates: list[tuple[chess.Move, int]] = []
    reported_seconds = 0.0
    searched_nodes = 0

    for item in infos:
        pv = item.get("pv") or []
        score_obj = item.get("score")
        if pv and score_obj is not None:
            cp = score_obj.pov(mover).score(mate_score=40000)
            if cp is not None:
                candidates.append((pv[0], int(cp)))
        reported_seconds = max(reported_seconds, float(item.get("time", 0.0) or 0.0))
        searched_nodes = max(searched_nodes, int(item.get("nodes", 0) or 0))

    if not candidates:
        raise RuntimeError("Omega returned no scored PV candidate.")

    spent_seconds = reported_seconds if reported_seconds > 0 else wall_seconds
    clocks[mover] = max(0.0, clocks[mover] - spent_seconds + increment_seconds)

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
    return chosen[0], chosen[1], chosen[2], preview, spent_seconds, searched_nodes


def main() -> None:
    args = parse_args()
    if args.tc_base_seconds <= 0:
        raise ValueError("--tc-base-seconds must be > 0")
    if args.tc_increment_seconds < 0:
        raise ValueError("--tc-increment-seconds must be >= 0")

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
    learning_stats: dict[str, int] = defaultdict(int)
    total_plies = 0
    total_engine_plies = 0
    total_search_seconds = 0.0
    total_searched_nodes = 0
    started = time.time()
    deadline = started + max(1.0, args.duration_minutes) * 60.0
    completed_games = 0

    white_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    black_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    configure_engine(white_engine, args.threads, args.hash_mb)
    configure_engine(black_engine, args.threads, args.hash_mb)

    try:
        with chess.polyglot.open_reader(book_path) as reader, open(
            pgn_path, "w", encoding="utf-8"
        ) as pgn_out:
            game_no = 0
            while True:
                # Do not start a new game after the production deadline. A game
                # already in progress is allowed to finish so it is never turned
                # into an artificial draw merely because the wall-clock window ended.
                if time.time() >= deadline:
                    break
                if args.games > 0 and game_no >= args.games:
                    break
                game_no += 1

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
                game.headers["ActiveBookSHA256"] = book_hash
                game.headers["BookPlies"] = str(len(seed_moves))
                game.headers["TimeControl"] = (
                    f"{args.tc_base_seconds:g}+{args.tc_increment_seconds:g}"
                )
                node = game

                for _key, move, weight in seed_moves:
                    node = node.add_variation(move)
                    node.comment = f"active-book weight={weight}"

                training: list[
                    tuple[int, int, chess.Color, int, int, int]
                ] = []
                bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
                equal_streak = 0
                result = "*"
                game_token = (args.seed, game_no)
                clocks = {
                    chess.WHITE: float(args.tc_base_seconds),
                    chess.BLACK: float(args.tc_base_seconds),
                }
                game_search_seconds = 0.0
                game_engine_plies = 0

                while True:
                    if args.max_plies > 0 and board.ply() >= args.max_plies:
                        result = "1/2-1/2"
                        break

                    outcome = board.outcome(claim_draw=True)
                    if outcome:
                        result = outcome.result()
                        break

                    mover = board.turn
                    engine = white_engine if mover == chess.WHITE else black_engine
                    key = chess.polyglot.zobrist_hash(board)
                    (
                        move,
                        cp,
                        gap_cp,
                        preview,
                        search_seconds,
                        searched_nodes,
                    ) = analyse_good_moves_ltc(
                        engine,
                        board,
                        clocks,
                        args.tc_increment_seconds,
                        args.multipv,
                        args.good_move_cp,
                        args.choice_temperature_cp,
                        rng,
                        game_token,
                    )

                    training.append(
                        (
                            key,
                            encode_polyglot_move(board, move),
                            mover,
                            board.ply(),
                            gap_cp,
                            cp,
                        )
                    )
                    game_engine_plies += 1
                    game_search_seconds += search_seconds
                    total_searched_nodes += searched_nodes

                    node = node.add_variation(move)
                    candidates_txt = ",".join(
                        f"{uci}:{score/100:.2f}/{gap}"
                        for uci, score, gap in preview
                    )
                    node.comment = (
                        f"eval={cp/100:.2f} gap={gap_cp}cp "
                        f"search_ms={search_seconds*1000:.1f} nodes={searched_nodes} "
                        f"clock_w={clocks[chess.WHITE]:.3f} "
                        f"clock_b={clocks[chess.BLACK]:.3f} "
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

                if result == "*":
                    continue

                game.headers["Result"] = result
                stats[result] += 1
                total_plies += board.ply()
                total_engine_plies += game_engine_plies
                total_search_seconds += game_search_seconds
                completed_games += 1

                for key, raw_move, color, ply, gap_cp, eval_cp in training:
                    delta, reason = learning_decision(
                        result,
                        color,
                        gap_cp,
                        eval_cp,
                        ply,
                        args.draw_learn_gap_cp,
                        args.win_learn_gap_cp,
                        args.learn_min_cp,
                    )
                    learning_stats[reason] += 1
                    if delta > 0:
                        updates[(key, raw_move)] += delta

                print(game, file=pgn_out, end="\n\n")
                pgn_out.flush()

                elapsed_min = (time.time() - started) / 60.0
                avg_search_ms = (
                    game_search_seconds / game_engine_plies * 1000.0
                    if game_engine_plies
                    else 0.0
                )
                print(
                    f"game {game_no} start={index:03d} result={result} "
                    f"plies={board.ply()} book_plies={len(seed_moves)} "
                    f"engine_plies={game_engine_plies} learned={len(updates)} "
                    f"avg_search_ms={avg_search_ms:.1f} "
                    f"clock_w={clocks[chess.WHITE]:.2f}s "
                    f"clock_b={clocks[chess.BLACK]:.2f}s "
                    f"elapsed={elapsed_min:.1f}m"
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
        "time_control": {
            "base_seconds": args.tc_base_seconds,
            "increment_seconds": args.tc_increment_seconds,
            "label": f"{args.tc_base_seconds:g}+{args.tc_increment_seconds:g}",
            "style": "Fishtest LTC",
        },
        "multipv": args.multipv,
        "good_move_cp": args.good_move_cp,
        "choice_temperature_cp": args.choice_temperature_cp,
        "book_prefix_plies": args.book_plies,
        "training_depth": "no_artificial_ply_cap"
        if args.max_plies == 0
        else f"emergency_cap_{args.max_plies}",
        "max_plies": args.max_plies,
        "seed": args.seed,
        "learning_policy": {
            "win": f"accepted only if eval >= {args.learn_min_cp}cp and gap <= {args.win_learn_gap_cp}cp",
            "draw": f"accepted only if eval >= {args.learn_min_cp}cp and gap <= {args.draw_learn_gap_cp}cp",
            "loss": "never learned",
            "existing_book_prefix": "never re-learned merely for replay",
            "depth_limit": "none; acceptance is based on result/quality filters rather than book depth",
            "quality_gap_points": {"<=2": 20, "<=5": 16, "<=10": 12, "<=20": 8},
            "result_factor": {"win": 1.50, "draw": 0.60, "loss": 0.0},
            "depth_decay": "weighting only, not rejection: max(0.30, exp(-max(0, ply-40)/140))",
        },
    }

    save_updates(updates_path, dict(updates), metadata)
    summary = {
        **metadata,
        "results": stats,
        "learning_decisions": dict(sorted(learning_stats.items())),
        "unique_learned_moves": len(updates),
        "total_plies": total_plies,
        "total_engine_plies": total_engine_plies,
        "total_search_seconds": round(total_search_seconds, 3),
        "total_searched_nodes": total_searched_nodes,
        "avg_search_ms_per_engine_ply": round(
            total_search_seconds / total_engine_plies * 1000.0, 3
        )
        if total_engine_plies
        else None,
        "elapsed_seconds": round(elapsed, 3),
        "games_per_hour": round(completed_games / elapsed * 3600, 2)
        if elapsed
        else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
