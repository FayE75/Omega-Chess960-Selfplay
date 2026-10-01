from __future__ import annotations

import argparse
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

from polyglot_utils import encode_polyglot_move, save_suppressions, save_updates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired Chess960 Omega-vs-latest-Stockfish LTC match with adversarial book learning"
    )
    p.add_argument("--omega", required=True)
    p.add_argument("--stockfish", required=True)
    p.add_argument("--omega-book", required=True)
    p.add_argument("--out-dir", default="output-adversarial")
    p.add_argument("--games", type=int, default=8, help="Must be an even number; games are color-paired")
    p.add_argument("--tc-base-seconds", type=float, default=60.0)
    p.add_argument("--tc-increment-seconds", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--omega-id", default="Omega-main")
    p.add_argument("--stockfish-id", default="Stockfish-main")
    p.add_argument("--book-weight-power", type=float, default=0.55)
    p.add_argument("--multipv", type=int, default=3)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--resign-cp", type=int, default=900)
    p.add_argument("--resign-confirmations", type=int, default=4)
    p.add_argument("--draw-cp", type=int, default=20)
    p.add_argument("--draw-after-ply", type=int, default=100)
    p.add_argument("--draw-confirmations", type=int, default=16)
    return p.parse_args()


def configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)


def choose_book_move(
    reader: chess.polyglot.MemoryMappedReader,
    board: chess.Board,
    rng: random.Random,
    power: float,
) -> tuple[chess.Move, int] | None:
    entries = [entry for entry in reader.find_all(board) if entry.move in board.legal_moves]
    if not entries:
        return None
    weights = [max(1.0, float(entry.weight)) ** power for entry in entries]
    entry = rng.choices(entries, weights=weights, k=1)[0]
    return entry.move, int(entry.weight)


def analyse_best(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    clocks: dict[chess.Color, float],
    increment: float,
    multipv: int,
    game_token: object,
) -> tuple[chess.Move, int, int, float, int, bool]:
    mover = board.turn
    before = clocks[mover]
    limit = chess.engine.Limit(
        white_clock=max(0.001, clocks[chess.WHITE]),
        black_clock=max(0.001, clocks[chess.BLACK]),
        white_inc=max(0.0, increment),
        black_inc=max(0.0, increment),
    )
    wall_started = time.perf_counter()
    info = engine.analyse(
        board,
        limit,
        multipv=max(1, multipv),
        game=game_token,
        info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    wall = time.perf_counter() - wall_started
    infos = info if isinstance(info, list) else [info]
    candidates: list[tuple[chess.Move, int]] = []
    reported = 0.0
    nodes = 0
    for item in infos:
        pv = item.get("pv") or []
        score_obj = item.get("score")
        if pv and score_obj is not None:
            cp = score_obj.pov(mover).score(mate_score=40000)
            if cp is not None and pv[0] in board.legal_moves:
                candidates.append((pv[0], int(cp)))
        reported = max(reported, float(item.get("time", 0.0) or 0.0))
        nodes = max(nodes, int(item.get("nodes", 0) or 0))
    if not candidates:
        raise RuntimeError("Engine returned no legal scored PV move")
    candidates.sort(key=lambda x: x[1], reverse=True)
    move, best_cp = candidates[0]
    second_cp = candidates[1][1] if len(candidates) > 1 else best_cp
    gap = max(0, best_cp - second_cp)
    spent = reported if reported > 0 else wall
    flagged = spent >= before
    clocks[mover] = max(0.0, before - spent)
    if not flagged:
        clocks[mover] += increment
    return move, best_cp, gap, spent, nodes, flagged


def omega_book_reward(result: str, omega_color: chess.Color, ply: int) -> int:
    omega_won = (result == "1-0" and omega_color == chess.WHITE) or (
        result == "0-1" and omega_color == chess.BLACK
    )
    if omega_won:
        base = 16.0
    elif result == "1/2-1/2":
        base = 4.0
    else:
        return 0
    depth_factor = max(0.30, math.exp(-max(0, ply - 40) / 160.0))
    return max(1, int(round(base * depth_factor)))


def stockfish_winner_reward(eval_cp: int, gap_cp: int, ply: int) -> int:
    if eval_cp < -100:
        return 0
    base = 24.0
    eval_factor = 1.0
    if eval_cp >= 300:
        eval_factor = 1.50
    elif eval_cp >= 100:
        eval_factor = 1.25
    confidence_factor = 1.0 + min(max(gap_cp, 0), 80) / 160.0
    depth_factor = max(0.30, math.exp(-max(0, ply - 40) / 160.0))
    return max(1, int(round(base * eval_factor * confidence_factor * depth_factor)))


def omega_result(result: str, omega_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if omega_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if omega_color == chess.BLACK else "loss"
    return "unknown"


def elo_from_wdl(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    n = wins + draws + losses
    if n <= 0:
        return 0.0, float("inf"), 0.0
    score = (wins + 0.5 * draws) / n
    p = min(1.0 - 1e-9, max(1e-9, score))
    elo = 400.0 * math.log10(p / (1.0 - p))
    mean2 = (wins + 0.25 * draws) / n
    var_game = max(0.0, mean2 - score * score)
    se_score = math.sqrt(var_game / n)
    deriv = 400.0 / math.log(10.0) / (p * (1.0 - p))
    half = 1.96 * deriv * se_score
    return elo, half, score * 100.0


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.games % 2:
        raise ValueError("--games must be a positive even number for paired colors")
    if args.tc_base_seconds <= 0 or args.tc_increment_seconds < 0:
        raise ValueError("Invalid time control")

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / "omega-vs-stockfish.pgn"

    positive_updates: dict[tuple[int, int], int] = defaultdict(int)
    remove_pairs: set[tuple[int, int]] = set()
    rehabilitate_pairs: set[tuple[int, int]] = set()
    results = {"win": 0, "draw": 0, "loss": 0}
    by_color = {
        "white": {"win": 0, "draw": 0, "loss": 0},
        "black": {"win": 0, "draw": 0, "loss": 0},
    }
    action_stats = defaultdict(int)
    total_nodes = {"omega": 0, "stockfish": 0}
    total_search_seconds = {"omega": 0.0, "stockfish": 0.0}
    started = time.time()

    omega = chess.engine.SimpleEngine.popen_uci(args.omega)
    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    configure_engine(omega, args.threads, args.hash_mb)
    configure_engine(stockfish, args.threads, args.hash_mb)

    try:
        with chess.polyglot.open_reader(args.omega_book) as book, open(
            pgn_path, "w", encoding="utf-8"
        ) as pgn_out:
            game_no = 0
            for pair_no in range(1, args.games // 2 + 1):
                index = rng.randrange(960)
                for omega_color in (chess.WHITE, chess.BLACK):
                    game_no += 1
                    board = chess.Board.from_chess960_pos(index)
                    game = chess.pgn.Game()
                    game.setup(board)
                    game.headers["Event"] = "Omega vs Latest Stockfish Chess960 LTC"
                    game.headers["Site"] = "GitHub Actions"
                    game.headers["Round"] = str(game_no)
                    game.headers["White"] = args.omega_id if omega_color == chess.WHITE else args.stockfish_id
                    game.headers["Black"] = args.stockfish_id if omega_color == chess.WHITE else args.omega_id
                    game.headers["Variant"] = "Chess960"
                    game.headers["Scharnagl"] = str(index)
                    game.headers["TimeControl"] = f"{args.tc_base_seconds:g}+{args.tc_increment_seconds:g}"
                    game.headers["OmegaBook"] = "unlimited-ply"
                    game.headers["StockfishBook"] = "none"
                    node = game
                    clocks = {
                        chess.WHITE: float(args.tc_base_seconds),
                        chess.BLACK: float(args.tc_base_seconds),
                    }
                    omega_book_moves: list[tuple[int, int, chess.Color, int]] = []
                    stockfish_moves: list[tuple[int, int, chess.Color, int, int, int]] = []
                    advantage_streak = {chess.WHITE: 0, chess.BLACK: 0}
                    draw_streak = 0
                    result = "*"

                    while True:
                        outcome = board.outcome(claim_draw=True)
                        if outcome:
                            result = outcome.result()
                            break
                        if board.ply() >= args.max_plies:
                            result = "1/2-1/2"
                            break

                        mover = board.turn
                        mover_is_omega = mover == omega_color
                        key = chess.polyglot.zobrist_hash(board)
                        ply = board.ply()

                        picked = (
                            choose_book_move(book, board, rng, args.book_weight_power)
                            if mover_is_omega
                            else None
                        )

                        if picked is not None:
                            move, book_weight = picked
                            raw = encode_polyglot_move(board, move)
                            omega_book_moves.append((key, raw, mover, ply))
                            clocks[mover] += args.tc_increment_seconds
                            node = node.add_variation(move)
                            node.comment = f"omega-book weight={book_weight} unlimited-ply"
                            board.push(move)
                            action_stats["omega_book_plies"] += 1
                            continue

                        engine = omega if mover_is_omega else stockfish
                        role = "omega" if mover_is_omega else "stockfish"
                        move, cp, gap, spent, nodes, flagged = analyse_best(
                            engine,
                            board,
                            clocks,
                            args.tc_increment_seconds,
                            args.multipv,
                            (args.seed, game_no, role),
                        )
                        total_nodes[role] += nodes
                        total_search_seconds[role] += spent
                        if flagged:
                            result = "0-1" if mover == chess.WHITE else "1-0"
                            action_stats[f"{role}_time_losses"] += 1
                            break

                        raw = encode_polyglot_move(board, move)
                        if not mover_is_omega:
                            stockfish_moves.append((key, raw, mover, ply, cp, gap))

                        node = node.add_variation(move)
                        node.comment = (
                            f"{role}-search eval={cp/100:.2f} gap={gap}cp "
                            f"search_ms={spent*1000:.1f} nodes={nodes}"
                        )
                        board.push(move)

                        if cp >= args.resign_cp:
                            advantage_streak[mover] += 1
                        else:
                            advantage_streak[mover] = 0
                        if advantage_streak[mover] >= args.resign_confirmations:
                            result = "1-0" if mover == chess.WHITE else "0-1"
                            break

                        if board.ply() >= args.draw_after_ply and abs(cp) <= args.draw_cp:
                            draw_streak += 1
                        else:
                            draw_streak = 0
                        if draw_streak >= args.draw_confirmations:
                            result = "1/2-1/2"
                            break

                    game.headers["Result"] = result
                    perspective = omega_result(result, omega_color)
                    if perspective == "unknown":
                        raise RuntimeError(f"Unexpected result: {result}")
                    results[perspective] += 1
                    color_name = "white" if omega_color == chess.WHITE else "black"
                    by_color[color_name][perspective] += 1

                    if perspective == "loss":
                        for key, raw, _color, _ply in omega_book_moves:
                            remove_pairs.add((key, raw))
                            action_stats["omega_loss_book_moves_marked_for_removal"] += 1
                        for key, raw, _color, ply, cp, gap in stockfish_moves:
                            weight = stockfish_winner_reward(cp, gap, ply)
                            if weight > 0:
                                positive_updates[(key, raw)] += weight
                                rehabilitate_pairs.add((key, raw))
                                action_stats["stockfish_winner_moves_learned"] += 1
                    else:
                        for key, raw, _color, ply in omega_book_moves:
                            reward = omega_book_reward(result, omega_color, ply)
                            if reward > 0:
                                positive_updates[(key, raw)] += reward
                                action_stats[
                                    "omega_win_book_moves_reinforced"
                                    if perspective == "win"
                                    else "omega_draw_book_moves_reinforced"
                                ] += 1

                    print(game, file=pgn_out, end="\n\n")
                    pgn_out.flush()
                    print(
                        f"game {game_no}/{args.games} pair={pair_no} start={index:03d} "
                        f"omega={'white' if omega_color else 'black'} result={perspective} "
                        f"book_moves={len(omega_book_moves)} sf_moves={len(stockfish_moves)}"
                    )
    finally:
        omega.quit()
        stockfish.quit()

    effective_removals = remove_pairs - rehabilitate_pairs
    wins, draws, losses = results["win"], results["draw"], results["loss"]
    elo, elo_half, score_percent = elo_from_wdl(wins, draws, losses)
    elapsed = time.time() - started

    save_updates(
        out_dir / "adversarial_positive_updates.json.gz",
        dict(positive_updates),
        {
            "policy": "omega-vs-latest-stockfish-v1",
            "stockfish_book": "none",
            "omega_book": "unlimited-ply",
            "stockfish_winner_moves": "learned when eval >= -100cp with eval/gap/depth weighting",
            "omega_loss_book_moves": "suppressed for the exact side-to-move Polyglot key+move",
            "omega_win_book_moves": "reinforced",
            "omega_draw_book_moves": "mildly reinforced",
        },
    )
    save_suppressions(
        out_dir / "remove_pairs.json.gz",
        effective_removals,
        {"reason": "Omega lost after using these exact book key-move pairs"},
    )
    save_suppressions(
        out_dir / "rehabilitate_pairs.json.gz",
        rehabilitate_pairs,
        {"reason": "Latest Stockfish used these moves in games it won"},
    )

    summary = {
        "policy": "omega-vs-latest-stockfish-v1",
        "omega": args.omega_id,
        "stockfish": args.stockfish_id,
        "games": args.games,
        "paired_colors": True,
        "time_control": f"{args.tc_base_seconds:g}+{args.tc_increment_seconds:g}",
        "threads_each": args.threads,
        "hash_mb_each": args.hash_mb,
        "omega_book": "unlimited-ply active recursive Polyglot book",
        "stockfish_book": "none",
        "wdl_omega": {"wins": wins, "draws": draws, "losses": losses},
        "by_omega_color": by_color,
        "score_percent": round(score_percent, 4),
        "descriptive_elo": round(elo, 3),
        "elo_95ci_half_width": round(elo_half, 3),
        "positive_update_entries": len(positive_updates),
        "book_moves_suppressed_this_run": len(effective_removals),
        "stockfish_winner_move_entries": len(rehabilitate_pairs),
        "action_counts": dict(sorted(action_stats.items())),
        "search_seconds": {k: round(v, 3) for k, v in total_search_seconds.items()},
        "searched_nodes": total_nodes,
        "elapsed_seconds": round(elapsed, 3),
        "notes": [
            "Elo is descriptive from this paired WDL sample; the confidence interval can be wide for small recurring batches.",
            "Polyglot keys include side to move, so suppressing an Omega move after a loss is inherently color-specific.",
            "Stockfish receives no opening book. Omega may query its active book on every Omega turn with no ply cap.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "### Omega vs latest Stockfish",
                "",
                f"- W/D/L: **{wins} / {draws} / {losses}**",
                f"- Score: **{score_percent:.2f}%**",
                f"- Descriptive Elo: **{elo:+.2f} ± {elo_half:.2f}** (95% CI)",
                "- Omega book: **unlimited ply**",
                "- Stockfish book: **none**",
                f"- Book moves suppressed: **{len(effective_removals)}**",
                f"- Stockfish winning move entries learned: **{len(rehabilitate_pairs)}**",
                f"- Positive book-update entries: **{len(positive_updates)}**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
