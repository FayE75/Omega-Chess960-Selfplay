from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn

from polyglot_utils import encode_polyglot_move, save_updates

GAP_RE = re.compile(r"\bgap=(\d+)cp\b")


def result_for_color(result: str, color: chess.Color) -> str:
    if result == "1-0":
        return "win" if color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if color == chess.BLACK else "loss"
    if result == "1/2-1/2":
        return "draw"
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


def accepted_weight(result: str, color: chess.Color, gap_cp: int, ply: int) -> int:
    perspective = result_for_color(result, color)

    if perspective in {"unknown", "loss"}:
        return 0

    if perspective == "draw" and gap_cp > 10:
        return 0
    if perspective == "win" and gap_cp > 20:
        return 0

    base = quality_points(gap_cp)
    if not base:
        return 0

    result_factor = 1.35 if perspective == "win" else 0.90
    depth_factor = max(0.30, math.exp(-max(0, ply - 40) / 140.0))
    return max(1, int(round(base * result_factor * depth_factor)))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build strict result-aware Polyglot learning from self-play PGN"
    )
    p.add_argument("--pgn", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    updates: dict[tuple[int, int], int] = defaultdict(int)
    games = accepted = rejected_loss = rejected_quality = book_plies = incomplete_games = 0

    with open(args.pgn, "r", encoding="utf-8") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result not in {"1-0", "0-1", "1/2-1/2"}:
                incomplete_games += 1
                continue

            games += 1
            board = game.board()

            for node in game.mainline():
                move = node.move
                comment = node.comment or ""
                mover = board.turn
                ply = board.ply()

                # Existing active-book prefix is context only. Never add it again
                # simply because it was replayed at the beginning of self-play.
                if "active-book" in comment or "seed-book" in comment:
                    book_plies += 1
                    board.push(move)
                    continue

                match = GAP_RE.search(comment)
                if not match:
                    board.push(move)
                    continue

                gap_cp = int(match.group(1))
                perspective = result_for_color(result, mover)
                weight = accepted_weight(result, mover, gap_cp, ply)

                if weight > 0:
                    key = chess.polyglot.zobrist_hash(board)
                    raw_move = encode_polyglot_move(board, move)
                    updates[(key, raw_move)] += weight
                    accepted += 1
                elif perspective == "loss":
                    rejected_loss += 1
                else:
                    rejected_quality += 1

                board.push(move)

    metadata = {
        "policy": "strict-result-aware-v2",
        "games": games,
        "incomplete_games_ignored": incomplete_games,
        "accepted_plies": accepted,
        "rejected_losing_side_plies": rejected_loss,
        "rejected_quality_plies": rejected_quality,
        "active_book_prefix_plies_ignored": book_plies,
        "winner_max_gap_cp": 20,
        "draw_max_gap_cp": 10,
        "loser_moves": "excluded",
        "book_depth_limit": "none",
    }
    save_updates(Path(args.out), dict(updates), metadata)
    print(metadata)
    print(f"unique learned moves: {len(updates)}")


if __name__ == "__main__":
    main()
