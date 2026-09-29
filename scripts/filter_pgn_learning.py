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
    return 0


def accepted_weight(result: str, color: chess.Color, gap_cp: int, ply: int) -> int:
    perspective = result_for_color(result, color)

    # Strict policy: never learn a continuation played by the losing side.
    if perspective == "loss":
        return 0

    # Draws must be very close to the engine's best move. Winners are allowed
    # a slightly broader but still strong <=20 cp envelope.
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
    p = argparse.ArgumentParser(description="Build strict result-aware Polyglot learning from self-play PGN")
    p.add_argument("--pgn", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    updates: dict[tuple[int, int], int] = defaultdict(int)
    games = accepted = rejected_loss = rejected_quality = seed_plies = 0

    with open(args.pgn, "r", encoding="utf-8") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            games += 1
            result = game.headers.get("Result", "1/2-1/2")
            board = game.board()

            for node in game.mainline():
                move = node.move
                comment = node.comment or ""
                mover = board.turn
                ply = board.ply()

                if "seed-book" in comment:
                    seed_plies += 1
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
        "policy": "strict-result-aware-v1",
        "games": games,
        "accepted_plies": accepted,
        "rejected_losing_side_plies": rejected_loss,
        "rejected_quality_plies": rejected_quality,
        "seed_book_plies_ignored": seed_plies,
        "winner_max_gap_cp": 20,
        "draw_max_gap_cp": 10,
        "loser_moves": "excluded",
    }
    save_updates(Path(args.out), dict(updates), metadata)
    print(metadata)
    print(f"unique learned moves: {len(updates)}")


if __name__ == "__main__":
    main()
