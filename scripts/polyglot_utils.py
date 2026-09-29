from __future__ import annotations

import gzip
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import chess

ENTRY = struct.Struct(">QHHI")
UpdateMap = Dict[Tuple[int, int], int]
POLYGLOT_MAX_WEIGHT = 65535


def encode_polyglot_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a python-chess move into the 16-bit Polyglot move format."""
    # Polyglot represents castling as king-from -> rook-from. python-chess's
    # helper performs that conversion for standard and Chess960 boards.
    poly_move = board._to_chess960(move) if board.is_castling(move) else move
    promo = 0
    if poly_move.promotion:
        promo = {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
        }.get(poly_move.promotion, 0)
    return poly_move.to_square | (poly_move.from_square << 6) | (promo << 12)


def load_updates(path: str | Path | None) -> UpdateMap:
    if not path:
        return {}
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    updates: UpdateMap = {}
    for item in payload.get("updates", []):
        updates[(int(item["key"], 16), int(item["move"]))] = int(item["weight"])
    return updates


def save_updates(path: str | Path, updates: UpdateMap, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 2,
        "metadata": metadata or {},
        "updates": [
            {"key": f"{key:016x}", "move": move, "weight": int(weight)}
            for (key, move), weight in sorted(updates.items())
            if weight > 0
        ],
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def merge_updates(*maps: UpdateMap) -> UpdateMap:
    merged: UpdateMap = defaultdict(int)
    for mapping in maps:
        for key_move, weight in mapping.items():
            merged[key_move] += int(weight)
    return dict(merged)


def iter_seed_groups(path: str | Path) -> Iterator[tuple[int, list[tuple[int, int, int]]]]:
    """Yield (key, [(raw_move, weight, learn), ...]) from a sorted Polyglot book."""
    with open(path, "rb") as fh:
        current_key = None
        group: list[tuple[int, int, int]] = []
        while chunk := fh.read(ENTRY.size):
            if len(chunk) != ENTRY.size:
                raise ValueError(f"Truncated Polyglot entry in {path}")
            key, move, weight, learn = ENTRY.unpack(chunk)
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, group
                current_key, group = key, []
            group.append((move, weight, learn))
        if current_key is not None:
            yield current_key, group


def normalize_weights(entries: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Fit arbitrary cumulative weights into Polyglot uint16 while preserving ratios."""
    if not entries:
        return []
    max_weight = max(weight for _move, weight, _learn in entries)
    if max_weight <= POLYGLOT_MAX_WEIGHT:
        return [(move, max(1, int(weight)), learn) for move, weight, learn in entries]

    scale = POLYGLOT_MAX_WEIGHT / float(max_weight)
    return [
        (move, max(1, min(POLYGLOT_MAX_WEIGHT, int(round(weight * scale)))), learn)
        for move, weight, learn in entries
    ]


def write_delta_book(path: str | Path, updates: UpdateMap) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (key, move), weight in updates.items():
        if weight > 0:
            by_key[key].append((move, int(weight), 0))

    with open(path, "wb") as out:
        for key in sorted(by_key):
            for move, weight, learn in sorted(normalize_weights(by_key[key]), key=lambda x: x[0]):
                out.write(ENTRY.pack(key, move, weight, learn))


def build_enriched_book(seed_path: str | Path, out_path: str | Path, updates: UpdateMap) -> None:
    """Stream-merge sparse learned weights into a large seed Polyglot book."""
    by_key: dict[int, dict[int, int]] = defaultdict(dict)
    for (key, move), weight in updates.items():
        if weight > 0:
            by_key[key][move] = by_key[key].get(move, 0) + int(weight)

    update_keys = sorted(by_key)
    ui = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_group(out, key: int, entries: list[tuple[int, int, int]]) -> None:
        for move, weight, learn in sorted(normalize_weights(entries), key=lambda x: x[0]):
            out.write(ENTRY.pack(key, move, weight, learn))

    with open(out_path, "wb") as out:
        for seed_key, seed_entries in iter_seed_groups(seed_path):
            while ui < len(update_keys) and update_keys[ui] < seed_key:
                key = update_keys[ui]
                learned_only = [(move, weight, 0) for move, weight in by_key[key].items() if weight > 0]
                write_group(out, key, learned_only)
                ui += 1

            additions = by_key.get(seed_key, {})
            seen: set[int] = set()
            merged_entries: list[tuple[int, int, int]] = []
            for move, weight, learn in seed_entries:
                learned = additions.get(move, 0)
                merged_entries.append((move, int(weight) + int(learned), learn))
                seen.add(move)
            for move, learned in additions.items():
                if move not in seen and learned > 0:
                    merged_entries.append((move, int(learned), 0))
            write_group(out, seed_key, merged_entries)

            if ui < len(update_keys) and update_keys[ui] == seed_key:
                ui += 1

        while ui < len(update_keys):
            key = update_keys[ui]
            learned_only = [(move, weight, 0) for move, weight in by_key[key].items() if weight > 0]
            write_group(out, key, learned_only)
            ui += 1
