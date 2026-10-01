from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from polyglot_utils import (
    build_enriched_book,
    load_suppressions,
    load_updates,
    merge_updates,
    save_suppressions,
    save_updates,
)


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
    p = argparse.ArgumentParser(description="Apply Omega-vs-Stockfish book actions to the newest cumulative state")
    p.add_argument("--seed-book", required=True)
    p.add_argument("--prior-updates", required=True)
    p.add_argument("--prior-suppressions")
    p.add_argument("--positive-updates", required=True)
    p.add_argument("--remove-pairs", required=True)
    p.add_argument("--rehabilitate-pairs", required=True)
    p.add_argument("--match-summary", required=True)
    p.add_argument("--prior-book-summary")
    p.add_argument("--history-in")
    p.add_argument("--cumulative-out", required=True)
    p.add_argument("--suppressions-out", required=True)
    p.add_argument("--enriched-out", required=True)
    p.add_argument("--book-summary-out", required=True)
    p.add_argument("--history-out", required=True)
    args = p.parse_args()

    prior = load_updates(args.prior_updates)
    positive = load_updates(args.positive_updates)
    suppressions = load_suppressions(args.prior_suppressions)
    removals = load_suppressions(args.remove_pairs)
    rehabilitations = load_suppressions(args.rehabilitate_pairs)

    cumulative = merge_updates(prior, positive)
    suppressions |= removals
    suppressions -= rehabilitations
    for pair in suppressions:
        cumulative.pop(pair, None)

    save_updates(
        args.cumulative_out,
        cumulative,
        {
            "source": "Omega-vs-latest-Stockfish adversarial learning",
            "prior_entries": len(prior),
            "positive_entries": len(positive),
            "active_suppressions": len(suppressions),
            "cumulative_entries": len(cumulative),
        },
    )
    save_suppressions(
        args.suppressions_out,
        suppressions,
        {
            "source": "Omega-vs-latest-Stockfish adversarial learning",
            "active_suppressions": len(suppressions),
        },
    )
    build_enriched_book(args.seed_book, args.enriched_out, cumulative, suppressions)

    match = json.loads(Path(args.match_summary).read_text(encoding="utf-8"))
    if args.prior_book_summary and Path(args.prior_book_summary).exists():
        book_summary = json.loads(Path(args.prior_book_summary).read_text(encoding="utf-8"))
    else:
        book_summary = {}
    book_summary["cumulative_learned_moves"] = len(cumulative)
    book_summary["suppressed_book_moves"] = len(suppressions)
    book_summary["enriched_book_bytes"] = Path(args.enriched_out).stat().st_size
    book_summary["latest_stockfish_match"] = match
    Path(args.book_summary_out).write_text(json.dumps(book_summary, indent=2) + "\n", encoding="utf-8")

    if args.history_in and Path(args.history_in).exists():
        history = json.loads(Path(args.history_in).read_text(encoding="utf-8"))
    else:
        history = {"format": 1, "runs": []}
    runs = list(history.get("runs", []))
    runs.append(match)
    if len(runs) > 200:
        runs = runs[-200:]
    agg_w = sum(int(r["wdl_omega"]["wins"]) for r in runs)
    agg_d = sum(int(r["wdl_omega"]["draws"]) for r in runs)
    agg_l = sum(int(r["wdl_omega"]["losses"]) for r in runs)
    elo, half, score = elo_from_wdl(agg_w, agg_d, agg_l)
    history = {
        "format": 1,
        "opponent_policy": "latest official Stockfish main at each run; no Stockfish book",
        "omega_policy": "latest Omega main with unlimited-ply active recursive book",
        "caveat": "Aggregate spans changing Stockfish/Omega commits and is descriptive, not a controlled single-version Elo estimate.",
        "aggregate": {
            "games": agg_w + agg_d + agg_l,
            "wins": agg_w,
            "draws": agg_d,
            "losses": agg_l,
            "score_percent": round(score, 4),
            "descriptive_elo": round(elo, 3),
            "elo_95ci_half_width": round(half, 3),
        },
        "runs": runs,
    }
    Path(args.history_out).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "prior_entries": len(prior),
        "positive_entries": len(positive),
        "active_suppressions": len(suppressions),
        "cumulative_entries": len(cumulative),
        "book_bytes": Path(args.enriched_out).stat().st_size,
        "aggregate_benchmark": history["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
