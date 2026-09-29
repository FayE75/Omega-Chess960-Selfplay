from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyglot_utils import build_enriched_book, load_updates, merge_updates, save_updates, write_delta_book


def main() -> None:
    p = argparse.ArgumentParser(description="Merge self-play learning into a Polyglot seed book")
    p.add_argument("--seed-book", required=True)
    p.add_argument("--run-updates", required=True)
    p.add_argument("--prior-updates")
    p.add_argument("--cumulative-out", required=True)
    p.add_argument("--enriched-out", required=True)
    p.add_argument("--delta-out", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    prior = load_updates(args.prior_updates)
    run = load_updates(args.run_updates)
    cumulative = merge_updates(prior, run)

    save_updates(
        args.cumulative_out,
        cumulative,
        {
            "prior_entries": len(prior),
            "run_entries": len(run),
            "cumulative_entries": len(cumulative),
        },
    )
    write_delta_book(args.delta_out, run)
    build_enriched_book(args.seed_book, args.enriched_out, cumulative)

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prior_learned_moves"] = len(prior)
    summary["run_learned_moves"] = len(run)
    summary["cumulative_learned_moves"] = len(cumulative)
    summary["enriched_book_bytes"] = Path(args.enriched_out).stat().st_size
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
