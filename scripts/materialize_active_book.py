from __future__ import annotations

import argparse
from pathlib import Path

from polyglot_utils import build_enriched_book, load_updates


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuild the active recursive Chess960 Polyglot book from the original Nimas seed plus cumulative learning"
    )
    p.add_argument("--seed-book", required=True)
    p.add_argument("--updates", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    updates = load_updates(args.updates)
    build_enriched_book(args.seed_book, args.out, updates)
    out = Path(args.out)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("Active book materialization produced an empty file")
    print(f"Materialized active book: {out} ({out.stat().st_size} bytes), learned entries={len(updates)}")


if __name__ == "__main__":
    main()
