"""
Example utility for reading the pickle dump produced by the Rust SLO client.

Usage:
  python slo/parse_elapsed_dump.py --dump-path elapsed.pkl --head 5 --summary
"""

import argparse
import pickle
from typing import Iterator, Mapping, Sequence


def iter_timelines(path: str) -> Iterator[Mapping[str, object]]:
    """Yield timelines from the pickle stream until EOF."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read elapsed_ms timelines written by the Rust SLO client."
    )
    parser.add_argument("--dump-path", required=True, help="Path to the pickle file.")
    parser.add_argument(
        "--head",
        type=int,
        default=3,
        help="Print the first N timelines for inspection (default: 3).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print simple aggregate stats (count, tokens, max length).",
    )
    args = parser.parse_args()

    head_left = args.head
    total_reqs = 0
    total_tokens = 0
    max_len = 0
    max_req = None

    for rec in iter_timelines(args.dump_path):
        total_reqs += 1
        elapsed: Sequence[float] = rec.get("elapsed_ms", [])  # type: ignore[assignment]
        total_tokens += len(elapsed)
        if len(elapsed) > max_len:
            max_len = len(elapsed)
            max_req = rec.get("request_id")

        if head_left > 0:
            req_id = rec.get("request_id")
            print(f"[{total_reqs}] request_id={req_id} tokens={len(elapsed)}")
            print(f"  first 5 elapsed_ms: {list(elapsed[:5])}")
            print(f"  last elapsed_ms: {elapsed[-1] if elapsed else 'n/a'}")
            head_left -= 1

    if args.summary:
        avg_tokens = total_tokens / total_reqs if total_reqs else 0
        print("\nSummary")
        print(f"- records: {total_reqs}")
        print(f"- total tokens: {total_tokens}")
        print(f"- avg tokens/req: {avg_tokens:.2f}")
        print(f"- max tokens: {max_len} (request_id={max_req})")


if __name__ == "__main__":
    main()
