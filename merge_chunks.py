#!/usr/bin/env python3
"""
merge_chunks.py — Reassemble candidates.jsonl from data_chunks/

Usage:
  python merge_chunks.py

  # Or merge specific chunks for partial testing:
  python merge_chunks.py --parts 1 2 3 --out candidates_30k.jsonl
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(
        description="Merge candidate data chunks into a single JSONL file."
    )
    parser.add_argument(
        "--chunks-dir", type=str, default="./data_chunks",
        help="Directory containing the chunk files (default: ./data_chunks)"
    )
    parser.add_argument(
        "--parts", type=int, nargs="*", default=None,
        help="Specific part numbers to merge (e.g., --parts 1 2 3). "
             "Default: merge all parts."
    )
    parser.add_argument(
        "--out", type=str, default="./candidates.jsonl",
        help="Output file path (default: ./candidates.jsonl)"
    )
    args = parser.parse_args()

    # Find chunk files
    if args.parts:
        chunk_files = [
            os.path.join(args.chunks_dir, f"candidates_part_{p:02d}.jsonl")
            for p in args.parts
        ]
        missing = [f for f in chunk_files if not os.path.exists(f)]
        if missing:
            print(f"ERROR: Missing chunk files: {missing}")
            return
    else:
        chunk_files = sorted(glob.glob(
            os.path.join(args.chunks_dir, "candidates_part_*.jsonl")
        ))

    if not chunk_files:
        print(f"ERROR: No chunk files found in {args.chunks_dir}/")
        return

    print(f"Merging {len(chunk_files)} chunks → {args.out}")

    total_lines = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for chunk_path in chunk_files:
            chunk_lines = 0
            with open(chunk_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    chunk_lines += 1
            total_lines += chunk_lines
            size_mb = os.path.getsize(chunk_path) / 1024 / 1024
            print(f"  ✅ {os.path.basename(chunk_path)}: "
                  f"{chunk_lines:,} candidates ({size_mb:.1f}MB)")

    out_size = os.path.getsize(args.out) / 1024 / 1024
    print(f"\n✅ Done! {args.out}: {total_lines:,} candidates ({out_size:.1f}MB)")


if __name__ == "__main__":
    main()
