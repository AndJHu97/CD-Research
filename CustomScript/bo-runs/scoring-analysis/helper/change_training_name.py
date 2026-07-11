#!/usr/bin/env python3
"""Rewrite the prefix in the Name column (text before the first '-').

Names like ``1BMQ-MNO`` become ``4g5j-MNO`` when the prefix is remapped.

Usage:
    # Replace one prefix
    python change_training_name.py training.csv --old 1BMQ --new 4g5j -o training_renamed.csv

    # Replace several prefixes from a mapping CSV (columns: old_prefix, new_prefix)
    python change_training_name.py training.csv --map prefix_map.csv -o training_renamed.csv

    # Change every prefix to the same value (keeps suffix after '-')
    python change_training_name.py training.csv --new 4g5j --all -o training_renamed.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _split_name(name: str) -> tuple[str, str | None]:
    text = str(name).strip()
    if "-" not in text:
        return text, None
    prefix, suffix = text.split("-", 1)
    return prefix.strip(), suffix


def _join_name(prefix: str, suffix: str | None) -> str:
    if suffix is None:
        return prefix
    return f"{prefix}-{suffix}"


def _prefix_matches(prefix: str, old_prefix: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return prefix == old_prefix
    return prefix.lower() == old_prefix.lower()


def rename_prefix(
    name: object,
    old_prefix: str,
    new_prefix: str,
    *,
    case_sensitive: bool = False,
) -> str:
    prefix, suffix = _split_name(name)
    if not _prefix_matches(prefix, old_prefix, case_sensitive):
        return str(name).strip()
    return _join_name(new_prefix, suffix)


def rename_all_prefixes(name: object, new_prefix: str) -> str:
    prefix, suffix = _split_name(name)
    if suffix is None:
        return new_prefix
    return _join_name(new_prefix, suffix)


def load_prefix_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    old_col = next(
        (c for c in df.columns if c.lower() in {"old_prefix", "old", "from", "source"}),
        None,
    )
    new_col = next(
        (c for c in df.columns if c.lower() in {"new_prefix", "new", "to", "target"}),
        None,
    )
    if old_col is None or new_col is None:
        raise ValueError(
            f"Mapping CSV must contain old/new prefix columns. Found: {list(df.columns)}"
        )

    mapping: dict[str, str] = {}
    for old_value, new_value in zip(df[old_col], df[new_col], strict=False):
        old_key = str(old_value).strip()
        new_value_clean = str(new_value).strip()
        if not old_key or not new_value_clean or pd.isna(old_value) or pd.isna(new_value):
            continue
        mapping[old_key.lower()] = new_value_clean
    if not mapping:
        raise ValueError(f"No usable prefix mappings found in {path}")
    return mapping


def apply_prefix_map(name: object, mapping: dict[str, str]) -> str:
    prefix, suffix = _split_name(name)
    new_prefix = mapping.get(prefix.lower())
    if new_prefix is None:
        return str(name).strip()
    return _join_name(new_prefix, suffix)


def change_training_names(
    input_path: Path,
    output_path: Path,
    *,
    column: str = "Name",
    old_prefix: str | None = None,
    new_prefix: str | None = None,
    replace_all: bool = False,
    map_path: Path | None = None,
    case_sensitive: bool = False,
) -> tuple[int, int]:
    df = pd.read_csv(input_path, low_memory=False)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    original = df[column].astype(str)
    updated = original.copy()

    if map_path is not None:
        mapping = load_prefix_map(map_path)
        updated = original.map(lambda value: apply_prefix_map(value, mapping))
    elif replace_all:
        if not new_prefix:
            raise ValueError("--new is required with --all")
        updated = original.map(lambda value: rename_all_prefixes(value, new_prefix))
    else:
        if not old_prefix or not new_prefix:
            raise ValueError("Provide both --old and --new, or use --map / --all")
        updated = original.map(
            lambda value: rename_prefix(
                value,
                old_prefix,
                new_prefix,
                case_sensitive=case_sensitive,
            )
        )

    changed_mask = original != updated
    df[column] = updated
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return int(changed_mask.sum()), len(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Name prefixes before '-' in a training CSV."
    )
    parser.add_argument("input_csv", help="Input training CSV")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV (default: <input>_renamed.csv)",
    )
    parser.add_argument(
        "--column",
        default="Name",
        help="Column to rewrite (default: Name)",
    )
    parser.add_argument(
        "--old",
        dest="old_prefix",
        default=None,
        help="Current prefix before '-' (e.g. 1BMQ)",
    )
    parser.add_argument(
        "--new",
        dest="new_prefix",
        default=None,
        help="Replacement prefix (e.g. 4g5j)",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        default=None,
        help="CSV with old_prefix/new_prefix columns for multiple remaps",
    )
    parser.add_argument(
        "--all",
        dest="replace_all",
        action="store_true",
        help="Replace every prefix with --new, keeping the suffix after '-'",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match old prefix with exact case (default: case-insensitive)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.is_file():
        print(f"Error: input CSV not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_renamed.csv")
    )

    mode_count = sum(
        [
            bool(args.map_path),
            bool(args.replace_all),
            bool(args.old_prefix or args.new_prefix),
        ]
    )
    if mode_count != 1:
        print(
            "Error: choose exactly one mode:\n"
            "  --old OLD --new NEW\n"
            "  --map prefix_map.csv\n"
            "  --new NEW --all",
            file=sys.stderr,
        )
        return 1

    try:
        changed, total = change_training_names(
            input_path,
            output_path,
            column=args.column,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            replace_all=args.replace_all,
            map_path=Path(args.map_path) if args.map_path else None,
            case_sensitive=args.case_sensitive,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {total} rows to {output_path}")
    print(f"Updated {changed} row(s) in column '{args.column}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
