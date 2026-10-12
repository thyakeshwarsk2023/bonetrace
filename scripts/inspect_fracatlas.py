#!/usr/bin/env python3
"""Read-only inventory of an extracted FracAtlas directory.

This is a preflight inventory, not the scientific data audit. It deliberately
reports structure and metadata without renaming, moving, or modifying files.
Uses only Python's standard library; Pillow is optional for image verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def print_tree(root: Path, max_depth: int = 3) -> None:
    print(f"\n[DIRECTORY TREE] {root}")
    print(root.name + "/")
    try:
        children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        print(f"  [ERROR] Cannot read directory: {exc}")
        return

    def walk(items: list[Path], prefix: str, depth: int) -> None:
        for i, item in enumerate(items):
            last = i == len(items) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if item.is_dir() else ""
            print(prefix + branch + item.name + suffix)
            if item.is_dir() and depth < max_depth:
                try:
                    sub = sorted(item.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                    # Avoid dumping hundreds of images in the tree.
                    if item.name.lower() == "images" and len(sub) > 12:
                        for image in sub[:5]:
                            print(prefix + ("    " if last else "│   ") + "├── " + image.name)
                        print(prefix + ("    " if last else "│   ") + f"└── ... ({len(sub) - 5} more entries)")
                    else:
                        walk(sub[:30], prefix + ("    " if last else "│   "), depth + 1)
                        if len(sub) > 30:
                            print(prefix + ("    " if last else "│   ") + f"└── ... ({len(sub) - 30} more entries)")
                except OSError as exc:
                    print(prefix + ("    " if last else "│   ") + f"[ERROR] {exc}")
    walk(children, "", 1)


def inspect_csv(path: Path) -> None:
    print(f"\n[CSV] {path.relative_to(path.parents[1]) if len(path.parents) > 1 else path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            counts = 0
            samples: list[dict[str, str]] = []
            for row in reader:
                counts += 1
                if len(samples) < 3:
                    samples.append(row)
        print(f"  rows (excluding header): {counts}")
        print(f"  columns ({len(headers)}): {headers}")
        for i, row in enumerate(samples, 1):
            # Truncate long cell values to keep output manageable.
            compact = {k: (v[:100] + "…" if isinstance(v, str) and len(v) > 100 else v) for k, v in row.items()}
            print(f"  sample row {i}: {compact}")
    except Exception as exc:
        print(f"  [ERROR] Could not read CSV: {type(exc).__name__}: {exc}")


def inspect_json(path: Path) -> None:
    print(f"\n[JSON] {path.relative_to(path.parents[1]) if len(path.parents) > 1 else path}")
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            payload: Any = json.load(f)
        if isinstance(payload, dict):
            print(f"  root keys: {list(payload.keys())}")
            for key in ("images", "annotations", "categories"):
                value = payload.get(key)
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} entries")
                    if value and isinstance(value[0], dict):
                        print(f"  {key}[0] keys: {list(value[0].keys())}")
            if "info" in payload and isinstance(payload["info"], dict):
                print(f"  info keys: {list(payload['info'].keys())}")
        elif isinstance(payload, list):
            print(f"  root list: {len(payload)} entries")
            if payload and isinstance(payload[0], dict):
                print(f"  first entry keys: {list(payload[0].keys())}")
        else:
            print(f"  root type: {type(payload).__name__}")
    except Exception as exc:
        print(f"  [ERROR] Could not read JSON: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/FracAtlas", help="Path to extracted FracAtlas root")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum folder-tree depth (default: 3)")
    args = parser.parse_args()
    root = Path(args.data_dir).expanduser().resolve()

    print("FracAtlas read-only preflight inventory")
    print("This script does NOT validate research labels, detect duplicates, or create splits.")
    print(f"Root: {root}")
    if not root.exists():
        print("[ERROR] Dataset directory does not exist. Extract the archive to this path first.")
        return 2
    if not root.is_dir():
        print("[ERROR] --data-dir must point to a directory.")
        return 2

    print_tree(root, max_depth=max(1, args.max_depth))
    files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    ext_counts = Counter(p.suffix.lower() or "[no extension]" for p in images)
    print("\n[FILE COUNTS]")
    print(f"  total files: {len(files)}")
    print(f"  image files by extension: {dict(sorted(ext_counts.items()))}")
    print(f"  total image files: {len(images)}")
    print(f"  CSV files: {len([p for p in files if p.suffix.lower() == '.csv'])}")
    print(f"  JSON files: {len([p for p in files if p.suffix.lower() == '.json'])}")

    csvs = sorted((p for p in files if p.suffix.lower() == ".csv"), key=lambda p: (p.name.lower(), str(p).lower()))
    for path in csvs[:10]:
        inspect_csv(path)
    if len(csvs) > 10:
        print(f"\n[NOTE] Skipped {len(csvs) - 10} additional CSV files to limit output.")

    jsons = sorted((p for p in files if p.suffix.lower() == ".json"), key=lambda p: (p.name.lower(), str(p).lower()))
    # Annotation files may be large; inspect up to 8 JSON files only.
    for path in jsons[:8]:
        inspect_json(path)
    if len(jsons) > 8:
        print(f"\n[NOTE] Skipped {len(jsons) - 8} additional JSON files to limit output.")

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print("\n[IMAGE OPEN TEST] Pillow is not installed; image decoding was not tested.")
    else:
        tested = 0
        failures: list[tuple[str, str]] = []
        for path in images:
            if tested >= 50:
                break
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                failures.append((str(path.relative_to(root)), f"{type(exc).__name__}: {exc}"))
            tested += 1
        print(f"\n[IMAGE OPEN TEST] checked first {tested} image files (cap: 50)")
        print(f"  failures: {len(failures)}")
        for name, error in failures[:10]:
            print(f"  - {name}: {error}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")

    print("\nPreflight complete. Treat counts as inventory only; scientific audit is a separate step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
