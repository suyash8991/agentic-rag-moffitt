import json
import os
from pathlib import Path
from collections import defaultdict


def title_case_name_parts(parts):
    return " ".join(p.capitalize() for p in parts if p)


def parse_filename_to_name(filename: str):
    # Expect format: first-last.json (last may contain hyphens)
    stem = Path(filename).stem
    if stem in {"summary", ".gitkeep"}:
        return None
    tokens = stem.split("-")
    if len(tokens) < 2:
        return None
    first = tokens[0]
    last_tokens = tokens[1:]
    return first, last_tokens


def build_alias_map(processed_dir: Path):
    files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]

    entries = []
    first_counts = defaultdict(int)
    last_counts = defaultdict(int)

    for f in files:
        parsed = parse_filename_to_name(f)
        if not parsed:
            continue
        first, last_tokens = parsed
        first_l = first.lower()
        last_key = " ".join(last_tokens).lower()
        first_counts[first_l] += 1
        last_counts[last_key] += 1
        entries.append((first, last_tokens))

    alias_map = {}

    def try_add(key: str, value: str):
        k = key.strip().lower()
        if not k:
            return
        # Avoid ambiguous collisions: don't overwrite existing with different value
        if k in alias_map and alias_map[k] != value:
            return
        alias_map[k] = value

    for first, last_tokens in entries:
        canonical = f"{first.capitalize()} {title_case_name_parts(last_tokens)}"
        first_l = first.lower()
        last_key = " ".join(last_tokens).lower()
        last_str = title_case_name_parts(last_tokens)

        full_lc = f"{first_l} {' '.join(last_tokens).lower()}".strip()

        # Always include full name
        try_add(full_lc, canonical)

        # Titles and suffixes for full name
        try_add(f"dr. {full_lc}", canonical)
        try_add(f"dr {full_lc}", canonical)
        try_add(f"{full_lc} phd", canonical)
        try_add(f"{full_lc} ph.d.", canonical)

        # Last-name-only variants (if unambiguous)
        if last_counts[last_key] == 1:
            try_add(last_key, canonical)
            try_add(f"dr. {last_key}", canonical)
            try_add(f"dr {last_key}", canonical)
            try_add(f"{last_key} phd", canonical)
            try_add(f"{last_key} ph.d.", canonical)

        # First-name-only variant (if unambiguous)
        if first_counts[first_l] == 1:
            try_add(first_l, canonical)

    return alias_map


def main():
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"
    out_dir = repo_root / "backend" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "researcher_aliases.json"

    alias_map = build_alias_map(processed_dir)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(alias_map)} aliases to {out_path}")


if __name__ == "__main__":
    main()

