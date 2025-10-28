from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    from thefuzz import process as fuzz_process  # type: ignore
except Exception:  # pragma: no cover - library may not be installed yet
    fuzz_process = None


@dataclass
class NormalizationResult:
    canonical: Optional[str]
    method: Optional[str]
    score: Optional[int]


class NameNormalizationService:
    """
    Normalize researcher names using an alias map first, then fuzzy matching.

    - Alias map is loaded from `backend/data/researcher_aliases.json` by default.
    - Canonical names default to the unique values of the alias map, but can be
      supplied via a provider for fresher data (e.g., from a DB cache).
    - Threshold controls fuzzy acceptance (default 85).
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        alias_map_path: Optional[Path] = None,
        canonical_names_provider: Optional[Callable[[], Iterable[str]]] = None,
        threshold: int = 85,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self._alias_map_path = (
            Path(alias_map_path)
            if alias_map_path
            else self._repo_root / "backend" / "data" / "researcher_aliases.json"
        )
        self._threshold = threshold
        self._alias_map: Dict[str, str] = {}
        self._canonical_names: List[str] = []
        self._canonical_names_provider = canonical_names_provider
        self.reload()

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = value

    def reload(self) -> None:
        """Reload alias map and canonical names."""
        if self._alias_map_path.exists():
            with open(self._alias_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # normalize keys to lowercase/trim
                self._alias_map = {str(k).strip().lower(): str(v) for k, v in data.items()}
        else:
            self._alias_map = {}

        if self._canonical_names_provider:
            self._canonical_names = list(self._canonical_names_provider())
        else:
            # Derive canonical names from alias values if provider not set
            seen = set()
            names: List[str] = []
            for v in self._alias_map.values():
                if v not in seen:
                    seen.add(v)
                    names.append(v)
            self._canonical_names = names

    def normalize(self, extracted_name: Optional[str]) -> NormalizationResult:
        """Return canonical name if found, with method and score for observability."""
        if not extracted_name:
            return NormalizationResult(None, None, None)

        query = extracted_name.strip()
        if not query:
            return NormalizationResult(None, None, None)

        # 1) Alias map (exact, case-insensitive)
        alias_hit = self._alias_map.get(query.lower())
        if alias_hit:
            return NormalizationResult(alias_hit, "alias", 100)

        # 2) Fuzzy match against canonical names
        if not self._canonical_names or fuzz_process is None:
            return NormalizationResult(None, None, None)

        best: Optional[Tuple[str, int]] = fuzz_process.extractOne(query, self._canonical_names)
        if best and best[1] >= self._threshold:
            return NormalizationResult(best[0], "fuzzy", int(best[1]))

        return NormalizationResult(None, None, best[1] if best else None)

