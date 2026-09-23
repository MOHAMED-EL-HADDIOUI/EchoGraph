from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Canonical dedupe key: trimmed, whitespace-collapsed, casefolded.

    The original title is always preserved on the node; normalization is
    only used for matching, so "Atlas Migration" == "ATLAS  migration".
    """
    return _WHITESPACE.sub(" ", title.strip()).casefold()


def quotes_match(quote: str, chunk: str) -> bool:
    """True when a model-supplied quote is verifiably a chunk substring."""
    q = quote.strip()
    return bool(q) and q.lower() in chunk.lower()
