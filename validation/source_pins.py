"""Pinned-source check that honours documented amendments (validation/SOURCE_AMENDMENTS.json)."""
import json
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "SOURCE_AMENDMENTS.json"


def matches(name, pinned, actual):
    """True if `actual` is the pinned digest or reachable from it through recorded amendments of `name`."""
    if actual == pinned:
        return True
    chain = json.loads(_FILE.read_text(encoding="utf-8")).get("amendments", {}).get(name, [])
    current, seen = pinned, set()
    while current not in seen:
        seen.add(current)
        nxt = next((a["to_sha256"] for a in chain if a["from_sha256"] == current and a.get("evidence_impact")), None)
        if nxt is None:
            return False
        if nxt == actual:
            return True
        current = nxt
    return False
