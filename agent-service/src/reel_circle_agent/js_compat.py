"""Number and text helpers that reproduce JavaScript semantics exactly.

The service must return byte-for-byte what the TypeScript recommendation agent returns (see
fixtures/agent-parity.json), and a few JavaScript built-ins differ from their Python namesakes.
"""

import math
import unicodedata
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal


def js_round(value: float) -> int:
    """`Math.round`: halves round towards +infinity (Python's `round` rounds them to even)."""
    floor = math.floor(value)
    return floor + 1 if value - floor >= 0.5 else floor


def js_sum(values: Iterable[float]) -> float:
    """Left-to-right float addition, like `reduce`. Python 3.12's `sum` uses compensated summation."""
    total = 0.0
    for value in values:
        total += value
    return total


def js_to_fixed(value: float, digits: int) -> str:
    """`Number.prototype.toFixed`: rounds the exact binary value half up (Python formats half to even)."""
    return str(Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def js_str(value: object) -> str:
    """String conversion of a number inside a template literal: `2020`, not `2020.0`."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def js_int32(value: int) -> int:
    """`ToInt32`: wraps an integer to a signed 32-bit value."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def strip_accents_lower(value: str) -> str:
    """`value.normalize('NFD').replace(/[\\u0300-\\u036f]/g, '').toLowerCase()`."""
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(character for character in decomposed if not "̀" <= character <= "ͯ").lower()


def unique(values: Iterable[str]) -> list[str]:
    """`[...new Set(values)]`: de-duplicates while keeping first-seen order."""
    return list(dict.fromkeys(values))
