"""Number and text helpers that reproduce JavaScript semantics exactly.

The service must return byte-for-byte what the TypeScript recommendation agent returns (see
fixtures/agent-parity.json), and a few JavaScript built-ins differ from their Python namesakes.
"""

import math
import re
import unicodedata
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import quote


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


_JS_DECIMAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_JS_RADIX = {"0x": 16, "0o": 8, "0b": 2}


def js_number(value: object) -> float:
    """`Number(value)` for strings, numbers and missing values: hex, `Infinity`, blank as 0, otherwise NaN."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return math.nan
    text = value.strip()
    if not text:
        return 0.0
    if text in ("Infinity", "+Infinity", "-Infinity"):
        return -math.inf if text.startswith("-") else math.inf
    radix = _JS_RADIX.get(text[:2].lower())
    if radix:
        try:
            return float(int(text[2:], radix))
        except ValueError:
            return math.nan
    return float(text) if _JS_DECIMAL.fullmatch(text) else math.nan


def js_number_value(value: float) -> int | float:
    """A number as JSON.stringify writes it: integral values without a trailing `.0`."""
    return int(value) if math.isfinite(value) and value.is_integer() else value


def encode_uri_component(value: str) -> str:
    """`encodeURIComponent`: leaves `A-Z a-z 0-9 - _ . ! ~ * ' ( )` unescaped."""
    return quote(value, safe="-_.!~*'()")


def strip_one_trailing_slash(value: str) -> str:
    r"""`value.replace(/\/$/, '')`: removes at most one trailing slash."""
    return value[:-1] if value.endswith("/") else value


def js_slice(value: str, end: int) -> str:
    """`value.slice(0, end)`: cuts after `end` UTF-16 code units, so an emoji counts as two."""
    encoded = value.encode("utf-16-le")[: end * 2]
    return encoded.decode("utf-16-le", errors="ignore")
