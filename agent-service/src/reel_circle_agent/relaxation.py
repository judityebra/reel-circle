"""Relaxations: loosening one hard constraint at a time when nothing in the candidate pool fits.

The order goes from what users usually care least about to what they care most about. Excluded
genres are never relaxed: the user explicitly ruled them out.
"""

from typing import Any, Literal, TypedDict

from .preferences import Preferences
from .scoring import HARD_CONSTRAINTS, Movie

RelaxableConstraint = Literal["yearRange", "languages", "genres", "maxRuntime", "platforms"]
RELAXATION_ORDER: list[RelaxableConstraint] = ["yearRange", "languages", "genres", "maxRuntime", "platforms"]
RUNTIME_STEP_MINUTES = 15

# Functional syntax because "from" is a Python keyword; the keys are the browser's wire format.
Proposal = TypedDict("Proposal", {"constraint": RelaxableConstraint, "from": Any, "to": Any})
RelaxationDecision = Literal["accept", "reject"]


def next_relaxation(preferences: Preferences, rejected: list[str], candidate_pool: list[Movie]) -> Proposal | None:
    """The next constraint to offer loosening, or None when every active one has been declined.

    Runtime is offered 15 minutes at a time, but not once the limit already covers the longest
    movie in the candidate pool: loosening it further could not change anything.
    """
    longest = max((movie["runtime"] for movie in candidate_pool), default=0)
    for constraint in RELAXATION_ORDER:
        current = preferences.get(constraint)
        if constraint in rejected or not current or (constraint == "maxRuntime" and current >= longest):
            continue
        if constraint == "yearRange":
            loosened = None
        elif constraint == "maxRuntime":
            loosened = current + RUNTIME_STEP_MINUTES
        else:
            loosened = []
        return {"constraint": constraint, "from": current, "to": loosened}
    return None


def apply_relaxation(preferences: Preferences, proposal: Proposal) -> Preferences:
    relaxed = {**preferences, proposal["constraint"]: proposal["to"]}
    return {key: value for key, value in relaxed.items() if value is not None}


def tightest_constraint(candidate_pool: list[Movie], preferences: Preferences) -> str | None:
    """The active hard constraint that, on its own, lets the fewest movies through (ties: relaxation order)."""
    active = [name for name in [*RELAXATION_ORDER, "excludedGenres"] if preferences.get(name)]
    if not active or not candidate_pool:
        return None
    return min(active, key=lambda name: sum(HARD_CONSTRAINTS[name](movie, preferences) for movie in candidate_pool))
