"""Named test personas for long-term negotiation tests.

The engine still keys participants as ``firm_a`` … ``firm_d``; tests should refer to
:class:`NamedTestAgent` instances (human display name + bound key) instead of raw ``firm_*`` literals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NamedTestAgent:
    """Human-readable identity for a test, bound to one canonical roster key."""

    display_name: str
    key: str

    def __repr__(self) -> str:
        return f"NamedTestAgent({self.display_name!r} -> {self.key!r})"


# —— Bilateral smoke (buyer lead + vendor) ——
ARI_LYNN = NamedTestAgent("Ari Lynn", "firm_a")
MEI_OKADA = NamedTestAgent("Mei Okada", "firm_b")

# —— Third / fourth commercial participants (firms-only roster) ——
CHEN_WEI = NamedTestAgent("Chen Wei", "firm_c")
DANA_RIOS = NamedTestAgent("Dana Rios", "firm_d")

ALL_COMMERCIAL_NAMED_AGENTS: tuple[NamedTestAgent, ...] = (
    ARI_LYNN,
    MEI_OKADA,
    CHEN_WEI,
    DANA_RIOS,
)


def roster_keys(*agents: NamedTestAgent) -> tuple[str, ...]:
    """Tuple of engine keys for ``build_rule_dummy_agents`` / roster assertions."""
    return tuple(a.key for a in agents)
