"""design_1 §3 — Scheduling window 的 deterministic session resolution（纯函数）。"""

from __future__ import annotations

from dataclasses import dataclass


def absorb_strict_subsets(unique_sets: list[frozenset[str]]) -> list[frozenset[str]]:
    """§3.4 rule 2 — 严格包含则保留更大 participant 集，吸收较小 request。"""
    uniq = list(dict.fromkeys(unique_sets))
    kept: list[frozenset[str]] = []
    for S in sorted(uniq, key=lambda x: (-len(x), sorted(x))):
        if any(S < T for T in kept):
            continue
        kept = [T for T in kept if not (T < S)]
        kept.append(S)
    return sorted(kept, key=lambda x: sorted(x))


def drop_overlapping_without_containment(
    sets: list[frozenset[str]],
) -> tuple[list[frozenset[str]], list[tuple[frozenset[str], frozenset[str]]]]:
    """§3.4 rule 3 — 有交且互不包含 ⇒ 双方都失败。"""
    doomed: set[int] = set()
    conflicts: list[tuple[frozenset[str], frozenset[str]]] = []
    for i, Si in enumerate(sets):
        for j, Sj in enumerate(sets):
            if i >= j:
                continue
            inter = Si & Sj
            if not inter:
                continue
            if Si < Sj or Sj < Si:
                continue
            doomed.add(i)
            doomed.add(j)
            conflicts.append((Si, Sj))
    survivors = [sets[i] for i in range(len(sets)) if i not in doomed]
    return survivors, conflicts


@dataclass(frozen=True)
class SchedulingResolveTrace:
    valid_sets: tuple[frozenset[str], ...]
    merged_unique: tuple[frozenset[str], ...]
    after_absorb: tuple[frozenset[str], ...]
    conflict_pairs: tuple[tuple[frozenset[str], frozenset[str]], ...]
    final_sessions: tuple[frozenset[str], ...]


def resolve_valid_session_sets(
    valid_participant_sets: list[frozenset[str]],
) -> tuple[tuple[frozenset[str], ...], SchedulingResolveTrace]:
    """对 ``R_valid`` 的 participant 集合应用 §3.4 v0 规则，返回最终并列 session 集。"""
    merged_list = list(dict.fromkeys(valid_participant_sets))
    merged_list.sort(key=lambda x: sorted(x))
    absorbed = absorb_strict_subsets(merged_list)
    survivors, conflicts = drop_overlapping_without_containment(absorbed)
    final = sorted(survivors, key=lambda x: sorted(x))
    trace = SchedulingResolveTrace(
        valid_sets=tuple(valid_participant_sets),
        merged_unique=tuple(merged_list),
        after_absorb=tuple(absorbed),
        conflict_pairs=tuple(conflicts),
        final_sessions=tuple(final),
    )
    return tuple(final), trace
