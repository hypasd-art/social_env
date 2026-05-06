"""Seed minimal EnvironmentProfile + AgentProfile records for a local-backend demo.

Run with the project's conda env:
    SOTOPIA_STORAGE_BACKEND=local python scripts/seed_local_demo.py

Idempotent-ish: it will simply append new records each time. Pass --reset to wipe
the local data dirs first.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete ~/.sotopia/data/{EnvironmentProfile,AgentProfile}/ before seeding.",
    )
    args = parser.parse_args()

    os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")
    assert os.environ["SOTOPIA_STORAGE_BACKEND"] == "local", (
        "This script only seeds the local JSON backend. "
        "Set SOTOPIA_STORAGE_BACKEND=local."
    )

    from sotopia.database import AgentProfile, EnvironmentProfile
    from sotopia.database.persistent_profile import RelationshipType

    base = Path.home() / ".sotopia" / "data"
    if args.reset:
        for sub in ("EnvironmentProfile", "AgentProfile"):
            target = base / sub
            if target.exists():
                shutil.rmtree(target)
                print(f"[reset] removed {target}")

    env = EnvironmentProfile(
        codename="coffee_shop_favor",
        source="manual_seed",
        scenario=(
            "Two coworkers run into each other at a coffee shop. Agent1 has been "
            "thinking about asking Agent2 for help on a side project, while Agent2 "
            "has limited free time on weekends. They have ten minutes before Agent2 "
            "must leave for a meeting."
        ),
        agent_goals=[
            (
                "You want to convince the other person to spend a few hours this "
                "weekend helping you debug a small Python project. Be friendly but "
                "not pushy."
            ),
            (
                "You want to be polite but you really need the weekend for personal "
                "errands. Try to defer or offer a smaller alternative form of help."
            ),
        ],
        relationship=RelationshipType.acquaintance,
        age_constraint=None,
        occupation_constraint=None,
        tag="local_demo_seed",
    )
    env.save()
    print(f"[ok] saved EnvironmentProfile pk={env.pk}")

    a1 = AgentProfile(
        first_name="Alex",
        last_name="Johnson",
        age=29,
        occupation="software engineer",
        gender="non-binary",
        gender_pronoun="they/them",
        public_info=(
            "Alex is an enthusiastic engineer who often hacks on side projects "
            "and likes asking colleagues for help."
        ),
        personality_and_values=(
            "Curious, persistent, slightly impatient. Values creativity and "
            "knowledge sharing."
        ),
        decision_making_style="Goal-driven, willing to negotiate.",
        big_five="Open, conscientious, mid-extraversion.",
        secret="Has been secretly applying for a graduate program.",
        tag="local_demo_seed",
    )
    a1.save()
    print(f"[ok] saved AgentProfile pk={a1.pk}  ({a1.first_name} {a1.last_name})")

    a2 = AgentProfile(
        first_name="Sam",
        last_name="Kim",
        age=31,
        occupation="UX designer",
        gender="female",
        gender_pronoun="she/her",
        public_info=(
            "Sam is a thoughtful designer who carefully guards her personal time "
            "and prefers planning her weekends in advance."
        ),
        personality_and_values=(
            "Empathetic, organized, conflict-averse. Values balance and reliability."
        ),
        decision_making_style="Thoughtful, prefers compromises over confrontations.",
        big_five="Agreeable, conscientious, introverted.",
        secret="Is quietly considering a career change to product management.",
        tag="local_demo_seed",
    )
    a2.save()
    print(f"[ok] saved AgentProfile pk={a2.pk}  ({a2.first_name} {a2.last_name})")

    env_dir = base / "EnvironmentProfile"
    agent_dir = base / "AgentProfile"
    print(
        f"\n[summary] EnvironmentProfile: {len(list(env_dir.glob('*.json')))} files"
        f"\n          AgentProfile:       {len(list(agent_dir.glob('*.json')))} files"
        f"\n          base path:          {base}"
    )


if __name__ == "__main__":
    main()
