"""Trigger kinds of each Translator event schema version the judges read: an explicit table.

- Schema 3 (current, ``translate.events``): ``start`` (the first instruction, or a preset point),
  ``change`` (the clef-flash detector; a new or changed tactic) and ``request``.
- Schema 2 (runs written before 2026-11-08): ``initial`` (the first instruction, or a preset point),
  ``start`` and ``change`` (the detector's two labels) and ``request``.

A run of any other version, or a reason whose kind its version does not list, fails to load: a
kind is never inferred. The assess judge config maps each version's kinds to its point kinds
(``triggers`` in ``configs/evaluation/assess*.yaml``).
"""

from __future__ import annotations

LEGACY_EVENT_SCHEMA_VERSION = 2
CURRENT_EVENT_SCHEMA_VERSION = 3
TRIGGER_KINDS: dict[int, tuple[str, ...]] = {
    LEGACY_EVENT_SCHEMA_VERSION: ("initial", "start", "change", "request"),
    CURRENT_EVENT_SCHEMA_VERSION: ("start", "change", "request"),
}
EVENT_SCHEMA_VERSIONS = tuple(sorted(TRIGGER_KINDS))
ALL_TRIGGER_KINDS = tuple(dict.fromkeys(kind for kinds in TRIGGER_KINDS.values() for kind in kinds))
