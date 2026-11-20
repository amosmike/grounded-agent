from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ProposedChange, World


class ToolError(Exception):
    """Bad tool input. The message goes back to the model as an error result."""


def load_world(path: Path) -> World:
    return World(**json.loads(path.read_text()))


class Sandbox:
    """Read-only access to a snapshot, plus a queue of proposed writes.

    Nothing the agent does changes the snapshot: writes only become proposals for a person to approve.
    """

    def __init__(self, world: World) -> None:
        self._world = world
        self._seen: set[tuple[str, str]] = set()
        self.proposals: list[ProposedChange] = []

    @property
    def collections(self) -> list[str]:
        return list(self._world.collections)

    def find_records(self, collection: str, field: str, value: str) -> list[dict[str, Any]]:
        rows = self._world.collections.get(collection)
        if rows is None:
            raise ToolError(f'Unknown collection "{collection}". Known: {", ".join(self.collections)}.')
        found = [r for r in rows if str(r.get(field)) == value]
        self._seen.update((collection, r["id"]) for r in found)
        return found

    def search_logs(self, query: str, limit: int = 50) -> list[dict[str, str]]:
        q = query.lower()
        hits = [l for l in self._world.logs if q in l["message"].lower() or q in l["service"].lower()]
        return hits[:limit]

    def propose(self, collection: str, record_id: str, field: str, new_value: Any, reason: str) -> ProposedChange:
        # Only records the agent actually read can be changed, so a fix can't target something it guessed at.
        if (collection, record_id) not in self._seen:
            raise ToolError(f"Read {collection}/{record_id} with find_records before proposing a change to it.")
        record = next((r for r in self._world.collections.get(collection, []) if r["id"] == record_id), None)
        if record is None:
            raise ToolError(f"No {collection} record with id {record_id}.")
        if field == "id" or field not in record:
            raise ToolError(f'{collection} records have no changeable field "{field}".')
        if any(p.collection == collection and p.record_id == record_id and p.field == field for p in self.proposals):
            raise ToolError(f"A change to {collection}/{record_id}.{field} is already proposed.")
        change = ProposedChange(
            id=f"chg_{len(self.proposals) + 1}",
            collection=collection,
            record_id=record_id,
            field=field,
            old_value=record[field],
            new_value=new_value,
            reason=reason,
        )
        self.proposals.append(change)
        return change
