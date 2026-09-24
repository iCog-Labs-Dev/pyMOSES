"""JSON-lines telemetry for the GOMEA inner optimizer.

One event per line: ``{"type": <event_type>, **fields}``. The file is
opened, appended to and closed for every event, so a long run that crashes
or is interrupted still leaves every event emitted so far on disk.
"""

from __future__ import annotations

import json


class TelemetrySink:
    """Append-only JSONL event writer; a no-op when ``path`` is ``None``."""

    def __init__(self, path: str | None) -> None:
        self.path = path

    def emit(self, event_type: str, **fields) -> None:
        """Write one ``{"type": event_type, **fields}`` line (keys sorted,
        non-JSON values rendered with ``str``)."""
        if self.path is None:
            return
        line = json.dumps({"type": event_type, **fields}, default=str, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
