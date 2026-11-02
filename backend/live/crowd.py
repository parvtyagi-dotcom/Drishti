"""Group gathering: several people staying together in one camera's view.

At a fence line a group is a signal on its own: people massing before a
crossing, or a smuggling party waiting for a vehicle. The rule is plain and
explainable: at least `min_people` in view for most of `hold_sec`. It is
measured on counts the tracker already produces, so it costs nothing extra.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

PERSON_CLASS = 0


@dataclass
class GroupConfig:
    min_people: int = 4
    hold_sec: float = 10.0
    # Share of samples in the hold window that must reach min_people, so one
    # missed detection does not reset the count.
    min_share: float = 0.8
    # After an alert, the group has to break up for this long before the
    # same camera can alert again. Stops one waiting group alerting forever.
    rearm_sec: float = 30.0


class GroupWatch:
    def __init__(self, config: GroupConfig | None = None, clock=time.monotonic):
        self.config = config or GroupConfig()
        self.clock = clock
        self._samples: deque[tuple[float, int]] = deque()
        self._alerted = False
        self._clear_since: float | None = None

    def observe(self, people: int) -> dict | None:
        """Feed one frame's person count; returns alert data when a group forms."""
        cfg = self.config
        now = self.clock()
        self._samples.append((now, people))
        while self._samples and now - self._samples[0][0] > cfg.hold_sec:
            self._samples.popleft()

        if self._alerted:
            if people < cfg.min_people:
                self._clear_since = self._clear_since or now
                if now - self._clear_since >= cfg.rearm_sec:
                    self._alerted = False
                    self._clear_since = None
            else:
                self._clear_since = None
            return None

        if not self._samples or now - self._samples[0][0] < cfg.hold_sec * 0.9:
            return None
        counts = [n for _, n in self._samples]
        share = sum(1 for n in counts if n >= cfg.min_people) / len(counts)
        if share < cfg.min_share:
            return None
        self._alerted = True
        ordered = sorted(counts)
        return {
            "people": ordered[len(ordered) // 2],
            "max_people": ordered[-1],
            "duration_sec": round(now - self._samples[0][0], 1),
            "min_people": cfg.min_people,
        }
