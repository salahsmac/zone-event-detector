
# This file is the "brain" for intrusion/loitering.
# It keeps a little memory per tracked person ID.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class TrackState:
    """
    One state bucket per person track_id.
    We reset timers when they leave the zone.
    """
    inside_since: Optional[float] = None
    last_seen: float = 0.0
    intrusion_fired: bool = False
    loiter_fired: bool = False


class EventEngine:
    """
    Event rules:
      - Intrusion: inside zone > intrusion_sec (fire once per visit)
      - Loitering: inside zone > loiter_sec (fire once per visit)

    Tracker dropout handling:
      - If an ID disappears briefly (< missing_tolerance_sec), we keep its state.
      - If it disappears longer than that, we delete it to avoid stale states.
    """

    def __init__(
        self,
        intrusion_sec: float = 1.0,
        loiter_sec: float = 15.0,
        missing_tolerance_sec: float = 1.0,
    ) -> None:
        self.intrusion_sec = float(intrusion_sec)
        self.loiter_sec = float(loiter_sec)
        self.missing_tolerance_sec = float(missing_tolerance_sec)
        self.state: Dict[int, TrackState] = {}

    def _cleanup_missing(self, now: float, active_ids: Set[int]) -> None:
        """Drop IDs that have been gone for too long."""
        dead = []
        for tid, st in self.state.items():
            if tid in active_ids:
                continue
            if (now - st.last_seen) > self.missing_tolerance_sec:
                dead.append(tid)

        for tid in dead:
            del self.state[tid]

    def update(
        self,
        now: float,
        observations: List[Tuple[int, bool]],
    ) -> Tuple[Dict[int, float], Dict[int, str], List[dict]]:
        """
        observations: list of (track_id, inside_zone)

        Returns:
          time_in_zone: track_id -> seconds inside
          status: track_id -> "normal" | "intruder" | "loitering"
          new_events: list of dict logs (time_sec, track_id, event_type)
        """
        time_in_zone: Dict[int, float] = {}
        status: Dict[int, str] = {}
        new_events: List[dict] = []

        active_ids: Set[int] = set()

        for tid, inside in observations:
            active_ids.add(tid)

            st = self.state.get(tid)
            if st is None:
                st = TrackState()
                self.state[tid] = st

            st.last_seen = now

            if not inside:
                # Outside zone = reset this "visit"
                st.inside_since = None
                st.intrusion_fired = False
                st.loiter_fired = False
                time_in_zone[tid] = 0.0
                status[tid] = "normal"
                continue

            # Inside zone
            if st.inside_since is None:
                # First frame we consider them inside
                st.inside_since = now
                st.intrusion_fired = False
                st.loiter_fired = False

            duration = now - st.inside_since
            time_in_zone[tid] = duration

            # Fire events once per inside-session
            if (duration >= self.intrusion_sec) and (not st.intrusion_fired):
                st.intrusion_fired = True
                new_events.append(
                    {"time_sec": round(now, 3), "track_id": int(tid), "event_type": "intrusion"}
                )

            if (duration >= self.loiter_sec) and (not st.loiter_fired):
                st.loiter_fired = True
                new_events.append(
                    {"time_sec": round(now, 3), "track_id": int(tid), "event_type": "loitering"}
                )

            # Status priority (loitering > intruder > normal)
            if st.loiter_fired:
                status[tid] = "loitering"
            elif st.intrusion_fired:
                status[tid] = "intruder"
            else:
                status[tid] = "normal"

        # Drop IDs that vanished for too long
        self._cleanup_missing(now, active_ids)

        return time_in_zone, status, new_events
