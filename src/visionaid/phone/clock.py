"""Phone ↔ computer clock synchronisation by ping/pong (simplified NTP style)."""

from __future__ import annotations

import math
from collections import OrderedDict, deque

from visionaid.bus.messages import ClockSync


class ClockEstimator:
    """Estimates `offset` such that t_pc ≈ t_phone + offset.

    Each ping `id` gives d = t_pc_received − t_phone_sent = offset + outbound delay. The phone
    measures the RTT of this exchange when the pong returns and sends it back with the next ping
    (`prev_id`, `rtt`). d and RTT of the **same** exchange are matched; over a sliding window, the
    exchange with minimal RTT is kept: offset = d − RTT/2, uncertainty RTT/2. Without a matched RTT,
    offset = min(d) (biased by the minimal outbound delay), with a default uncertainty.
    """

    def __init__(self, window: int = 30, default_uncertainty: float = 0.05) -> None:
        self.samples: deque[tuple[float, float, float]] = deque(maxlen=window)
        self.raw: deque[tuple[float, float]] = deque(maxlen=window)
        self.pending: OrderedDict = OrderedDict()
        self.default_uncertainty = default_uncertainty

    def ping(self, ping_id, t_phone: float, t_pc: float) -> None:
        """Records the reception of a ping."""
        d = t_pc - t_phone
        self.raw.append((t_pc, d))
        if ping_id is not None:
            self.pending[ping_id] = (t_pc, d)
            while len(self.pending) > 64:
                self.pending.popitem(last=False)

    def rtt(self, ping_id, rtt: float | None) -> None:
        """Associates the RTT measured by the phone with the ping `ping_id`."""
        if ping_id is None or rtt is None or not math.isfinite(rtt) or rtt < 0:
            return
        hit = self.pending.pop(ping_id, None)
        if hit is not None:
            self.samples.append((hit[0], hit[1], float(rtt)))

    def estimate(self, t: float = 0.0) -> ClockSync:
        """Estimate from the exchanges in the window."""
        if self.samples:
            _, d, r = min(self.samples, key=lambda s: s[2])
            return ClockSync(t, d - r / 2.0, r / 2.0, r)
        if self.raw:
            d = min(d for _, d in self.raw)
            return ClockSync(t, d, self.default_uncertainty, 0.0)
        return ClockSync(t, 0.0, math.inf, 0.0)
