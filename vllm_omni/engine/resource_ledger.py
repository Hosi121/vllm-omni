# SPDX-License-Identifier: Apache-2.0
"""Atomic reservations shared by native and external stage startup.

Capacities are admission ceilings for this controller, captured before loads.
They are not fresh `available` samples: resident allocations remain charged,
so sampling available memory cannot subtract our own allocation a second time.
Refresh ceilings only with an explicit, reconciled controller budget.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class ResourceUnavailable(RuntimeError):  # noqa: N818 - public admission result, like PlanNotAdmitted
    pass


@dataclass(frozen=True)
class Reservation:
    owner: str
    demands: Mapping[str, int]


class ResourceLedger:
    def __init__(self, capacities: Mapping[str, int]) -> None:
        if not capacities or any(type(n) is not int or n < 0 for n in capacities.values()):
            raise ValueError("explicit nonnegative memory ceilings are required")
        self.capacities = MappingProxyType(dict(capacities))
        self._active: dict[str, Reservation] = {}
        self._quarantined: set[str] = set()
        self._lock = threading.RLock()

    def reserve(self, owner: str, demands: Mapping[str, int]) -> Reservation:
        with self._lock:
            if not owner or owner in self._active:
                raise ValueError(f"reservation owner missing or already active: {owner}")
            if not demands or any(type(n) is not int or n < 0 for n in demands.values()):
                raise ValueError("nonnegative memory demands are required")
            for pool, count in demands.items():
                if pool not in self.capacities:
                    raise ResourceUnavailable(f"unknown memory pool: {pool}")
                used = sum(r.demands.get(pool, 0) for r in self._active.values())
                if used + count > self.capacities[pool]:
                    raise ResourceUnavailable(f"{pool}: {used} reserved + {count} requested > {self.capacities[pool]}")
            reservation = Reservation(owner, MappingProxyType(dict(demands)))
            self._active[owner] = reservation
            return reservation

    def release(self, reservation: Reservation, *, drained: bool) -> bool:
        with self._lock:
            if self._active.get(reservation.owner) is not reservation:
                return False  # stale token must never release a newer allocation
            if not drained:
                self._quarantined.add(reservation.owner)
                return False
            del self._active[reservation.owner]
            self._quarantined.discard(reservation.owner)
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "capacities": dict(self.capacities),
                "reserved": {p: sum(r.demands.get(p, 0) for r in self._active.values()) for p in self.capacities},
                "owners": list(self._active),
                "quarantined": sorted(self._quarantined),
            }


class GraphRequestGate:
    """One admitted pipeline request, held through the final consumer ACK.

    Stage reservations already include its payload allowance. Rejecting excess
    work before the ingress queue keeps slow consumers from accumulating inputs
    or error outputs, and leaves the existing control queue available to abort.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = None

    def acquire(self, request_id):
        with self._lock:
            if self._active is not None:
                raise ResourceUnavailable("graph pipeline capacity is one; consume or cancel the active request first")
            ticket = (request_id, object())
            self._active = ticket
            return ticket

    def current(self, request_id):
        with self._lock:
            return self._active if self._active is not None and self._active[0] == request_id else None

    def release(self, ticket):
        with self._lock:
            if ticket is not None and self._active is ticket:
                self._active = None
