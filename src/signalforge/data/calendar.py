"""US trading calendar utility for determining last trading day."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# NYSE holidays 2025-2027 (New Year, MLK Day, Presidents Day, Good Friday,
# Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas)
_NYSE_HOLIDAYS: frozenset[date] = frozenset(
    [
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),  # Observed (Jul 4 = Sat)
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),  # Observed (Jun 19 = Sat)
        date(2027, 7, 5),  # Observed (Jul 4 = Sun)
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),  # Observed (Dec 25 = Sat)
    ]
)


def last_trading_day(reference: datetime | date | None = None) -> date:
    """Return the most recent US trading day on or before *reference*.

    Rolls back past weekends and NYSE holidays.  If *reference* is ``None``
    the current UTC date is used.
    """
    if reference is None:
        now_utc = datetime.now(timezone.utc)
        # Daily bar data becomes available after market close (ET 16:00 = UTC 20:00).
        # Add 2h buffer for data publication → treat today as expected only after UTC 22:00.
        if now_utc.hour < 22:
            day = (now_utc - timedelta(days=1)).date()
        else:
            day = now_utc.date()
    elif isinstance(reference, datetime):
        day = reference.date()
    else:
        day = reference

    # Roll back until we land on a weekday that is not a holiday
    while day.weekday() >= 5 or day in _NYSE_HOLIDAYS:
        day = date.fromordinal(day.toordinal() - 1)

    return day
