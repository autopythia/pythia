from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math

@dataclass
class LocalDate:
    _d: date = None

    def __post_init__(self):
        if self._d is None:
            self._d = datetime.utcnow().date()

    def from_str(s: str) -> "LocalDate":
        d = date.fromisoformat(s)
        return LocalDate(d)

    def __str__(self):
        return f"{self._d.isoformat()}"

    def is_weekend(self) -> bool:
        return self._d.weekday() >= 5

    def next(self) -> "LocalDate":
        return LocalDate(self._d + timedelta(days=1))

    def prev(self) -> "LocalDate":
        return LocalDate(self._d - timedelta(days=1))

@dataclass
class Timedelta:
    _dt: timedelta = None

    def seconds(self) -> float:
        return self._dt.total_seconds()

    def pretty_format(self) -> str:
        sec = self.seconds()
        rem = sec
        h = math.floor(rem / 3600)
        rem = rem - h * 3600
        m = math.floor(rem / 60)
        rem = rem - m * 60
        ms = math.ceil(rem * 1000)
        if h > 0:
            s = int(math.ceil(ms / 1000))
            return f"{h}h {m:02}m {s:02}s"
        elif m > 0:
            s = int(math.ceil(ms / 1000))
            return f"{m}m {s:02}s"
        else:
            # FIXME
            s = int(math.ceil(ms / 1000))
            return f"{s}s"
            # s = ms / 1000
            # return f"{s:.01}s"

@dataclass
class Timestamp:
    _dt: datetime = None

    def __post_init__(self):
        if self._dt is None:
            self._dt = datetime.utcnow()

    def __str__(self):
        return f"{self._dt.isoformat()}Z"

    def __sub__(self, rhs: "Timestamp") -> Timedelta:
        return Timedelta(self._dt - rhs._dt)
