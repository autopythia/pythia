from typing import Optional
from dataclasses import dataclass
import functools
# import gzip
import json

@dataclass
class SnapshotIndex:
    pass

@dataclass
class JsonSnapshotLog:
    _log_path: str = None
    _log: Optional[list] = None
    _eq_ctr: int = 0

    def __post_init__(self):
        try:
            with open(self._log_path, "r") as file:
                log_items = []
                for line in file:
                    item = json.loads(line)
                    log_items.append(item)
            self._log = log_items
        except OSError:
            self._log = None

    def __eq__(self, value) -> bool:
        if self._log is None or self._eq_ctr >= len(self._log):
            data = json.dumps(value)
            with open(self._log_path, "a") as file:
                print(data, file=file)
            result = True
        else:
            result = (self._log[self._eq_ctr] == value)
        self._eq_ctr += 1
        return result

def snapshot_test(fun):
    @functools.wraps(fun)
    def wrapped_fun(*args, **kwargs):
        fun_path = f"{fun.__module__}.{fun.__qualname__}"
        snapshot = JsonSnapshotLog(f"pythia_test.__testdata__/{fun_path}.__snapshot__")
        fun(*args, **kwargs, snapshot=snapshot)
    return wrapped_fun
