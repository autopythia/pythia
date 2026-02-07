from dataclasses import dataclass
import functools
import importlib
import pkgutil

_INDEX = None

@dataclass
class _AutoTapIndex:
    pass

    @classmethod
    def get(cls):
        global _INDEX
        if _INDEX is None:
            _INDEX = cls()
        return _INDEX

def test(fun=None, name=None):
    pass

def snapshot_test(fun=None, name=None):
    pass

@dataclass
class AutoTap:
    pass

    def main(self, package):
        self._load(package)
        index = _AutoTapIndex.get()
        # TODO

    def _load(self, package):
        if isinstance(package, str):
            package = importlib.import_module(package)
        for _loader, mod_name, is_pkg in pkgutil.walk_packages(package.__path__):
            mod_path = f"{package.__path__}.{mod_name}"
            try:
                importlib.import_module(mod_path)
            except ModuleNotFoundError:
                continue
            if is_pkg:
                self._load(mod_path)

def main(package):
    return AutoTap().main(package)
