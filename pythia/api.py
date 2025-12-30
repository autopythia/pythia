from typing import Optional
from dataclasses import dataclass

from pythia.api_client import APIClient, APIJournal
from pythia.api_registry import APIRegistry

@dataclass
class APIServices:
    registry: APIRegistry = None
    client: APIClient = None
    enable_journal: Optional[bool] = None

    def __post_init__(self):
        if self.registry is None:
            self.registry = APIRegistry()
        if self.client is None:
            if self.enable_journal is None or self.enable_journal:
                self.client = APIClient(self.registry)
            else:
                self.client = APIClient(self.registry, _journal=APIJournal(enable=False))
