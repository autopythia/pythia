"""Pythia's tests.

The registry of root contexts is off for every CLI that the tests run, child
processes included, so that no test registers its saves in the user's
registry; the registry's own tests name temporary registries.
"""

import os

os.environ["PYTHIA_REGISTRY"] = "none"
