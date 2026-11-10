"""Pythia's tests.

The registry of root contexts is off for every CLI that the tests run, child
processes included, so that no test registers its saves in the user's
registry; the registry's own tests name temporary registries. So is the node
server's settings file, so that no test reaches the user's remote nodes, or
serves on the port that they set; the settings' own tests name theirs.
"""

import os

os.environ["PYTHIA_REGISTRY"] = "none"
os.environ["PYTHIA_NODE_CONFIG"] = "none"
