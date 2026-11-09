"""The node server: this node's root contexts, as web pages, read-only.

``python3 -m pythia.interaction.node`` serves the root contexts that the
node's registry lists (``pythia/interaction/registry.py``), ``~/.pythia/root/``
by default, wherever their saves are, at ``http://127.0.0.1:8765/TOKEN/``:
the contexts, each one's turns, each turn's items, and each item whole, as
plain pages without script; and the same in glui's widgets, at
``/TOKEN/lem/`` (``view.py``). It began as wui, a server for a directory of
saves, and it uses only the standard library.
"""
