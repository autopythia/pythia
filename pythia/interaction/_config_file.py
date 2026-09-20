"""Bounded configuration reads; special files must not block frontend startup."""

import os
import stat


def read_config_bytes(path, limit):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Configuration must be a regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Configuration exceeds the size limit.")
        return data
    finally:
        os.close(descriptor)
