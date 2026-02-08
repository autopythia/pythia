def rclear() -> str:
    return "\x1b[K"

def bold(s: str, bold=True) -> str:
    if bold:
        return f"\x1b[1m{s}\x1b[0m"
    else:
        return f"{s}"

def plain(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[1m{s}\x1b[0m"
    else:
        return f"{s}"

def red(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[31;1m{s}\x1b[0m"
    else:
        return f"\x1b[31m{s}\x1b[0m"

def green(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[32;1m{s}\x1b[0m"
    else:
        return f"\x1b[32m{s}\x1b[0m"

def magenta(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[35;1m{s}\x1b[0m"
    else:
        return f"\x1b[35m{s}\x1b[0m"

def cyan(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[36;1m{s}\x1b[0m"
    else:
        return f"\x1b[36m{s}\x1b[0m"

def gray(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[37;1m{s}\x1b[0m"
    else:
        return f"\x1b[37m{s}\x1b[0m"

def bright_key(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[90;1m{s}\x1b[0m"
    else:
        return f"\x1b[90m{s}\x1b[0m"

def dim(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[2;1m{s}\x1b[0m"
    else:
        return f"\x1b[2m{s}\x1b[0m"

def underline(s: str, bold=False) -> str:
    if bold:
        return f"\x1b[4;1m{s}\x1b[0m"
    else:
        return f"\x1b[4m{s}\x1b[0m"
