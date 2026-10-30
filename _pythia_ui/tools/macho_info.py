#!/usr/bin/env python3
"""Check that Mach-O executables and dylibs built here can load on macOS.

For each binary: the CPU and minimum macOS version, the ad-hoc code
signature (every page re-hashed; Apple Silicon kills binaries whose
signature doesn't match), and how many symbols it imports from each
library. For a dylib, also its install name and the symbols it exports.
Exits 1 if something would stop the binary from loading.

usage: macho_info.py [-v] BINARY...   (-v also lists the imported symbols)
"""

import hashlib
import struct
import sys
import textwrap

# From research-gui (_glomery/tools/macho_info.py, commit a5099c9), with dylibs
# added: pythia's plugin is one.

CPU_TYPES = {0x0100000C: "arm64", 0x01000007: "x86_64"}
PLATFORM_MACOS = 1

LC_REQ_DYLD = 0x80000000
LC_SYMTAB = 0x2
LC_CODE_SIGNATURE = 0x1D
LC_BUILD_VERSION = 0x32
LC_MAIN = 0x28 | LC_REQ_DYLD
LC_ID_DYLIB = 0xD
MH_EXECUTE, MH_DYLIB = 2, 6
# Load commands that each take the next library ordinal (1, 2, ...).
DYLIB_COMMANDS = {
    0xC,  # LC_LOAD_DYLIB
    0x18 | LC_REQ_DYLD,  # LC_LOAD_WEAK_DYLIB
    0x1F | LC_REQ_DYLD,  # LC_REEXPORT_DYLIB
    0x20,  # LC_LAZY_LOAD_DYLIB
    0x23 | LC_REQ_DYLD,  # LC_LOAD_UPWARD_DYLIB
}

N_STAB, N_TYPE, N_EXT, N_UNDF, N_SECT = 0xE0, 0x0E, 0x01, 0x00, 0x0E
# Imports bound to this ordinal are searched for in every loaded image.
FLAT_LOOKUP_ORDINAL = 254
SPECIAL_ORDINALS = {
    0: "<self>",
    FLAT_LOOKUP_ORDINAL: "<flat namespace lookup>",
    255: "<main executable>",
}

CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
CSMAGIC_CODEDIRECTORY = 0xFADE0C02
CS_FLAG_NAMES = {0x2: "adhoc", 0x20000: "linker-signed"}
CS_HASH_TYPES = {1: "sha1", 2: "sha256"}


def version(packed):
    """Formats a Mach-O packed version (xxxx.yy.zz)."""
    return f"{packed >> 16}.{(packed >> 8) & 0xFF}.{packed & 0xFF}"


class MachO:
    """The parts of a thin 64-bit Mach-O executable or dylib that matter for loading."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = data = f.read()
        magic, cputype, _, self.filetype, ncmds = struct.unpack_from("<5I", data)
        if magic != 0xFEEDFACF:
            raise SystemExit(f"{path}: not a thin 64-bit Mach-O file")
        self.arch = CPU_TYPES.get(cputype, f"cputype {cputype:#x}")
        self.install_name = None  # a dylib's own name (LC_ID_DYLIB)
        self.dylibs = []  # (install name, current, compatibility); ordinal = index + 1
        self.build = None  # (platform, minos, sdk)
        self.has_main = False
        self.symtab = None  # (symoff, nsyms, stroff, strsize)
        self.signature = None  # (dataoff, datasize)
        off = 32
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<2I", data, off)
            if cmd in DYLIB_COMMANDS:
                name_off, _, current, compat = struct.unpack_from("<4I", data, off + 8)
                self.dylibs.append((self.cstr(off + name_off), current, compat))
            elif cmd == LC_BUILD_VERSION:
                self.build = struct.unpack_from("<3I", data, off + 8)
            elif cmd == LC_MAIN:
                self.has_main = True
            elif cmd == LC_ID_DYLIB:
                self.install_name = self.cstr(off + struct.unpack_from("<I", data, off + 8)[0])
            elif cmd == LC_SYMTAB:
                self.symtab = struct.unpack_from("<4I", data, off + 8)
            elif cmd == LC_CODE_SIGNATURE:
                self.signature = struct.unpack_from("<2I", data, off + 8)
            off += size

    def cstr(self, off):
        return self.data[off : self.data.index(b"\0", off)].decode()

    def library(self, ordinal):
        if 1 <= ordinal <= len(self.dylibs):
            return self.dylibs[ordinal - 1][0]
        return SPECIAL_ORDINALS.get(ordinal, f"<bad ordinal {ordinal}>")

    def imports(self):
        """Yields (symbol, library ordinal) for each undefined external symbol."""
        if self.symtab is None:
            return
        symoff, nsyms, stroff, _ = self.symtab
        for i in range(nsyms):
            strx, ntype, _, desc, _ = struct.unpack_from("<IBBHQ", self.data, symoff + 16 * i)
            if not ntype & N_STAB and ntype & N_TYPE == N_UNDF and ntype & N_EXT:
                yield self.cstr(stroff + strx), desc >> 8

    def exports(self):
        """Yields each external symbol defined in a section: what a dylib exports."""
        if self.symtab is None:
            return
        symoff, nsyms, stroff, _ = self.symtab
        for i in range(nsyms):
            strx, ntype, _, _, _ = struct.unpack_from("<IBBHQ", self.data, symoff + 16 * i)
            if not ntype & N_STAB and ntype & N_TYPE == N_SECT and ntype & N_EXT:
                yield self.cstr(stroff + strx)

    def check_signature(self):
        """Re-hashes the pages the signature covers; returns (ok, description)."""
        if self.signature is None:
            return False, "missing (arm64 macOS refuses to run unsigned code)"
        d, base = self.data, self.signature[0]
        magic, _, count = struct.unpack_from(">3I", d, base)
        if magic != CSMAGIC_EMBEDDED_SIGNATURE:
            return False, f"bad magic {magic:#x}"
        for i in range(count):
            _, blob = struct.unpack_from(">2I", d, base + 12 + 8 * i)
            cd = base + blob
            if struct.unpack_from(">I", d, cd)[0] != CSMAGIC_CODEDIRECTORY:
                continue
            fields = struct.unpack_from(">7I4B", d, cd + 8)
            _, flags, hash_off, ident_off, _, n_pages, code_limit = fields[:7]
            hash_size, hash_type, _, page_shift = fields[7:]
            if hash_type not in CS_HASH_TYPES:
                return False, f"unknown hash type {hash_type}"
            page = 1 << page_shift
            hashes = d[cd + hash_off : cd + hash_off + n_pages * hash_size]
            bad = sum(
                hashlib.new(CS_HASH_TYPES[hash_type], d[n * page : min((n + 1) * page, code_limit)]).digest()
                != hashes[n * hash_size : (n + 1) * hash_size]
                for n in range(n_pages)
            )
            kinds = ", ".join(name for bit, name in CS_FLAG_NAMES.items() if flags & bit)
            kinds = kinds or f"flags {flags:#x}"
            if bad or n_pages * page < code_limit:
                return False, f"{kinds}; {bad} of {n_pages} page hashes don't match"
            ident = self.cstr(cd + ident_off)
            return True, f"{kinds}, identifier {ident!r}, {n_pages} pages verified"
        return False, "no CodeDirectory"


def report(path, verbose):
    """Prints a summary of one binary and returns the problems found."""
    m = MachO(path)
    problems = []
    print(path)
    if m.build is None:
        print(f"  {m.arch}")
        problems.append("no LC_BUILD_VERSION")
    else:
        platform, minos, sdk = m.build
        print(f"  {m.arch}, platform {platform}, min OS {version(minos)}, SDK {version(sdk)}")
        if platform != PLATFORM_MACOS:
            problems.append(f"platform {platform} is not macOS ({PLATFORM_MACOS})")
    if m.filetype == MH_EXECUTE:
        if not m.has_main:
            problems.append("no LC_MAIN; not an executable?")
    elif m.filetype == MH_DYLIB:
        print(f"  install name: {m.install_name}")
        if m.install_name is None:
            problems.append("no LC_ID_DYLIB; not a dylib?")
        elif not m.install_name.startswith("@"):
            problems.append(f"install name {m.install_name!r} is a path on the build machine, not an @rpath/ name")
        exports = sorted(m.exports())
        shown = ", ".join(exports) if verbose or len(exports) <= 8 else f"{len(exports)} symbols (-v lists them)"
        print(f"  exports: {shown or 'none'}")
    else:
        problems.append(f"file type {m.filetype} is neither an executable nor a dylib")
    ok, desc = m.check_signature()
    print(f"  signature: {desc}")
    if not ok:
        problems.append(f"code signature: {desc}")
    imports = {}
    for name, ordinal in m.imports():
        imports.setdefault(ordinal, []).append(name)
    for ordinal in sorted(set(imports) | set(range(1, len(m.dylibs) + 1))):
        symbols = sorted(imports.get(ordinal, []))
        library = m.library(ordinal)
        if verbose and 1 <= ordinal <= len(m.dylibs):
            _, current, compat = m.dylibs[ordinal - 1]
            library += f" (current {version(current)}, compatibility {version(compat)})"
        print(f"  {library}: {len(symbols)} imports")
        if verbose and symbols:
            print(textwrap.fill(", ".join(symbols), 100, initial_indent=" " * 6, subsequent_indent=" " * 6))
        if ordinal == FLAT_LOOKUP_ORDINAL:
            problems.append(f"{len(symbols)} imports left for flat lookup (a `make tbd` build?)")
        elif not 1 <= ordinal <= len(m.dylibs):
            problems.append(f"{len(symbols)} imports bound to {library}")
    return problems


def main(argv):
    verbose = "-v" in argv
    paths = [arg for arg in argv if arg != "-v"]
    if not paths:
        sys.exit(__doc__.strip())
    failed = False
    for path in paths:
        problems = report(path, verbose)
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        failed = failed or bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
