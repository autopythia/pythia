#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import shlex
import stat

def main(args):
    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = os.path.join(os.environ["HOME"], ".local")
    cwd = os.getcwd()
    qqq_src_path = os.path.join(cwd, "qqq_vim.py")
    qqq_dst_path = os.path.join(prefix, "bin", "_qqq_vim")
    qqq_content = (
f"""PYTHONPATH={shlex.quote(cwd)} python3 {shlex.quote(qqq_src_path)} "$@"
"""
    )
    with open(qqq_dst_path, "w") as f:
        print(qqq_content, end="", file=f)
    os.chmod(qqq_dst_path, 0o755)
    print("Installed: _qqq_vim")
    qq_src_path = os.path.join(cwd, "qq_vim.py")
    qq_dst_path = os.path.join(prefix, "bin", "_qq_vim")
    qq_content = (
f"""PYTHONPATH={shlex.quote(cwd)} python3 {shlex.quote(qq_src_path)} "$@"
"""
    )
    with open(qq_dst_path, "w") as f:
        print(qq_content, end="", file=f)
    os.chmod(qq_dst_path, 0o755)
    print("Installed: _qq_vim")
    print("Done installation.")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--prefix", type=str, default=None)
    args = args.parse_args()
    main(args)
