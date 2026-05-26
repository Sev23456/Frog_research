#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a long-running child process and mirror its lifecycle to files.")
    parser.add_argument("--cwd", required=True, help="Working directory for the child process.")
    parser.add_argument("--stdout", required=True, help="File path for child stdout.")
    parser.add_argument("--stderr", required=True, help="File path for child stderr.")
    parser.add_argument("--pid-file", required=True, help="Where to write the child PID.")
    parser.add_argument("--returncode-file", required=True, help="Where to write the child return code after exit.")
    parser.add_argument("--meta-file", required=True, help="Where to write launcher metadata.")
    parser.add_argument("--detach-child", action="store_true", help="Spawn the child as a detached process and exit immediately.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute. Prefix with -- before the command.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing child command after --")
    return args


def main() -> int:
    args = parse_args()

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    pid_path = Path(args.pid_file)
    returncode_path = Path(args.returncode_file)
    meta_path = Path(args.meta_file)

    for path in (stdout_path, stderr_path, pid_path, returncode_path, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "launched_at": datetime.now().isoformat(),
        "launcher_pid": os.getpid(),
        "cwd": args.cwd,
        "command": args.command,
        "detach_child": bool(args.detach_child),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")

    creationflags = 0
    if args.detach_child and os.name == "nt":
        # Keep the child alive even if the short-lived launcher process exits.
        creationflags |= 0x00000008  # DETACHED_PROCESS
        creationflags |= 0x00000200  # CREATE_NEW_PROCESS_GROUP

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        child = subprocess.Popen(
            args.command,
            cwd=args.cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
            creationflags=creationflags,
        )
        pid_path.write_text(str(child.pid), encoding="utf-8")
        if args.detach_child:
            returncode_path.write_text("detached", encoding="utf-8")
            return 0
        returncode = child.wait()

    returncode_path.write_text(str(returncode), encoding="utf-8")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
