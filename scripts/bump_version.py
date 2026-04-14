#!/usr/bin/env python3
"""Script simples para atualizar a versão semântica do projeto.

Uso:
  python scripts/bump_version.py patch
  python scripts/bump_version.py minor
  python scripts/bump_version.py major
  python scripts/bump_version.py 1.2.3

O script atualiza `VERSION` e `src/version.py` e tenta commitar/taggear.
"""
import sys
from pathlib import Path
import subprocess

VFILE = Path("VERSION")
SVFILE = Path("src") / "version.py"

def read_version():
    if not VFILE.exists():
        return "0.1.0"
    return VFILE.read_text().strip()

def write_version(v):
    VFILE.write_text(v + "\n")
    SVFILE.write_text(f'__version__ = "{v}"\n')

def bump(current, part):
    major, minor, patch = map(int, current.split("."))
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError("unknown part")
    return f"{major}.{minor}.{patch}"

def git_commit_and_tag(new_version):
    try:
        subprocess.run(["git", "add", "VERSION", "src/version.py"], check=True)
        subprocess.run(["git", "commit", "-m", f"chore(release): bump version to {new_version}"], check=True)
        subprocess.run(["git", "tag", f"v{new_version}"], check=True)
        print("Git commit and tag created.")
    except Exception as e:
        print("Git commit/tag failed (is repository initialized?):", e)

def main():
    if len(sys.argv) < 2:
        print("Usage: bump_version.py [patch|minor|major|<version>]")
        sys.exit(1)

    arg = sys.argv[1]
    cur = read_version()
    if arg in ("patch", "minor", "major"):
        new = bump(cur, arg)
    else:
        new = arg

    write_version(new)
    print(f"Version updated: {cur} -> {new}")
    git_commit_and_tag(new)

if __name__ == "__main__":
    main()
