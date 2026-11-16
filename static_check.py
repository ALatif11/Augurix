"""Reject generated strategy code before it runs.

This is a cheap filter, not a jail: the Docker sandbox is the security boundary.
It exists to give the coder agent a clear reason, to keep strategies from reading
data they were not handed, and to catch dates hard-coded from the model's memory,
which the look-ahead test cannot see.
"""
import ast
import re

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint"}
FILE_READERS = {"load", "loadtxt", "genfromtxt", "fromfile", "memmap"}
YEAR = re.compile(r"(?<![\w.])20(?:09|[12]\d)(?![\w.])")


def check(code):
    """Return None if the code may run, otherwise a one-line reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error on line {e.lineno}: {e.msg}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "strategy" for n in tree.body):
        return "no top-level function named strategy(df)"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                return f"import not allowed: {bad[0] or 'relative import'} (only numpy, pandas, math)"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"{node.id} is not allowed"
        elif isinstance(node, ast.Attribute) and (
                node.attr.startswith(("__", "read_")) or node.attr in FILE_READERS):
            return f".{node.attr} is not allowed"
    if YEAR.search(code):
        return ("numbers that look like calendar years (2009-2029) are not allowed, even in comments: "
                "no dates, and write a window such as 2016 as 24 * 84")
    return None
