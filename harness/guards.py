"""Static safety gate for agent-generated solution code, run BEFORE execution.

Defense-in-depth layer on top of the structural guarantee (valid/test labels
are never present in the sandboxed process — see harness/context.py). This
scan exists to produce an explicit, loggable "attempted violation" event, and
to catch mistakes (banned imports, accidental file I/O) before they ever reach
a subprocess.
"""
import ast

ALLOWED_IMPORTS = {
    "numpy", "math", "collections", "itertools", "functools", "operator",
    "heapq", "bisect", "random", "time", "dataclasses", "typing", "enum",
    "warnings", "array", "abc",
}

BANNED_NAME_PATTERNS = [
    "open", "os", "io", "pathlib", "shutil", "glob", "subprocess", "socket",
    "urllib", "http", "requests", "importlib", "ctypes", "mmap", "pickle",
    "marshal", "eval", "exec", "__import__", "globals", "compile", "input",
    "exit", "quit", "breakpoint",
]

# Structurally pointless given the label-stripping in harness/context.py (a hit
# finds only -1), but rejecting it produces good evidence of an attempted
# label-access, which is worth having in the run log for judges.
PEEK_STRING_SUBSTRINGS = [
    "log_standard", "log_random", "kuairand", ".csv", "..",
]


class GuardViolation(Exception):
    def __init__(self, kind, detail):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


def _check_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    raise GuardViolation("banned_import", top)
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in ALLOWED_IMPORTS:
                raise GuardViolation("banned_import", top)


def _check_names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAME_PATTERNS:
            raise GuardViolation("banned_name", node.id)
        if isinstance(node, ast.Attribute) and node.attr in BANNED_NAME_PATTERNS:
            raise GuardViolation("banned_attribute", node.attr)


def _check_peek_strings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            for pat in PEEK_STRING_SUBSTRINGS:
                if pat in low:
                    raise GuardViolation("peek_string", f"literal contains {pat!r}: {node.value!r}")


def _check_test_label_subscript(tree):
    """Reject ctx.splits['test'].<outcome column> and similar label-shaped
    attribute chains on valid/test. Pointless against the actual data (it's
    all -1) but a good explicit signal to log."""
    from harness.task_spec import OUTCOME_COLUMNS

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in OUTCOME_COLUMNS:
            # walk up to see if this chain roots in something indexed by 'test' or 'valid'
            src = ast.dump(node)
            if "'test'" in src or "'valid'" in src or '"test"' in src or '"valid"' in src:
                raise GuardViolation("label_access_attempt", ast.unparse(node))


def check_source(source: str):
    """Raises GuardViolation on the first problem found. Returns True if clean."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise GuardViolation("syntax_error", str(e))
    _check_imports(tree)
    _check_names(tree)
    _check_peek_strings(tree)
    _check_test_label_subscript(tree)
    return True


def ast_normalized_hash(source: str) -> str:
    """Hash of the parsed AST dump: insensitive to comments, whitespace, and
    blank lines, but NOT to variable renaming (ast.dump includes identifier
    names). This catches copy-paste-identical or reformatted duplicates; it
    will not catch a semantically identical solution the LLM rewrote with
    different variable names. That's an accepted scope limit -- true rename
    -insensitive dedup needs real normalization (alpha-renaming) that isn't
    worth building for this. Used for duplicate-node detection in agent/policy.py."""
    import hashlib

    tree = ast.parse(source)
    dump = ast.dump(tree, annotate_fields=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
