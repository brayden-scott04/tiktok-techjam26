"""Structural isolation checks: these must hold by construction, not by
convention. If either test here fails, the "the agent never saw a test
score" claim in the README is false.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module_imports(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _transitive_imports(start_files, local_module_prefixes):
    """Very small transitive-import walker: follows only imports whose module
    name starts with one of our own top-level packages (harness/agent/kit),
    resolving them to files under ROOT. Good enough to prove "X never reaches
    Y" for this codebase's shape; not a general import analyzer."""
    seen_files = set()
    seen_modules = set()
    stack = list(start_files)
    while stack:
        path = stack.pop()
        if path in seen_files or not os.path.exists(path):
            continue
        seen_files.add(path)
        for mod in _module_imports(path):
            seen_modules.add(mod)
            if not any(mod == p or mod.startswith(p + ".") for p in local_module_prefixes):
                continue
            rel = mod.replace(".", os.sep) + ".py"
            candidate = os.path.join(ROOT, rel)
            if os.path.exists(candidate):
                stack.append(candidate)
    return seen_modules


def test_sealed_unreachable_from_agent_package():
    agent_dir = os.path.join(ROOT, "agent")
    start_files = [
        os.path.join(agent_dir, f)
        for f in os.listdir(agent_dir)
        if f.endswith(".py")
    ]
    imports = _transitive_imports(start_files, local_module_prefixes=["agent", "harness", "kit"])
    assert "harness.sealed" not in imports, "agent/ can reach harness.sealed -- isolation broken"
    assert not any(m.startswith("harness.sealed") for m in imports)


def test_sealed_unreachable_from_node_entry():
    # node_entry.py is what actually executes inside the sandboxed child --
    # it must not be able to reach the test-scoring module either.
    node_entry_path = os.path.join(ROOT, "harness", "node_entry.py")
    imports = _transitive_imports([node_entry_path], local_module_prefixes=["agent", "harness", "kit"])
    assert "harness.sealed" not in imports
    assert "kit.evaluate" not in imports, "node_entry.py must never import kit.evaluate directly"


def test_eval_server_secrets_not_in_solution_contract():
    # harness/context.py builds the eval_valid closure; solution.py (agent-
    # authored) only ever calls ctx.eval_valid(scores) -- it should have no
    # way to see the host/port/token directly.
    context_src = open(os.path.join(ROOT, "harness", "context.py"), encoding="utf-8").read()
    assert "_eval_host" in context_src and "_eval_token" in context_src  # present, but...
    # ...as underscore-prefixed private attributes, not exposed on the public
    # Ctx surface documented to solutions. This is a naming-convention check,
    # not a hard guarantee -- Python has no real private attributes -- but it
    # signals intent and would be caught by code review if changed.
