#!/usr/bin/env python3
"""List all Gym environments registered under ``atec_rl_lab.tasks`` without importing Isaac Sim/Kit.

This script avoids `import atec_rl_lab.tasks` because that may import IsaacLab/Omniverse modules (e.g., `carb`)
in a pure Python environment. Instead, it scans task modules and executes only `gym.register(...)` statements.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import gymnasium as gym
from prettytable import PrettyTable


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_python_files(pkg_dir: Path) -> list[Path]:
    return [
        p for p in pkg_dir.rglob("*.py")
        if p.name != "__pycache__"
    ]


def _exec_only_gym_register(py_file: Path) -> bool:
    """Parse a python file and execute only top-level `gym.register(...)` calls.

    Returns True if at least one register call was executed.
    """
    src = py_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_file))

    register_calls: list[ast.stmt] = []
    for node in tree.body:

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            else:
                mod = node.module or ""
                names = [mod]
            if any("gym" in n for n in names):
                register_calls.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            # detect gym.register(...)
            if isinstance(call.func, ast.Attribute) and call.func.attr == "register":
                if isinstance(call.func.value, ast.Name) and call.func.value.id == "gym":
                    register_calls.append(node)

    if not register_calls:
        return False

    mod = ast.Module(body=register_calls, type_ignores=[])
    code = compile(mod, filename=str(py_file), mode="exec")

    # Execute in a controlled namespace: provide gymnasium as gym
    ns: dict = {"gym": gym}
    try:
        exec(code, ns, ns)
        return True
    except Exception as e:
        print(f"[WARN] Failed executing gym.register in {py_file}: {e}", file=sys.stderr)
        return False


def discover_tasks_without_import() -> int:
    root = _project_root()
    pkg_dir = root / "source" / "atec_rl_lab" / "atec_rl_lab" / "tasks"
    if not pkg_dir.exists():
        pkg_dir = root / "atec_rl_lab" / "tasks"

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Cannot find tasks package dir under: {root}")

    count = 0
    for py in _find_python_files(pkg_dir):
        if py.name in {"env_cfg.py", "scene_cfg.py", "envs_base_cfg.py"}:
            continue
        if _exec_only_gym_register(py):
            count += 1
    return count


def main() -> None:
    executed_files = discover_tasks_without_import()

    table = PrettyTable(["S. No.", "Task Name", "Entry Point"])
    table.title = f"Available Environments in ATEC RL Lab (scanned {executed_files} files)"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "ATEC" in task_spec.id and "Isaac" not in task_spec.id:
            table.add_row([index + 1, task_spec.id, str(task_spec.entry_point)])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"Failed to import dependencies while discovering tasks: {exc}")
        raise
