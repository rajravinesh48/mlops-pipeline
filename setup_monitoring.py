"""Idempotently enable Prometheus metrics without overwriting src/app.py.

Run from the repository root: python setup_monitoring.py
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "src" / "app.py"
REQUIREMENTS = ROOT / "requirements.txt"
GITIGNORE = ROOT / ".gitignore"
INJECTION = "from .monitoring import setup_monitoring\nsetup_monitoring(app, lambda: model is not None)"


def main():
    if not APP.is_file():
        raise SystemExit("Expected src/app.py in the repository. Run from the project root.")

    content = APP.read_text(encoding="utf-8")
    if "setup_monitoring(app" not in content:
        tree = ast.parse(content, filename=str(APP))
        candidates = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and (
                isinstance(node.value.func, ast.Name) and node.value.func.id == "FastAPI"
                or isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "FastAPI"
            )
        ]
        if len(candidates) != 1:
            raise SystemExit("Expected one top-level `app = FastAPI(...)` assignment; no file changed.")
        insertion_line = candidates[0].end_lineno
        lines = content.splitlines(keepends=True)
        lines.insert(insertion_line, "\n" + INJECTION + "\n")
        revised = "".join(lines)
        ast.parse(revised, filename=str(APP))  # Validation before writing.
        APP.write_text(revised, encoding="utf-8")
        print("Enabled /metrics instrumentation in src/app.py")
    else:
        print("Monitoring integration already exists in src/app.py")

    if not REQUIREMENTS.exists():
        raise SystemExit("requirements.txt missing; add prometheus-client manually.")
    dependencies = REQUIREMENTS.read_text(encoding="utf-8")
    if not any(line.strip().lower().startswith(("prometheus-client", "prometheus_client"))
               for line in dependencies.splitlines()):
        REQUIREMENTS.write_text(dependencies.rstrip() + "\nprometheus-client>=0.20,<1\n", encoding="utf-8")
        print("Added prometheus-client to requirements.txt")

    ignore = GITIGNORE.read_text(encoding="utf-8") if GITIGNORE.exists() else ""
    if ".env.monitoring" not in ignore.splitlines():
        GITIGNORE.write_text(ignore.rstrip() + "\n.env.monitoring\n", encoding="utf-8")
        print("Excluded .env.monitoring from Git")
    print("Done. Install dependencies, run pytest, then start your app and monitoring stack.")


if __name__ == "__main__":
    main()
