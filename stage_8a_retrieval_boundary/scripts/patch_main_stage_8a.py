from __future__ import annotations

from pathlib import Path


MAIN_PATH = Path("app/main.py")
IMPORT_LINE = "from app.retrieval_api import register_retrieval_routes"
REGISTER_LINE = "register_retrieval_routes(app)"


def main() -> int:
    if not MAIN_PATH.exists():
        raise FileNotFoundError(MAIN_PATH)

    original = MAIN_PATH.read_text(encoding="utf-8")
    updated = original

    if IMPORT_LINE not in updated:
        lines = updated.splitlines()
        insertion = 0
        for index, line in enumerate(lines):
            if line.startswith("from app.") or line.startswith("import app."):
                insertion = index + 1
        lines.insert(insertion, IMPORT_LINE)
        updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if REGISTER_LINE not in updated:
        lines = updated.splitlines()
        insertion = None
        for index, line in enumerate(lines):
            if line.startswith("@app."):
                insertion = index
                break
        if insertion is None:
            insertion = len(lines)
        lines[insertion:insertion] = [REGISTER_LINE, ""]
        updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    backup = MAIN_PATH.with_suffix(".py.stage_8a_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    MAIN_PATH.write_text(updated, encoding="utf-8")

    print(f"Patched: {MAIN_PATH}")
    print(f"Backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
