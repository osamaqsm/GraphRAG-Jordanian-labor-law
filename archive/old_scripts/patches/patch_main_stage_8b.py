from __future__ import annotations

from pathlib import Path


MAIN_PATH = Path("app/main.py")
IMPORT_LINE = "from app.generation_api import register_generation_routes"
REGISTER_LINE = "register_generation_routes(app)"


def main() -> int:
    if not MAIN_PATH.exists():
        raise FileNotFoundError(MAIN_PATH)

    text = MAIN_PATH.read_text(encoding="utf-8")
    original = text

    if IMPORT_LINE not in text:
        lines = text.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index + 1
        lines.insert(insert_at, IMPORT_LINE)
        text = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if REGISTER_LINE not in text:
        retrieval_registration = "register_retrieval_routes(app)"
        if retrieval_registration not in text:
            raise RuntimeError(
                "Stage 8-A registration was not found. Install and register "
                "Stage 8-A before Stage 8-B."
            )
        text = text.replace(
            retrieval_registration,
            retrieval_registration + "\n" + REGISTER_LINE,
            1,
        )

    if text == original:
        print("Stage 8-B was already registered.")
        return 0

    backup = MAIN_PATH.with_suffix(".py.stage_8b_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    MAIN_PATH.write_text(text, encoding="utf-8")
    print(f"Patched: {MAIN_PATH}")
    print(f"Backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
