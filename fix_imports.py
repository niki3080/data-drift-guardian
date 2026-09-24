"""
Заменяет `from drift_guardian` / `import drift_guardian`
на `from drift_guardian` / `import drift_guardian` во всех .py файлах проекта.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".idea"}

PATTERN = re.compile(r"\b(from|import)\s+src\.(drift_guardian\b)")

# Кодировки, которые пробуем по порядку
ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "cp1252"]


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def read_with_fallback(path: Path):
    """Пытается прочитать файл в разных кодировках, возвращает (text, encoding)."""
    last_error = None
    for enc in ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
            return text, enc
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise last_error


def fix_file(path: Path) -> bool:
    try:
        text, enc = read_with_fallback(path)
    except UnicodeDecodeError:
        print(f"[SKIP] {path.relative_to(ROOT)} — не удалось определить кодировку, файл пропущен")
        return False

    new_text, count = PATTERN.subn(r"\1 \2", text)
    if count > 0:
        # Сохраняем обратно строго в UTF-8, чтобы привести файл к единому стандарту
        path.write_text(new_text, encoding="utf-8")
        note = f" (было в {enc})" if enc != "utf-8" else ""
        print(f"[OK] {path.relative_to(ROOT)} — заменено {count} импорт(ов){note}")
        return True
    return False


def main():
    py_files = [p for p in ROOT.rglob("*.py") if not should_skip(p)]

    changed = 0
    skipped = []
    for f in py_files:
        try:
            if fix_file(f):
                changed += 1
        except Exception as e:
            print(f"[ERROR] {f.relative_to(ROOT)} — {e}")
            skipped.append(f)

    print(f"\nГотово. Изменено файлов: {changed} из {len(py_files)} проверенных .py файлов.")
    if skipped:
        print(f"Файлы с ошибками: {len(skipped)}")
        for f in skipped:
            print(f"  - {f}")


if __name__ == "__main__":
    main()

