"""Check the book: every idea file is listed in SUMMARY.md and every relative link points to a file. usage: python3 tools/check.py"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent; bad = 0
listed = set(re.findall(r"\((findings/[^)]+)\)", (root / "SUMMARY.md").read_text()))
for f in sorted((root / "findings").glob("*.md")):
    if f"findings/{f.name}" not in listed: print("not in SUMMARY.md:", f.name); bad += 1
for f in root.rglob("*.md"):
    for link in re.findall(r"\]\(([^)#]+)\)", f.read_text()):
        if not link.startswith("http") and not (f.parent / link).exists(): print(f"broken link in {f.relative_to(root)}: {link}"); bad += 1
print("ok" if not bad else f"{bad} problems")
