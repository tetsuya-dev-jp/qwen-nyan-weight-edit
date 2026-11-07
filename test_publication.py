"""Check tracked content for local paths and accidentally included private files."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
patterns = [
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\"),
    re.compile(r"\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
files = [ROOT / name for name in tracked if name]
assert files, "Stage files before running the publication check."
for path in files:
    relative = path.relative_to(ROOT)
    assert relative.parts[0] not in {".venv", "quality", "model_nyan", "models"}, relative
    assert path.suffix not in {".log", ".pt"} and not path.name.startswith(".env"), relative
    assert path.stat().st_size < 25_000_000, relative
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    for pattern in patterns:
        assert not pattern.search(text), f"Local path or credential in {relative}"

outputs = json.loads((ROOT / "results/heldout_outputs.json").read_text())
assert len(outputs) == 120
assert len({(row["condition"], row["id"]) for row in outputs}) == 120
print(f"Publication checks passed for {len(files)} tracked files.")
