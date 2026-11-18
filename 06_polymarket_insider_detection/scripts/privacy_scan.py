"""Privacy scan: no wallet-like address anywhere outside ``data/``.

A wallet address is exactly ``0x`` + 40 hex digits (a bare 40-hex string counts too), word-bounded, so
64-hex condition ids and transaction hashes do not match. Everything under the project folder is scanned
except ``data/`` (private by design), virtualenvs, caches, logs and ``demo_output/``. Text files are read
as text; Parquet files are scanned by their string columns; PNG/JPEG files are scanned as raw bytes.

The only allowed 40-hex strings are the public contract / burn addresses in ``ALLOWED_CONTRACTS``
(Polymarket exchange, conditional-token and adapter contracts, USDC.e, the zero and dead addresses).
They are not wallets of persons.

    python scripts/privacy_scan.py            # exit code 0 = clean, 1 = a finding (paths only, never the value)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERN = re.compile(r"(?<![0-9A-Za-z])(?:0x)?[0-9a-fA-F]{40}(?![0-9A-Za-z])")

# Public contract addresses (not wallets of persons). Keep in step with src/pmdetect/wallets/filters.py.
ALLOWED_CONTRACTS = frozenset(
    a.lower()
    for a in (
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # CTF Exchange
        "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # Neg Risk CTF Exchange
        "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",  # Conditional Tokens
        "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",  # Neg Risk Adapter
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC.e
        "0x3A3BD7bb9528E159577F7C2e685CC81A765002E2",  # Neg Risk fee module
        "0x0000000000000000000000000000000000000000",  # zero address
        "0x000000000000000000000000000000000000dead",  # burn address
    )
)

SKIP_DIRS = {
    "data", ".venv", "venv", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    "logs", "demo_output", ".idea", "node_modules", "build", "dist",
}  # fmt: skip
TEXT_SUFFIXES = {
    ".md", ".txt", ".csv", ".json", ".jsonl", ".py", ".toml", ".yml", ".yaml", ".sql", ".bat",
    ".cfg", ".ini", ".html", ".svg", ".gitignore", "",
}  # fmt: skip
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _is_allowed(token: str) -> bool:
    t = token.lower()
    if not t.startswith("0x"):
        t = "0x" + t
    return t in ALLOWED_CONTRACTS


def _hits_in_text(text: str) -> int:
    return sum(1 for m in PATTERN.finditer(text) if not _is_allowed(m.group(0)))


def _hits_in_parquet(path: Path) -> int:
    import polars as pl

    n = 0
    df = pl.read_parquet(path)
    for col, dtype in df.schema.items():
        if dtype == pl.Utf8:
            s = df[col].drop_nulls().unique().to_list()
            n += sum(_hits_in_text(x) for x in s)
    return n


def iter_files(root: Path = ROOT):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        yield p


def scan(root: Path = ROOT) -> list[tuple[Path, int]]:
    """Return ``[(path, n_findings)]`` for every file with at least one non-allowed 40-hex string."""
    findings: list[tuple[Path, int]] = []
    for p in iter_files(root):
        suf = p.suffix.lower()
        try:
            if suf in BINARY_SUFFIXES:
                n = _hits_in_text(p.read_bytes().decode("latin-1"))
            elif suf == ".parquet":
                n = _hits_in_parquet(p)
            elif suf in TEXT_SUFFIXES or p.name.startswith("."):
                n = _hits_in_text(p.read_text(encoding="utf-8", errors="ignore"))
            else:
                continue
        except Exception as exc:  # unreadable file: report, do not hide
            print(f"WARNING: could not scan {p.relative_to(root)}: {exc}", file=sys.stderr)
            continue
        if n:
            findings.append((p, n))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("privacy scan: clean (no 40-hex address outside data/, allow-listed contracts aside)")
        return 0
    for p, n in findings:
        print(f"FINDING: {p.relative_to(ROOT)}: {n} address-like string(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
