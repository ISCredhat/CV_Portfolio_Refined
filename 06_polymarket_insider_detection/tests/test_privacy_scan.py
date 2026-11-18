"""The privacy scan: no wallet-like 40-hex address anywhere outside data/ (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pmdetect.wallets.filters import KNOWN_CONTRACTS

ROOT = Path(__file__).resolve().parents[1]


def _load_scan():
    spec = importlib.util.spec_from_file_location(
        "privacy_scan", ROOT / "scripts" / "privacy_scan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_project_has_no_wallet_addresses():
    scan = _load_scan()
    assert scan.scan(ROOT) == []


def test_allow_list_is_exactly_the_contract_constants():
    scan = _load_scan()
    assert set(scan.ALLOWED_CONTRACTS) == set(KNOWN_CONTRACTS)


def test_scan_matches_40_hex_but_not_64_hex(tmp_path):
    scan = _load_scan()
    wallet = "0x" + "ab12" * 10  # 40 hex digits
    cond = "0x" + "ab12" * 16  # 64 hex digits (condition id / tx hash)
    (tmp_path / "a.md").write_text(f"wallet {wallet}\n", encoding="utf-8")
    (tmp_path / "b.md").write_text(f"market {cond}\n", encoding="utf-8")
    (tmp_path / "c.md").write_text(f"usdc {next(iter(KNOWN_CONTRACTS))}\n", encoding="utf-8")
    (tmp_path / "d.md").write_text(
        "fake 0xwallet00000000000000000000000000000001\n", encoding="utf-8"
    )
    found = {p.name for p, _ in scan.scan(tmp_path)}
    assert found == {"a.md"}
