import pytest
from tribescore import memguard

# A representative `vm_stat` block (page size 16384). active=3,000,000 +
# wired=1,000,000 + compressor=500,000 = 4,500,000 pages × 16384 ≈ 73.7 GB.
SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          100000.
Pages active:                       3000000.
Pages inactive:                      200000.
Pages speculative:                     5000.
Pages throttled:                          0.
Pages wired down:                   1000000.
Pages purgeable:                       1000.
"Translation faults":              999999999.
Pages copy-on-write:                 100000.
Pages zero filled:                 500000000.
Pages reactivated:                  1000000.
Pages purged:                        100000.
File-backed pages:                   400000.
Anonymous pages:                    2800000.
Pages stored in compressor:         1200000.
Pages occupied by compressor:        500000.
"""

def test_parse_vm_stat_sums_active_wired_compressed():
    used = memguard._parse_vm_stat(SAMPLE)
    assert used == pytest.approx(4_500_000 * 16384 / 1e9, rel=1e-6)  # ≈ 73.7 GB

def test_headroom_uses_gate_minus_used_minus_safety(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 73.7)
    assert memguard.headroom_gb(gate_gb=105.0, safety_gb=10.0) == pytest.approx(21.3, abs=0.1)

def test_require_headroom_raises_when_below_floor(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 90.0)  # headroom = 105-90-10 = 5
    with pytest.raises(MemoryError, match="headroom"):
        memguard.require_headroom(min_gb=25.0)

def test_require_headroom_ok_when_room(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 40.0)  # headroom = 55
    memguard.require_headroom(min_gb=25.0)  # no raise

def test_check_or_abort_raises_above_ceiling(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 92.0)
    with pytest.raises(MemoryError, match="90"):
        memguard.check_or_abort(abort_gb=90.0)
