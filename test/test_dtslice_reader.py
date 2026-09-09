# -*- coding: utf-8 -*-
"""Regression: layout binario dei file .DT (IDS raw).

Il file reale LA010001.DT usava un layout 'a blocchi' (1 traccia per blocco
da 1028 byte = 4 byte di intestazione + 512 campioni int16) che il reader
interpretava come stream contiguo con n_samples=200 dall'header, producendo
un radargramma disallineato (rumore a strisce verticali).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# import piatto del modulo (senza package), come gli altri test headless
_MOD = "gpr_dtslice_reader"
if _MOD not in sys.modules:
    _root = Path(__file__).resolve().parent.parent
    for dep in ("gpr_ogpr_reader", _MOD):
        src = (_root / f"{dep}.py").read_text(encoding="utf-8").replace(
            "from .gpr_", "from gpr_"
        )
        spec = importlib.util.spec_from_file_location(dep, str(_root / f"{dep}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dep] = mod
        exec(compile(src, f"{dep}.py", "exec"), mod.__dict__)

dt = sys.modules[_MOD]

HB = dt.HEADER_BLOCK_SIZE            # 1028
NS = dt._SAMPLES_PER_TRACE_BLOCK     # 512


def _v0_header_bytes(n_samples_field: int, time_window_s: float, dx_m: float) -> bytes:
    """10 blocchi header: il primo e' un V0 ASCII plausibile, gli altri vuoti."""
    def f(x):
        return ("%16.6E" % x).encode("ascii")
    fields = b"".join([
        f(n_samples_field), f(1.0e8), f(time_window_s), f(time_window_s),
        f(n_samples_field), f(time_window_s), f(dx_m), f(dx_m), f(dx_m), f(6.0 / 512),
    ])
    blk0 = b"V0\x00\x00\x04\x04\x00\x00" + fields
    blk0 = blk0.ljust(HB, b" ")
    return blk0 + b" " * (HB * 9)


def _blocked_dt(n_traces: int, seed: int = 0) -> tuple[bytes, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(-3000, 3000, size=(n_traces, NS)).astype("<i2")
    samples[:, 5] = 9000            # onda diretta coerente al campione 5
    body = bytearray()
    for k in range(n_traces):
        flag = 0x07 if k in (0, n_traces - 1) else 0x00
        body += bytes([0x52, flag, 0, 0])                       # int32 marker
        body += int(1_000_000 + k).to_bytes(4, "little", signed=True)  # counter
        body += samples[k].tobytes()
    raw = _v0_header_bytes(200, 1.2e-7, 0.0324) + bytes(body)
    return raw, samples.T.astype(np.float32)   # (NS, n_traces)


class BlockedLayoutTest(unittest.TestCase):
    def test_reads_blocked_layout_not_header_nsamples(self):
        raw, expect = _blocked_dt(307)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "LA010001.DT"
            p.write_bytes(raw)
            data, header = dt._read_dt_binary(p)
        self.assertEqual(data.shape, (NS, 307))          # non (200, ...)
        self.assertEqual(header["n_samples"], NS)
        self.assertEqual(header["trace_layout"], "blocked")
        self.assertEqual(header["data_start"], 10 * HB)
        np.testing.assert_array_equal(data, expect)
        # onda diretta ricostruita sulla riga giusta
        row_energy = np.abs(data).mean(axis=1)
        self.assertEqual(int(np.argmax(row_energy)), 5)

    def test_find_trace_block_start(self):
        raw, _ = _blocked_dt(20)
        self.assertEqual(dt._find_trace_block_start(raw), 10 * HB)
        self.assertIsNone(dt._find_trace_block_start(raw[:-1]))   # non divisibile

    def test_contiguous_fallback_still_used(self):
        # nessun marker di blocco -> percorso storico (n_samples dall'header)
        ns = 128
        n_traces = 40
        rng = np.random.default_rng(1)
        body = rng.integers(-2000, 2000, size=(n_traces, ns)).astype("<i2").tobytes()
        raw = _v0_header_bytes(ns, 1.2e-7, 0.0324) + body
        # rendi la size NON multipla di 1028 cosi' il ramo 'blocked' e' escluso
        raw = raw + b"\x00"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "OLD.DT"
            p.write_bytes(raw)
            data, header = dt._read_dt_binary(p)
        self.assertEqual(header["trace_layout"], "contiguous")
        self.assertEqual(data.shape[0], ns)


if __name__ == "__main__":
    unittest.main()
