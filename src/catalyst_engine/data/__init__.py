"""Data platform: catalysts, samples, runs and raw analyzer files with provenance, plus the prediction log.

Two stores never mix: `lab` holds real measurements, `simulated` holds everything a simulator made.
Claims (hit rates, gate scores) come from the lab store only.
"""
