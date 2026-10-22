"""Simulated analyzer logs in the rig logger's format, for the demo store and tests.

Every file starts with the simulated marker, so the lab store refuses it. A run reads the inlet in bypass at the
start of each step, then holds each setpoint while temperature and outlet settle toward the pack model's steady
state, with the protocol analyzer's reading noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from catalyst_engine.data.store import SIMULATED_MARKER, now
from catalyst_engine.packs import Pack, ProtocolSpec
from catalyst_engine.units import k_to_f


def simulated_log(pack: Pack, protocol: ProtocolSpec, params: dict[str, float] | None = None, *, seed: int = 0,
                  rig: str = "simulated rig", sample: str = "", operator: str = "simulator", started_at: str = "",
                  every_s: float = 10.0, bypass_s: float = 300.0, hold_s: float = 600.0, tau_s: float = 60.0,
                  tau_T_s: float = 90.0, stop_after_points: int | None = None,
                  short_holds: dict[tuple[str, float], float] | None = None) -> str:
    """The log text. `stop_after_points` ends the run early (a failed run); `short_holds` maps
    (step, setpoint °F) to a hold time in seconds that ends before the readings settle."""
    rng = np.random.default_rng(seed)
    an = protocol.analyzer
    species = an.measured
    steady = {(r["step"], round(k_to_f(r["T_K"]), 1)): r for r in pack.simulate(params, protocol.id)}
    short = {(s, round(t, 1)): v for (s, t), v in (short_holds or {}).items()}

    def noisy(v: float) -> float:
        return max(0.0, v + rng.normal(0.0, np.hypot(an.sigma_abs_ppm, an.sigma_rel * v)))

    lines = [SIMULATED_MARKER, f"# rig: {rig}", f"# protocol: {protocol.id}", f"# sample: {sample}",
             f"# operator: {operator}", f"# started_at: {started_at or now()}",
             "time_s,step,setpoint_F,mode,T_F," + ",".join(f"{s} (ppm)" for s in species)]
    t, temp, done = 0.0, 300.0, 0
    for step in protocol.steps:
        inlet = step.feed_fractions(pack.network.species, pack.spec.water) * 1e6
        inlet_ppm = {s: inlet[pack.network.index(s)] for s in species}
        setpoints = [round(k_to_f(x), 1) for x in step.kelvin()]
        start_T = temp
        for k in range(int(bypass_s / every_s)):
            temp = setpoints[0] + (start_T - setpoints[0]) * np.exp(-k * every_s / tau_T_s)
            lines.append(f"{t:.0f},{step.id},{setpoints[0]:g},bypass,{temp + rng.normal(0, 0.5):.1f},"
                         + ",".join(f"{noisy(inlet_ppm[s]):.3f}" for s in species))
            t += every_s
        reading = dict(inlet_ppm)
        for sp in setpoints:
            if stop_after_points is not None and done >= stop_after_points:
                return "\n".join(lines) + "\n"
            target = steady[(step.id, sp)]
            start_T, start = temp, dict(reading)
            for k in range(int(short.get((step.id, sp), hold_s) / every_s)):
                decay_T, decay = np.exp(-k * every_s / tau_T_s), np.exp(-k * every_s / tau_s)
                temp = sp + (start_T - sp) * decay_T
                reading = {s: target[f"out_{s}_ppm"] + (start[s] - target[f"out_{s}_ppm"]) * decay for s in species}
                lines.append(f"{t:.0f},{step.id},{sp:g},reactor,{temp + rng.normal(0, 0.5):.1f},"
                             + ",".join(f"{noisy(reading[s]):.3f}" for s in species))
                t += every_s
            done += 1
    return "\n".join(lines) + "\n"


def write_simulated_log(path: Path, *args, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(simulated_log(*args, **kwargs))
    return path
