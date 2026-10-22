"""`ce`: command line for the catalyst engine."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from catalyst_engine.data.cli import data_app
from catalyst_engine.packs import list_packs, load_pack
from catalyst_engine.units import f_to_k, k_to_f

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Closed-loop catalyst discovery engine.")
app.add_typer(data_app, name="data")
console = Console()


def _parse_sets(sets: list[str] | None) -> dict[str, float]:
    out = {}
    for item in sets or []:
        name, _, value = item.partition("=")
        if not value:
            raise typer.BadParameter(f"expected name=value, got {item!r}")
        out[name.strip()] = float(value)
    return out


def _num(v: float) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:.3f}"


def _params_file(path: Optional[Path]) -> dict[str, float]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text())
    return {k: float(v) for k, v in (data.get("params", data)).items()}


@app.command()
def packs() -> None:
    """List reaction packs and check that each one loads."""
    table = Table("pack", "name", "species", "pathways", "protocols", "status")
    for pid in list_packs():
        try:
            p = load_pack(pid)
            table.add_row(pid, p.spec.name, str(len(p.network.species)), str(len(p.network.pathways)),
                          ", ".join(p.protocols), p.spec.status)
        except Exception as exc:  # report every broken pack, not just the first
            table.add_row(pid, f"[red]failed: {exc}[/red]", "", "", "", "")
    console.print(table)


@app.command()
def show(pack: str) -> None:
    """Show a pack's pathways, default parameters, metrics and protocols."""
    p = load_pack(pack)
    console.print(f"[bold]{p.spec.name}[/bold]  ({p.id} v{p.spec.version}, {p.spec.status})")
    console.print(f"Reference temperature {k_to_f(p.network.t_ref):.0f} F ({p.network.t_ref:.1f} K)\n")
    t = Table("pathway", "role", "equation", "k_ref (SI)", "Ea kJ/mol")
    for pid, pw in p.spec.pathways.items():
        t.add_row(pid, pw.role, pw.equation, f"{np.exp(pw.rate.ln_k_ref):.3g}", f"{pw.rate.Ea_kJ:g}")
    console.print(t)
    m = Table("metric", "kind", "unit", "description")
    for mid, spec in p.spec.metrics.items():
        m.add_row(mid, spec.kind, spec.label_unit(), spec.description or "")
    console.print(m)
    for proto in p.protocols.values():
        console.print(f"\n[bold]Protocol {proto.id}[/bold] v{proto.version} ({proto.status}): "
                      f"{', '.join(s.id for s in proto.steps)}")
        for q in proto.open_questions:
            console.print(f"  [yellow]open question:[/yellow] {q}")


@app.command()
def simulate(
    pack: str,
    protocol: Annotated[Optional[str], typer.Option(help="Protocol id; required when a pack has several")] = None,
    params: Annotated[Optional[Path], typer.Option(help="YAML of parameter values (e.g. a fit's output)")] = None,
    set_: Annotated[Optional[list[str]], typer.Option("--set", help="Override a parameter: name=value")] = None,
) -> None:
    """Run a protocol through the reactor model and print the pack's metrics."""
    p = load_pack(pack)
    values = _params_file(params) | _parse_sets(set_)
    rows = p.simulate(values, protocol)
    t = Table("step", "T (F)", *p.spec.metrics)
    for r in rows:
        t.add_row(r["step"], f"{k_to_f(r['T_K']):.0f}", *(_num(r[m]) for m in p.spec.metrics))
    console.print(t)
    console.print("Units: " + ", ".join(f"{m} {spec.label_unit()}" for m, spec in p.spec.metrics.items()))


@app.command()
def fit(
    pack: str,
    data: Path,
    protocol: Annotated[Optional[str], typer.Option()] = None,
    out: Annotated[Optional[Path], typer.Option(help="Write fitted parameters to this YAML file")] = None,
    starts: Annotated[int, typer.Option(help="Multi-start count")] = 6,
) -> None:
    """Fit pathway parameters to bench data (CSV: step, T_F, out_<species>_ppm ...)."""
    from catalyst_engine.fit.kinetics import fit_kinetics, read_observations

    p = load_pack(pack)
    proto = p.protocol(protocol)
    obs = read_observations(p, proto, data)
    res = fit_kinetics(p, proto, obs, n_starts=starts)
    t = Table("parameter", "estimate", "1-sigma", "")
    for n in res.fitted:
        flag = "[yellow]poorly determined[/yellow]" if n in res.poorly_determined else ""
        t.add_row(n, f"{res.params[n]:.4g}", f"{res.std[n]:.3g}", flag)
    console.print(t)
    console.print(f"{res.n_points} points, reduced chi-square {res.chi2_reduced:.2f}")
    for pw in p.network.pathways:
        k, factor = res.rate_constant_ref(pw.id)
        console.print(f"  k_ref[{pw.id}] = {k:.4g} (x/÷ {factor:.3f}) at {k_to_f(p.network.t_ref):.0f} F")
    if res.poorly_determined:
        console.print("[yellow]Poorly determined parameters mean the protocol cannot separate them; "
                      "extend the protocol rather than trusting these values.[/yellow]")
    if out:
        out.write_text(yaml.safe_dump({"pack": p.id, "protocol": proto.id,
                                       "params": {n: res.params[n] for n in res.fitted},
                                       "std": {n: res.std[n] for n in res.fitted}}, sort_keys=False))
        console.print(f"wrote {out}")


@app.command()
def decay(
    data: Path,
    project: Annotated[list[float], typer.Option(help="Hours to project activity to")] = [2000.0, 40000.0],
    end_of_life: Annotated[float, typer.Option(help="Activity at which the catalyst is spent")] = 0.85,
) -> None:
    """Fit aging data (CSV: hours, activity[, sigma]) and project activity and life."""
    from catalyst_engine.fit.decay import fit_decay

    with data.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    t_h = [float(r["hours"]) for r in rows]
    a = [float(r["activity"]) for r in rows]
    sig = [float(r["sigma"]) for r in rows] if rows and "sigma" in rows[0] else None
    res = fit_decay(t_h, a, sig)
    console.print(f"model {res.model}: kd = {res.kd:.3g} 1/h, levels off at {res.a_eq:.3f} (AICc {res.aicc:.1f})")
    for hours in project:
        lo, hi = res.interval(hours)
        ratio = res.extrapolation_ratio(hours)
        warn = f"  [yellow]{ratio:.0f}x beyond the longest test[/yellow]" if ratio > 3 else ""
        console.print(f"  activity at {hours:g} h: {res.activity(hours):.3f} (90%: {lo:.3f}-{hi:.3f}){warn}")
    life = res.hours_to(end_of_life)
    console.print(f"  hours to {end_of_life:.2f} activity: {'beyond the fitted plateau' if np.isinf(life) else f'{life:,.0f}'}")


@app.command()
def scenario(path: Path) -> None:
    """Size and compare emission-control designs for one exhaust (see reactions/*/scenarios)."""
    from catalyst_engine.value.scenario import evaluate_scenario, load_scenario

    spec = load_scenario(path)
    console.print(f"[bold]{spec.name}[/bold]")
    if spec.note:
        console.print(spec.note.strip())
    results = evaluate_scenario(spec)
    t = Table("", *(r.name for r in results))

    def row(label: str, fmt) -> None:
        t.add_row(label, *(fmt(r) for r in results))

    row("catalyst temperature (F)", lambda r: f"{r.catalyst_T_F:.0f}")
    row("meets targets", lambda r: "yes" if r.feasible else "[red]no amount of catalyst[/red]")
    row("tempering air (of exhaust mass)", lambda r: f"{r.tempering_mass_ratio:.0%}")
    row("reagent dose ratio", lambda r: f"{r.reagent_ratio:.2f}" if r.reagent_ratio is not None else "-")
    row("space velocity (1/h)", lambda r: f"{r.ghsv_per_h:,.0f}" if r.ghsv_per_h else "-")
    row("catalyst volume (m3)", lambda r: f"{r.catalyst_volume_m3:.1f}" if r.catalyst_volume_m3 else "-")
    row("bed depth (m)", lambda r: f"{r.bed_depth_m:.2f}" if r.bed_depth_m else "-")
    row("pressure drop (Pa)", lambda r: f"{r.pressure_drop_Pa:.0f}" if r.pressure_drop_Pa else "-")
    row("fan power (kW)", lambda r: f"{r.fan_power_kW:,.0f}")
    row("fan energy ($/yr)", lambda r: f"{r.fan_energy_usd_per_year:,.0f}")
    row("reagent ($/yr)", lambda r: f"{r.reagent_usd_per_year:,.0f}")
    for m in spec.targets:
        row(f"{m} at design", lambda r, m=m: _num(r.metrics[m]) if m in r.metrics else "-")
    console.print(t)


@app.command()
def tempering(
    exhaust_f: Annotated[float, typer.Option("--exhaust-f")],
    target_f: Annotated[float, typer.Option("--target-f")] = 900.0,
    ambient_f: Annotated[float, typer.Option("--ambient-f")] = 80.0,
) -> None:
    """Tempering air needed to cool natural-gas turbine exhaust to a catalyst's limit."""
    from catalyst_engine.value.tempering import tempering_air

    exhaust = {"N2": 0.745, "O2": 0.125, "CO2": 0.039, "H2O": 0.083, "AR": 0.008}
    r = tempering_air(exhaust, f_to_k(exhaust_f), f_to_k(target_f), f_to_k(ambient_f))
    console.print(f"{r.mass_ratio:.1%} of exhaust mass ({r.mole_ratio:.1%} by moles) to cool "
                  f"{exhaust_f:g} F to {target_f:g} F with {ambient_f:g} F air")
    console.print("Example exhaust composition: 74.5% N2, 12.5% O2, 3.9% CO2, 8.3% H2O, 0.8% Ar (by volume, wet)")


@app.command()
def permit(
    ppm: Annotated[float, typer.Option(help="ppmvd @ 15% O2")],
    species: Annotated[str, typer.Option(help="NOx, CO, NH3, HCHO or VOC_as_CH4")] = "NOx",
    heat_rate: Annotated[float, typer.Option(help="Btu/kWh, HHV")] = 9500.0,
    hours: Annotated[float, typer.Option()] = 8760.0,
    cap_tpy: Annotated[float, typer.Option(help="Annual cap, short tons")] = 100.0,
) -> None:
    """Emission rate, tons per year per MW, and the megawatts that fit under an annual cap."""
    from catalyst_engine.value.permits import lb_per_mmbtu, max_mw_under_cap, tons_per_year

    rate = lb_per_mmbtu(ppm, species)
    console.print(f"{ppm:g} ppmvd @ 15% O2 {species} = {rate:.5f} lb/MMBtu "
                  f"= {tons_per_year(rate, 1.0, heat_rate, hours):.3f} tons/yr per MW")
    console.print(f"MW under a {cap_tpy:g} ton/yr cap: {max_mw_under_cap(cap_tpy, rate, heat_rate, hours):,.0f}")



@app.command()
def ui(port: Annotated[int, typer.Option(help="Local port")] = 8501) -> None:
    """Open the engine's app in your browser."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
           "--browser.gatherUsageStats", "false", "--theme.primaryColor", "#0f766e"]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
