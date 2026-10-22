"""`ce data`: the lab store and the prediction log from the command line."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich.console import Console
from rich.table import Table

from catalyst_engine.data.ledger import Ledger
from catalyst_engine.data.store import Store, store_path
from catalyst_engine.packs import load_pack

data_app = typer.Typer(no_args_is_help=True, help="Lab data: catalysts, samples, runs and the prediction log.")
console = Console()
StoreOpt = Annotated[Literal["lab", "simulated"], typer.Option("--store", help="lab (real data) or simulated")]


def _open(kind: str) -> Store:
    path = store_path(kind)  # type: ignore[arg-type]
    if not (path / "store.db").exists():
        hint = "ce data demo" if kind == "simulated" else "ce data init"
        raise typer.BadParameter(f"no {kind} store at {path}; create it with `{hint}`")
    return Store(path)


def _pack_for(store: Store, pack: Optional[str], campaign: Optional[str]):
    if pack:
        return load_pack(pack)
    if campaign:
        return load_pack(store.get("campaigns", campaign)["pack"])
    raise typer.BadParameter("give --pack or --campaign")


@data_app.command()
def init() -> None:
    """Create the lab store (once per machine)."""
    path = store_path("lab")
    store = Store.open_or_create(path, "lab")
    console.print(f"lab store ready at {store.root}")


@data_app.command()
def demo(pack: Annotated[Optional[str], typer.Option(help="Pack with a demo.yaml")] = None) -> None:
    """Rebuild the simulated demo store from a pack's demo.yaml."""
    from catalyst_engine.data.demo import build_demo, demo_packs

    choices = demo_packs()
    pid = pack or (choices[0] if choices else None)
    if pid is None:
        raise typer.BadParameter("no pack has a demo.yaml")
    store = build_demo(load_pack(pid))
    console.print(f"simulated demo store rebuilt at {store.root}: " +
                  ", ".join(f"{n} {t}" for t, n in store.counts().items() if n))


@data_app.command("add-formulation")
def add_formulation_cmd(
    name: str,
    component: Annotated[list[str], typer.Option(help="material:role:amount:unit, e.g. V2O5:active:1.5:wt%")],
    role: Annotated[str, typer.Option(help="reference or candidate")] = "candidate",
    preparation: Annotated[str, typer.Option()] = "",
    notes: Annotated[str, typer.Option()] = "",
    supersedes: Annotated[Optional[str], typer.Option(help="Formulation this one corrects")] = None,
    store: StoreOpt = "lab",
) -> None:
    """Add a catalyst formulation."""
    from catalyst_engine.data.records import add_formulation

    comps = []
    for c in component:
        parts = c.split(":")
        if len(parts) != 4:
            raise typer.BadParameter(f"expected material:role:amount:unit, got {c!r}")
        material, crole, amount, unit = parts
        comps.append({"material": material, "role": crole, "amount": float(amount) if amount != "balance" else amount,
                      "unit": unit})
    rec = add_formulation(_open(store), name, comps, role, preparation, notes, supersedes)
    console.print(f"added {rec['id']}: {name}")


@data_app.command("add-sample")
def add_sample_cmd(
    formulation: str,
    form: Annotated[str, typer.Option(help="e.g. monolith core, powder")] = "monolith core",
    mass_g: Annotated[Optional[float], typer.Option()] = None,
    volume_ml: Annotated[Optional[float], typer.Option("--volume-ml")] = None,
    aged_from: Annotated[Optional[str], typer.Option(help="Sample this one was aged from")] = None,
    notes: Annotated[str, typer.Option()] = "",
    store: StoreOpt = "lab",
) -> None:
    """Add a physical sample of a formulation."""
    from catalyst_engine.data.records import add_sample

    rec = add_sample(_open(store), formulation, form, mass_g, volume_ml, aged_from, notes=notes)
    console.print(f"added {rec['id']} ({formulation}, {form})")


@data_app.command("add-campaign")
def add_campaign_cmd(
    name: str,
    pack: Annotated[str, typer.Option()],
    rig: Annotated[str, typer.Option()],
    reference: Annotated[str, typer.Option(help="Reference formulation id")],
    metric: Annotated[str, typer.Option(help="Metric the hit rule scores")],
    step: Annotated[str, typer.Option(help="Protocol step the hit rule reads")],
    temperature: Annotated[list[float], typer.Option(help="°F; repeat for several")],
    protocol: Annotated[Optional[str], typer.Option()] = None,
    rule: Annotated[str, typer.Option(help="beats_reference or meets_target")] = "beats_reference",
    margin: Annotated[float, typer.Option(help="How far past the reference counts as a hit")] = 0.0,
    notes: Annotated[str, typer.Option()] = "",
    store: StoreOpt = "lab",
) -> None:
    """Open a campaign: one protocol version, one rig, one reference, one hit rule, fixed before any pick."""
    from catalyst_engine.data.records import add_campaign

    p = load_pack(pack)
    proto = p.protocol(protocol)
    rec = add_campaign(_open(store), p, name, proto.id, rig, reference,
                       {"metric": metric, "step": step, "temperatures_F": temperature, "rule": rule, "margin": margin},
                       notes)
    console.print(f"added {rec['id']}: {name} (protocol {proto.id}, hash {rec['protocol_hash'][:12]})")


@data_app.command()
def pick(
    campaign: str,
    formulation: str,
    arm: Annotated[str, typer.Option(help="model, expert or random")],
    rationale: Annotated[str, typer.Option()] = "",
    store: StoreOpt = "lab",
) -> None:
    """Log a pick in the prediction log before the catalyst is made or run."""
    e = Ledger(_open(store)).pick(campaign, formulation, arm, rationale)
    console.print(f"logged pick #{e['seq']} ({e['hash'][:12]})")


@data_app.command()
def predict(
    campaign: str,
    formulation: str,
    metric: Annotated[str, typer.Option()],
    step: Annotated[str, typer.Option()],
    temperature: Annotated[float, typer.Option(help="°F")],
    value: Annotated[float, typer.Option()],
    low: Annotated[Optional[float], typer.Option()] = None,
    high: Annotated[Optional[float], typer.Option()] = None,
    model: Annotated[str, typer.Option(help="Model name and version")] = "",
    store: StoreOpt = "lab",
) -> None:
    """Log a numeric prediction (with its interval) before the run."""
    e = Ledger(_open(store)).predict(campaign, formulation, metric, step, temperature, value, low, high, model)
    console.print(f"logged prediction #{e['seq']} ({e['hash'][:12]})")


@data_app.command()
def ingest(
    file: Path,
    sample: Annotated[Optional[str], typer.Option(help="Sample id (or a # sample: header line)")] = None,
    campaign: Annotated[Optional[str], typer.Option()] = None,
    pack: Annotated[Optional[str], typer.Option(help="Needed when the run is outside a campaign")] = None,
    protocol: Annotated[Optional[str], typer.Option()] = None,
    rig: Annotated[Optional[str], typer.Option()] = None,
    operator: Annotated[Optional[str], typer.Option()] = None,
    status: Annotated[str, typer.Option(help="auto, complete, failed or aborted")] = "auto",
    notes: Annotated[str, typer.Option()] = "",
    store: StoreOpt = "lab",
) -> None:
    """Store an analyzer log, derive its steady-state points and log the run."""
    from catalyst_engine.data.ingest import ingest_run

    s = _open(store)
    p = _pack_for(s, pack, campaign)
    run = ingest_run(s, p, file, sample_id=sample, campaign_id=campaign, protocol_id=protocol, rig=rig,
                     operator=operator, status=status, notes=notes)
    d = s.latest_derivation(run["id"])
    console.print(f"stored {run['id']} ({run['status']}), raw {run['raw_sha256'][:12]}, {len(s.points(run['id']))} points")
    for f in d["run_flags"] if d else []:
        console.print(f"  [yellow]{f}[/yellow]")


@data_app.command()
def runs(campaign: Annotated[Optional[str], typer.Option()] = None, store: StoreOpt = "lab") -> None:
    """List runs."""
    from catalyst_engine.data.ingest import issues

    s = _open(store)
    t = Table("run", "sample", "campaign", "status", "rig", "started", "points", "flagged")
    for r in s.current("runs"):
        if campaign and r["campaign_id"] != campaign:
            continue
        pts = s.points(r["id"])
        t.add_row(r["id"], r["sample_id"], r["campaign_id"] or "-", r["status"], r["rig"], r["started_at"][:16],
                  str(len(pts)), str(sum(bool(issues(p)) for p in pts)))
    console.print(t)


@data_app.command("export-fit")
def export_fit(run: str, out: Path, pack: Annotated[Optional[str], typer.Option()] = None,
               store: StoreOpt = "lab") -> None:
    """Write a run's settled points as a CSV that `ce fit` reads."""
    from catalyst_engine.data.ingest import fit_rows
    from catalyst_engine.data.records import stored_protocol

    s = _open(store)
    r = s.get("runs", run)
    p = _pack_for(s, pack, r["campaign_id"])
    rows = fit_rows(p, stored_protocol(s, r["protocol_hash"]), s.points(run))
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    console.print(f"wrote {len(rows)} settled points to {out}; fit with `ce fit {p.id} {out}`")


@data_app.command()
def unblind(campaign: str, store: StoreOpt = "lab") -> None:
    """Reveal the arms of a campaign; picks close at this point."""
    e = Ledger(_open(store)).unblind(campaign)
    console.print(f"{campaign} unblinded (entry #{e['seq']})")


@data_app.command()
def score(campaign: str, store: StoreOpt = "lab") -> None:
    """Hit rates per arm, from the prediction log and the stored runs."""
    from catalyst_engine.data.scoring import score_campaign

    s = _open(store)
    sc = score_campaign(s, _pack_for(s, None, campaign), campaign)
    if sc.simulated:
        console.print("[yellow]Simulated store: these numbers exercise the platform. Claims come from the lab store.[/yellow]")
    console.print(f"reference ({', '.join(sc.reference_runs) or 'awaiting a run'}): " +
                  ", ".join(f"{t:g} F {v:.3f}" if v is not None else f"{t:g} F pending" for t, v in sc.reference.items()))
    t = Table("formulation", "name", "arm", "state", "run", *(f"{x:g} F" for x in sc.campaign["hit_rule"]["temperatures_F"]))
    for p in sc.picks:
        t.add_row(p.formulation, p.name, p.arm or "blinded", p.state, p.run or "-",
                  *(f"{v:.3f}" if v is not None else "-" for v in p.values.values()))
    console.print(t)
    for a in sc.arms:
        rate = f"{a.hit_rate:.0%}" if a.hit_rate is not None else "-"
        console.print(f"  {a.arm}: {a.hits}/{a.tested} hits ({rate}), {a.picked} picked")
    for c in sc.comparisons:
        console.print(f"  model vs {c['against']}: one-sided Fisher p = {c['p_value']:.3f}")
    if sc.coverage is not None:
        console.print(f"  prediction intervals held the measured value {sc.coverage:.0%} of the time")


@data_app.command()
def verify(store: StoreOpt = "lab") -> None:
    """Check the prediction log's hash chain and every raw file's hash."""
    s = _open(store)
    check = Ledger(s).verify()
    bad = [r["sha256"] for r in s.rows("raw_files") if not s.verify_raw(r["sha256"])]
    if check.ok:
        console.print(f"[green]log intact[/green]: {check.entries} entries, head {check.head}")
    else:
        console.print(f"[red]log check failed[/red]: {check.problem}")
    console.print(f"raw files: {len(s.rows('raw_files')) - len(bad)} match their hashes" +
                  (f", [red]{len(bad)} changed[/red]" if bad else ""))
    if bad or not check.ok:
        raise typer.Exit(1)


@data_app.command()
def anchor() -> None:
    """Append the lab log's head hash to docs/prediction_anchors.md; commit that file to timestamp the log."""
    line = Ledger(_open("lab")).anchor()
    console.print(f"anchored: {line.strip()}\ncommit docs/prediction_anchors.md to timestamp it")
