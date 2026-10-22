"""Digitize the measured points of the figures listed in sources.yaml.

    uv run --with pymupdf python experiments/literature_scr/digitize.py [figure-id prefix ...]

Downloads each source PDF to cache/ (git-ignored), extracts the figure image at its native resolution, crops the
panel when the image holds several, finds the plot frame, and picks out marker blobs by color and shape. Model
curves, error bars and text are thin strokes, so a morphological opening removes them and leaves the filled
markers; open markers (rings, square frames) are found by matching their outline instead. Pixel centers map onto
data values either through the axis limits at the frame edges or, for axes whose
range runs past the end ticks, through the major tick marks detected on the axis. Writes data/<figure>.csv and,
for checking by eye, cache/<figure>-check.png with every detected point circled.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pymupdf
from scipy import ndimage, signal

from catalyst_engine.packs import read_yaml

HERE = Path(__file__).resolve().parent
CACHE, DATA = HERE / "cache", HERE / "data"
COLORS = {
    "black": lambda r, g, b: (r < 90) & (g < 90) & (b < 90),
    "red": lambda r, g, b: (r > 150) & (g < 90) & (b < 90),
    "blue": lambda r, g, b: (b > 150) & (r < 90) & (g > 60) & (g < 180),
}


def color_mask(img: np.ndarray, color, tol: float) -> np.ndarray:
    """A named color class, or every pixel within `tol` (RGB distance) of an [r, g, b] marker color."""
    if isinstance(color, str):
        return COLORS[color](img[:, :, 0], img[:, :, 1], img[:, :, 2])
    return np.sqrt(((img - np.array(color)) ** 2).sum(axis=2)) < tol


def fetch(url: str, name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 catalyst-engine literature check"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return path


def figure_image(pdf: Path, page: int, index: int) -> np.ndarray:
    doc = pymupdf.open(pdf)
    xref = doc[page - 1].get_images(full=True)[index][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].astype(int)


def _edges(counts: np.ndarray, need: float) -> tuple[float, float]:
    """Centers of the first and last runs of lines whose ink count passes `need` (axis lines are 1-3 px wide)."""
    idx = np.where(counts > need)[0]
    if len(idx) < 2:
        raise ValueError("plot frame not found")
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return float(runs[0].mean()), float(runs[-1].mean())


def frame(img: np.ndarray, ink_below: float = 245) -> tuple[float, float, float, float]:
    """Pixel columns of the left axis and right frame edge, rows of the top frame edge and bottom axis.
    Light, anti-aliased gray frames need ink_below near white; frames over shaded backgrounds need a darker cut."""
    ink = img.mean(axis=2) < ink_below
    h, w = ink.shape
    left, right = _edges(ink.sum(axis=0), 0.5 * h)
    top, bottom = _edges(ink.sum(axis=1), 0.5 * w)
    return left, right, top, bottom


def tick_centers(img: np.ndarray, box: tuple[float, float, float, float], axis: str, n: int,
                 side: str | None = None) -> list[float]:
    """The n major tick marks on one frame edge: the bottom (x) or left (y) axis, or the top or right edge when
    markers crowd the axis. A tick is a dark run that starts at the edge line and points out of the frame or into
    it; tick labels sit apart from the line and drop out. Majors are the n longest marks, so minor ticks drop out
    too (a curve touching a tick only makes it longer)."""
    dark = img.mean(axis=2) < 128
    left, right, top, bottom = (int(round(v)) for v in box)
    side = side or ("bottom" if axis == "x" else "left")
    line, out = {"bottom": (bottom, 1), "top": (top, -1), "left": (left, -1), "right": (right, 1)}[side]
    best: list[float] = []
    for step in (out, -out):  # outward, then inward
        steps = np.arange(line + 2 * step, line + 16 * step, step)
        if axis == "x":
            seg, start = dark[steps][:, left + 4: right - 3], left + 4
        else:
            seg, start = dark[top + 4: bottom - 3][:, steps].T, top + 4
        length = np.cumprod(seg, axis=0).sum(axis=0)  # dark pixels in a row, counted from the edge line
        idx = np.where(length >= 4)[0]
        runs = [r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if 0 < len(r) <= 6]  # wide = a marker
        if not runs:
            continue
        longest = sorted(runs, key=lambda r: length[r].max(), reverse=True)[:n]
        found = sorted(start + float(r.mean()) for r in longest)
        best = found if len(found) > len(best) else best
    if not best:
        raise ValueError(f"no {axis} ticks found")
    return best


def calibrate(spec: dict, img: np.ndarray, box: tuple[float, float, float, float], axis: str):
    """Pixel -> value along one axis: from the frame edges (min/max) or a straight-line fit to the major ticks."""
    left, right, top, bottom = box
    if "ticks" not in spec:
        lo_px, hi_px = (left, right) if axis == "x" else (bottom, top)
        return lambda px: spec["min"] + (px - lo_px) / (hi_px - lo_px) * (spec["max"] - spec["min"]), 0.0
    values = spec["ticks"] if axis == "x" else spec["ticks"][::-1]  # y ticks run top to bottom in pixels
    found = tick_centers(img, box, axis, len(values), spec.get("tick_side"))
    if len(found) != len(values):
        raise ValueError(f"{axis} axis: found {len(found)} major ticks at {[round(f) for f in found]}, "
                         f"expected {len(values)}")
    slope, icept = np.polyfit(found, values, 1)
    resid_px = float(np.max(np.abs(np.polyval([slope, icept], found) - values)) / abs(slope))
    if resid_px > 2.0:
        raise ValueError(f"{axis} axis: ticks at {[round(f) for f in found]} sit off an even spacing by {resid_px:.1f} px")
    return lambda px: slope * px + icept, resid_px


def _shape(blob: np.ndarray) -> str:
    """Circle or upward triangle, from the row widths of the marker with thin lines already removed: a
    triangle's widest row is its base at the bottom; a circle narrows again below its middle."""
    widths = blob.sum(axis=1)
    widths = widths[widths > 0]
    return "triangle" if widths[-1] >= 0.8 * widths.max() else "circle"


def markers(img: np.ndarray, color, shape: str | None, box: tuple[float, float, float, float], *, tol: float = 60,
            opening: int = 4, size: tuple[int, int] = (4, 24), exclude: list | None = None) -> list[tuple[float, float]]:
    """Centres of filled markers of one color (and, when given, one shape) inside the plot frame, leaving out any
    inside the `exclude` boxes (legend entries)."""
    mask = color_mask(img, color, tol)
    opened = ndimage.binary_opening(mask, structure=np.ones((opening, opening)))
    labels, _ = ndimage.label(opened)
    left, right, top, bottom = box
    found = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        hgt, wid = ys.stop - ys.start, xs.stop - xs.start
        if not (size[0] <= hgt <= size[1] and size[0] <= wid <= size[1] and 0.5 <= hgt / wid <= 2.0):
            continue
        if any(x0 <= (xs.start + xs.stop) / 2 <= x1 and y0 <= (ys.start + ys.stop) / 2 <= y1
               for x0, y0, x1, y1 in exclude or []):
            continue
        if shape is not None and _shape(labels[sl] == k) != shape:
            continue
        cx, cy = (xs.start + xs.stop - 1) / 2, (ys.start + ys.stop - 1) / 2
        if left - 8 <= cx <= right + 8 and top - 8 <= cy <= bottom + 8:
            found.append((cx, cy))
    return sorted(found)


def outline(shape: str, diameter: int, width: int) -> np.ndarray:
    """An open-marker template: a ring (circle) or a square frame, `width` px thick, on an odd-sized grid."""
    n = diameter | 1
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    d = np.hypot(xx, yy) if shape == "circle" else np.maximum(np.abs(xx), np.abs(yy))
    return (d <= diameter / 2 - 0.5) & (d >= diameter / 2 - width - 0.5)


def open_markers(img: np.ndarray, color, shape: str, box: tuple[float, float, float, float], *, diameter: int,
                 width: int = 3, tol: float = 60, score: float = 0.8, exclude: list | None = None) -> list[tuple[float, float]]:
    """Centres of open markers (rings, square frames) of one color: the spots where an outline of the marker's size
    lies almost wholly on that color. A thin curve crosses an outline at two places only, so curves drop out;
    markers closer together than half their size read as one."""
    tpl = outline(shape, diameter, width).astype(float)
    mask = color_mask(img, color, tol).astype(float)
    cover = signal.fftconvolve(mask, tpl[::-1, ::-1], mode="same") / tpl.sum()
    peaks = (cover >= score) & (cover >= ndimage.maximum_filter(cover, size=diameter // 2) - 1e-9)
    labels, n = ndimage.label(peaks)
    left, right, top, bottom = box
    found = []
    for cy, cx in ndimage.center_of_mass(peaks, labels, range(1, n + 1)):
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in exclude or []):
            continue
        if left - 8 <= cx <= right + 8 and top - 8 <= cy <= bottom + 8:
            found.append((float(cx), float(cy)))
    return sorted(found)


def check_image(img: np.ndarray, points: list[tuple[float, float]], path: Path) -> None:
    out = img.copy().astype(np.uint8)
    for cx, cy in points:
        yy, xx = np.ogrid[: out.shape[0], : out.shape[1]]
        ring = np.abs(np.hypot(xx - cx, yy - cy) - 11) < 1.2
        out[ring] = (0, 200, 0)
    pix = pymupdf.Pixmap(pymupdf.csRGB, out.shape[1], out.shape[0], out.tobytes(), False)
    pix.save(path)


def main() -> None:
    spec = read_yaml(HERE / "sources.yaml")  # keeps the name NO a string
    DATA.mkdir(exist_ok=True)
    only = sys.argv[1:]
    for fig in spec["figures"]:
        if only and not fig["id"].startswith(tuple(only)):
            continue
        src = spec["sources"][fig["source"]]
        name = fig.get("file", f"{fig['source']}.pdf")
        # Sources behind a browser check list no download link; their files are saved into cache/ by hand.
        pdf = fetch(src["pdf"], name) if src.get("pdf") and "file" not in fig else CACHE / name
        img = figure_image(pdf, fig["page"], fig["image"])
        x0, y0 = 0, 0
        if "crop" in fig:  # several panels in one image: work on this panel only, in full-image coordinates
            x0, y0, x1, y1 = fig["crop"]
            full, img = img, img[y0:y1, x0:x1]
        box = left, right, top, bottom = frame(img, fig.get("frame_ink_below", 245))
        to_x, res_x = calibrate(fig["x"], img, box, "x")
        to_y, res_y = calibrate(fig["y"], img, box, "y")
        exclude = [(a - x0, b - y0, c - x0, d - y0) for a, b, c, d in fig.get("exclude", [])]
        kw = {k: fig[k] for k in ("tol", "opening") if k in fig} | ({"size": tuple(fig["size"])} if "size" in fig else {})
        to_c = (lambda v: v - 273.15) if fig["x"].get("unit") == "K" else (lambda v: v)
        rows, every = [], []
        shared = [c for c in (str(x["color"]) for x in fig["series"])]
        for s in fig["series"]:
            if s.get("open"):  # hollow markers, found by their outline
                pts = open_markers(img, s["color"], s["shape"], box, diameter=s["open"]["diameter"],
                                   width=s["open"].get("width", 3), score=s["open"].get("score", 0.8), exclude=exclude,
                                   **{k: fig[k] for k in ("tol",) if k in fig})
            else:
                shape = s["shape"] if shared.count(str(s["color"])) > 1 else None  # shape only matters within a color
                pts = markers(img, s["color"], shape, box, exclude=exclude, **kw)
            every += pts
            for cx, cy in pts:
                rows.append({"figure": fig["id"], "series": s["name"], "catalyst": s.get("catalyst", fig.get("catalyst")),
                             "quantity": s["quantity"], "T_C": round(float(to_c(to_x(cx))), 1),
                             "value": round(float(to_y(cy)), 4), "x_px": round(cx + x0, 1), "y_px": round(cy + y0, 1)})
        with (DATA / f"{fig['id']}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        check_image(img, every, CACHE / f"{fig['id']}-check.png")
        counts = {s: sum(r["series"] == s for r in rows) for s in dict.fromkeys(r["series"] for r in rows)}
        print(f"{fig['id']}: frame x {left:.1f}-{right:.1f}, y {top:.1f}-{bottom:.1f}; tick fit within "
              f"{max(res_x, res_y):.2f} px; points {counts}")


if __name__ == "__main__":
    main()
