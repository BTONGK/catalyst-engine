"""Digitize the measured points of the figures listed in sources.yaml.

    uv run --with pymupdf python experiments/literature_scr/digitize.py

Downloads each source PDF to cache/ (git-ignored), extracts the figure image at its native resolution, finds the
plot frame, and picks out marker blobs by color and shape. Model curves and text are thin strokes, so a
morphological opening removes them and leaves the filled markers. Pixel centers map linearly onto the axis
limits given in sources.yaml. Writes data/<figure>.csv and, for checking by eye, cache/<figure>-check.png with
every detected point circled.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np
import pymupdf
import yaml
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CACHE, DATA = HERE / "cache", HERE / "data"
COLORS = {
    "black": lambda r, g, b: (r < 90) & (g < 90) & (b < 90),
    "red": lambda r, g, b: (r > 150) & (g < 90) & (b < 90),
    "blue": lambda r, g, b: (b > 150) & (r < 90) & (g > 60) & (g < 180),
}


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


def frame(img: np.ndarray) -> tuple[float, float, float, float]:
    """Pixel columns of the left axis and right frame edge, rows of the top frame edge and bottom axis.
    Frame lines are light, anti-aliased gray, so anything darker than near-white counts as ink."""
    ink = img.mean(axis=2) < 245
    h, w = ink.shape
    left, right = _edges(ink.sum(axis=0), 0.5 * h)
    top, bottom = _edges(ink.sum(axis=1), 0.5 * w)
    return left, right, top, bottom


def _shape(blob: np.ndarray) -> str:
    """Circle or upward triangle, from the row widths of the marker with thin lines already removed: a
    triangle's widest row is its base at the bottom; a circle narrows again below its middle."""
    widths = blob.sum(axis=1)
    widths = widths[widths > 0]
    return "triangle" if widths[-1] >= 0.8 * widths.max() else "circle"


def markers(img: np.ndarray, color: str, shape: str | None, box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Centres of filled markers of one color (and, when given, one shape) inside the plot frame."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = COLORS[color](r, g, b)
    opened = ndimage.binary_opening(mask, structure=np.ones((4, 4)))
    labels, _ = ndimage.label(opened)
    left, right, top, bottom = box
    found = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        hgt, wid = ys.stop - ys.start, xs.stop - xs.start
        if not (4 <= hgt <= 24 and 4 <= wid <= 24 and 0.5 <= hgt / wid <= 2.0):
            continue
        if shape is not None and _shape(labels[sl] == k) != shape:
            continue
        cx, cy = (xs.start + xs.stop - 1) / 2, (ys.start + ys.stop - 1) / 2
        if left - 8 <= cx <= right + 8 and top - 8 <= cy <= bottom + 8:
            found.append((cx, cy))
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
    spec = yaml.safe_load((HERE / "sources.yaml").read_text())
    DATA.mkdir(exist_ok=True)
    for fig in spec["figures"]:
        src = spec["sources"][fig["source"]]
        img = figure_image(fetch(src["pdf"], f"{fig['source']}.pdf"), fig["page"], fig["image"])
        box = left, right, top, bottom = frame(img)
        x, y = fig["x"], fig["y"]
        rows, every = [], []
        shared = [c for c in (x["color"] for x in fig["series"])]
        for s in fig["series"]:
            shape = s["shape"] if shared.count(s["color"]) > 1 else None  # shape only matters within a color
            pts = markers(img, s["color"], shape, box)
            every += pts
            for cx, cy in pts:
                xv = x["min"] + (cx - left) / (right - left) * (x["max"] - x["min"])
                yv = y["max"] - (cy - top) / (bottom - top) * (y["max"] - y["min"])
                rows.append({"figure": fig["id"], "series": s["name"], "catalyst": s.get("catalyst", fig.get("catalyst")),
                             "quantity": s["quantity"], "T_C": round(xv, 1), "value": round(yv, 4),
                             "x_px": cx, "y_px": cy})
        with (DATA / f"{fig['id']}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        check_image(img, every, CACHE / f"{fig['id']}-check.png")
        counts = {s: sum(r["series"] == s for r in rows) for s in dict.fromkeys(r["series"] for r in rows)}
        print(f"{fig['id']}: frame x {left}-{right}, y {top}-{bottom}; points {counts}")


if __name__ == "__main__":
    main()
