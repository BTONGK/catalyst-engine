"""Read the per-metal turnover frequencies of Lee's thesis from its bar charts, the way PASS_RULE.md fixes it.

    uv run --no-project --with pymupdf --with numpy --with pillow python experiments/reaction_check/read_bars.py

Needs the thesis in cache/lee2017_thesis.pdf (git-ignored; G. Lee, Monometallic and bimetallic catalysis for C1-C4
alkane and propylene total oxidation, M.Ch.E. thesis, University of Delaware, 2017, from UDSpace). Each figure sits in
the PDF as 150-dpi raster strips; they are stitched and read at their own pixels. For each bar's central column,
below the frame's tick marks, the first row darker than half the background is the bar's top outline; the row above
it, darkened by a share f of the outline's contrast, puts the top f of a pixel higher (anti-aliasing). A bar's
value is its height above the x-axis line as a share of the frame's height, times the axis's top value. Bars whose
tops sit within one pixel of each other share a rank. The printed values are the ones in data/published.yaml.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE / "cache" / "lee2017_thesis.pdf"
SHA256 = "98fdd3ada4e7b32617929f8515136ad8399da4c406a5398b9b89dfd9f9e1678f"

# page (1-based), first rows of the frame's top line and of the x-axis line, the axis's top value, and the central
# column of each metal's bar in the group read (the legends: Ag black, Co red, Cu green, Ni blue, Pd cyan, Pt
# magenta, Rh yellow)
FIGURES = {
    "propane, 340 C (Figure D-1(c), the fourth group)": {
        "page": 105, "frame_top": 372, "axis": 620, "ymax": 4.0,
        "columns": {"Ag": [239], "Co": [243, 244], "Cu": [247, 248], "Ni": [252, 253], "Pd": [257], "Pt": [261],
                    "Rh": [265, 266]}},
    "propylene, 170 C (Figure 4-4, the fifth group)": {
        "page": 78, "frame_top": 77, "axis": 560, "ymax": 1.8,
        "columns": {"Ag": [655], "Co": [666, 668], "Cu": [678, 680], "Ni": [690, 693], "Pd": [702], "Pt": [714],
                    "Rh": [726]}},
}


def figure_image(doc: pymupdf.Document, page: int) -> np.ndarray:
    """The page's raster strips, top to bottom, as one RGB array at their own resolution."""
    strips = []
    for info in sorted(doc[page - 1].get_image_info(xrefs=True), key=lambda i: i["bbox"][1]):
        pix = pymupdf.Pixmap(doc, info["xref"])
        if pix.n - pix.alpha != 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        strips.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3])
    return np.vstack(strips).astype(float)


def bar_top(img: np.ndarray, col: int, frame_top: int, axis: int) -> float:
    s = img[:, col].sum(axis=1)
    background = np.median(s[frame_top + 15: frame_top + 40])
    r = frame_top + 15  # below the tick marks along the frame's top
    while r < axis and s[r] > background / 2:
        r += 1
    f = float(np.clip((background - s[r - 1]) / max(background - s[r], 1.0), 0, 1))
    return r - f


def main() -> None:
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    print(f"{PDF.name}: sha256 {digest} ({'the copy read' if digest == SHA256 else 'a different copy'})")
    doc = pymupdf.open(PDF)
    for title, fig in FIGURES.items():
        img = figure_image(doc, fig["page"])
        per_px = fig["ymax"] / (fig["axis"] - fig["frame_top"])
        heights = {m: float(np.mean([fig["axis"] - bar_top(img, c, fig["frame_top"], fig["axis"]) for c in cols]))
                   for m, cols in fig["columns"].items()}
        print(f"\n{title}: one pixel = {per_px:.4f} 1/s")
        for m, h in sorted(heights.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {h:6.2f} px  {h * per_px:.4f} 1/s  -> {round(h * per_px, 2):.2f}")
        names = sorted(heights, key=lambda m: -heights[m])
        close = [(a, b) for a, b in zip(names, names[1:]) if heights[a] - heights[b] < 1.0]
        print("  within one pixel (one shared rank):", close or "none")


if __name__ == "__main__":
    main()
