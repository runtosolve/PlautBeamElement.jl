"""
Digitize the Abaqus S4R shell results (blue markers) from the figures of
"Z8 Purlin Three-Way Validation" (Elikem, rev. 29 Oct 2026). The raw Abaqus output
(.odb) is not available, so the marker centers are read from the report's embedded
plot images. The load of each incremental-run marker is then snapped to the exact
load of its converged increment, read from the run's status file
(nlgeom_full_run/validate_nlgeom_full.sta; reference load 120 lb/ft); every marker
lies within 0.06 lb/ft of one. Only the u, v, phi values are digitized. Writes
abaqus_digitized.csv next to this script:

    quantity, q_plf, value          (Abaqus frame: v and phi negative under gravity)

quantity = u (in), v (in), phi_lin (rad, linearised -dU1/h, incremental run),
phi_atan (rad, true web inclination), phi_fixed (rad, the fixed 4.8 lb/ft run).
Checked against the report's tables: u within 0.013 in, v within 0.025 in, and
phi_atan within 0.012 rad at every tabulated load (2.4-23.8 lb/ft).

Run:  python digitize_abaqus.py
"""
import csv
import os
import re

import fitz                     # PyMuPDF
import numpy as np
from scipy import ndimage

PDF = (r"G:\Shared drives\RunToSolve\Projects\PurlinLine\Validation study_2026"
       r"\Abaqus Validation Study_2026\2026-10-09-model1"
       r"\Z8_Purlin_Three-Way_Validation_Report_Elikem_3.pdf")
STA = (r"G:\Shared drives\RunToSolve\Papers\SSRC_2027\Abaqus\Z8x2.5_16g_single_span"
       r"\nlgeom_full_run\validate_nlgeom_full.sta")
Q_REF = 120.0                   # lb/ft at load proportionality factor 1
HERE = os.path.dirname(os.path.abspath(__file__))


def increment_loads(path):
    """Loads (lb/ft) of the converged increments in an Abaqus/Standard .sta file."""
    loads = []
    for line in open(path):
        m = re.match(r"\s+1\s+\d+\s+\d+(U?)\s+\d+\s+\d+\s+\d+\s+([\d.Ee+-]+)", line)
        if m and not m.group(1):
            loads.append(float(m.group(2)) * Q_REF)
    return np.array(loads)

# page (0-based), quantity, x range of the frame, y gridline values from top to bottom
PLOTS = [
    (5, "phi_small", (0.0, 6.0), [0.0, -0.05, -0.10, -0.15, -0.20]),
    (6, "u",   (0.0, 25.0), [4.0, 3.0, 2.0, 1.0, 0.0]),
    (7, "v",   (0.0, 25.0), [0.0, -2.0, -4.0, -6.0, -8.0, -10.0]),
    (4, "phi", (0.0, 25.0), [0.0, -1.0, -2.0, -3.0]),
]


def page_image(doc, page):
    img = doc[page].get_images(full=True)[0]
    pix = fitz.Pixmap(doc, img[0])
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[:, :, :3].astype(int)


def groups(idx):
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            out.append(cur); cur = [i]
    out.append(cur)
    return out


def frame(a):
    """Pixel bounds of the plot frame: the first two long dark horizontal lines
    (the legend box below the plot is also long and dark) and the dark verticals."""
    dark = a.sum(axis=2) < 200
    hg = groups(np.where(dark.sum(axis=1) > 0.6 * a.shape[1])[0])
    top, bottom = int(np.mean(hg[0])), int(np.mean(hg[1]))
    vg = groups(np.where(dark[top:bottom].sum(axis=0) > 0.8 * (bottom - top))[0])
    left, right = int(np.mean(vg[0])), int(np.mean(vg[-1]))
    return left, right, top, bottom


def gridline_rows(a, left, right, top, bottom):
    """Rows of the horizontal gridlines (light, low-saturation), looked for in the right
    third of the frame, clear of the shaded small-deformation band on the left."""
    sub = a[top:bottom + 1, left + 2 * (right - left) // 3:right - 1]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    grayish = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r < 236) & (r > 120)
    frac = grayish.mean(axis=1)
    cand = np.where(frac > 0.45)[0] + top
    groups, cur = [], [cand[0]]
    for y in cand[1:]:
        if y - cur[-1] <= 2:
            cur.append(y)
        else:
            groups.append(cur); cur = [y]
    groups.append(cur)
    return [np.mean(gp) for gp in groups]


def digitize(a, xr, ygrid):
    left, right, top, bottom = frame(a)
    rows = gridline_rows(a, left, right, top, bottom)
    # every gridline value has a row; the outermost may coincide with the frame
    rows = [r for r in rows if top + 1 < r < bottom - 1]
    inner = [v for v in ygrid if True]
    if len(rows) < len(inner):
        # an end tick sits on the frame edge, where no separate gridline is drawn:
        # add the edge that lies one gridline spacing beyond the detected lines
        sp = (rows[-1] - rows[0]) / (len(rows) - 1)
        if abs(rows[0] - sp - top) < 3:
            rows = [top] + rows
        if abs(rows[-1] + sp - bottom) < 3:
            rows = rows + [bottom]
    assert len(rows) == len(inner), (rows, inner)
    ky, cy = np.polyfit(rows, inner, 1)
    kx = (xr[1] - xr[0]) / (right - left)

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > 170) & (r < 110) & (g > 80) & (g < 170)
    blue[:top + 1, :] = False; blue[bottom:, :] = False
    blue[:, :left + 1] = False; blue[:, right:] = False
    filled = ndimage.binary_fill_holes(blue)
    core = ndimage.binary_erosion(filled, iterations=2)
    lab, n = ndimage.label(core)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = len(xs)
        if area < 12:
            continue
        q = xr[0] + kx * (xs.mean() - left)
        val = ky * ys.mean() + cy
        # an upward triangle has its centroid below the middle of its bounding box
        tri = ys.mean() - 0.5 * (ys.min() + ys.max()) > 0.6
        out.append((q, val, area, None, tri))
    # The twist plot overlays filled markers (linearised twist, small blobs) and larger
    # hollow circles (true inclination). Blobs much larger than a single hollow circle
    # are overlapping circles whose centroid is not a data point, so they are dropped.
    areas = sorted(p[2] for p in out)
    if areas and areas[-1] > 2.5 * areas[0]:
        small = areas[0]
        single = np.median([ar for ar in areas if ar > 2.5 * small])
        out = [(q, v, ar, ar > 2.5 * small, tri) for q, v, ar, _, tri in out if ar < 1.6 * single]
    return sorted(out)


def main():
    doc = fitz.open(PDF)
    inc = increment_loads(STA)

    def snap(q):
        i = np.argmin(np.abs(inc - q))
        assert abs(inc[i] - q) < 0.15, f"marker at {q:.3f} lb/ft matches no increment"
        return inc[i]

    rows = []
    for page, name, xr, ygrid in PLOTS:
        pts = digitize(page_image(doc, page), xr, ygrid)
        for q, val, area, hollow, tri in pts:
            if name == "phi":
                rows.append(("phi_atan" if hollow else "phi_lin", snap(q), val, "fig. 0-25 lb/ft"))
            elif name == "phi_small":
                # the fixed 4.8 lb/ft run has no status file here; its loads stay digitized
                rows.append(("phi_fixed", q, val, "fig. 0-6 lb/ft") if tri else
                            ("phi_lin", snap(q), val, "fig. 0-6 lb/ft"))
            else:
                rows.append((name, snap(q), val, "fig. 0-25 lb/ft"))
        print(name, len(pts), "markers")
    with open(os.path.join(HERE, "abaqus_digitized.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "q_plf", "value", "source"])
        for qty, q, val, src in rows:
            w.writerow([qty, f"{q:.4f}", f"{val:.5f}", src])


if __name__ == "__main__":
    main()
