"""Minimal, dependency-free SVG chart writer for the GOMEA benchmark harness.

Two chart types: `line_chart` (step lines -- e.g. empirical CDFs of
evaluations-to-target) and `box_chart` (per-group box-and-whisker of a
numeric metric). Both write self-contained SVG (no external stylesheets,
fonts or scripts) sized 720x420 with an explicit `viewBox`, white
background, 12px sans-serif text, and a fixed colour-blind-safe (Okabe-Ito)
palette.
"""

from __future__ import annotations

import math
from pathlib import Path

__all__ = ["line_chart", "box_chart"]

_WIDTH = 720
_HEIGHT = 420
_MARGIN_L = 70
_MARGIN_R = 170  # room for the legend
_MARGIN_T = 40
_MARGIN_B = 55

# Okabe-Ito, colour-blind-safe, black held back as the last resort so early
# series don't default to it.
_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
    "#CC79A7",
    "#F0E442",
    "#000000",
]


def _nice_step(span: float, target: int = 5) -> float:
    if span <= 0:
        return 1.0
    raw_step = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 5, 10):
        step = m * mag
        if step >= raw_step:
            return step
    return 10 * mag


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    step = _nice_step(hi - lo, target)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:g}"


def _svg_header(title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
        f'viewBox="0 0 {_WIDTH} {_HEIGHT}" font-family="Helvetica, Arial, sans-serif">\n'
        f'<rect x="0" y="0" width="{_WIDTH}" height="{_HEIGHT}" fill="#ffffff"/>\n'
        f'<text x="{_WIDTH / 2}" y="20" font-size="14" text-anchor="middle" '
        f'fill="#111111">{_esc(title)}</text>\n'
    )


def _axes(plot_w: float, plot_h: float) -> str:
    return (
        f'<line x1="{_MARGIN_L}" y1="{_MARGIN_T + plot_h}" x2="{_MARGIN_L + plot_w}" '
        f'y2="{_MARGIN_T + plot_h}" stroke="#333333" stroke-width="1"/>\n'
        f'<line x1="{_MARGIN_L}" y1="{_MARGIN_T}" x2="{_MARGIN_L}" '
        f'y2="{_MARGIN_T + plot_h}" stroke="#333333" stroke-width="1"/>\n'
    )


def _y_axis_label(plot_h: float, ylabel: str) -> str:
    cy = _MARGIN_T + plot_h / 2
    return (
        f'<text x="16" y="{cy:.2f}" font-size="12" text-anchor="middle" fill="#111111" '
        f'transform="rotate(-90 16 {cy:.2f})">{_esc(ylabel)}</text>\n'
    )


def _x_axis_label(plot_w: float, xlabel: str) -> str:
    cx = _MARGIN_L + plot_w / 2
    return (
        f'<text x="{cx:.2f}" y="{_HEIGHT - 10}" font-size="12" text-anchor="middle" '
        f'fill="#111111">{_esc(xlabel)}</text>\n'
    )


def line_chart(
    series: dict[str, list[tuple[float, float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | Path,
    xlog: bool = False,
) -> None:
    """Write a step-line chart (one path per `series` entry, in dict order)
    to `path`. Every series gets a `<path>` element even when it has no
    points, so the SVG always has exactly `len(series)` line paths.
    """
    plot_w = _WIDTH - _MARGIN_L - _MARGIN_R
    plot_h = _HEIGHT - _MARGIN_T - _MARGIN_B

    all_x = [x for pts in series.values() for x, _ in pts]
    all_y = [y for pts in series.values() for _, y in pts]
    if not all_x:
        all_x = [0.0, 1.0]
    if not all_y:
        all_y = [0.0, 1.0]

    if xlog:
        pos_x = [x for x in all_x if x > 0]
        xmin = min(pos_x) if pos_x else 1.0
        xmax = max(pos_x) if pos_x else 10.0
        if xmin == xmax:
            xmin, xmax = xmin / 2.0, xmax * 2.0
        log_lo, log_hi = math.log10(xmin), math.log10(xmax)
        log_span = (log_hi - log_lo) or 1.0

        def sx(x: float) -> float:
            xx = max(x, xmin)
            return _MARGIN_L + (math.log10(xx) - log_lo) / log_span * plot_w
    else:
        xmin, xmax = min(all_x), max(all_x)
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        x_span = xmax - xmin

        def sx(x: float) -> float:
            return _MARGIN_L + (x - xmin) / x_span * plot_w

    ymin, ymax = min(0.0, min(all_y)), max(all_y)
    if ymin == ymax:
        ymax = ymin + 1.0
    y_span = ymax - ymin

    def sy(y: float) -> float:
        return _MARGIN_T + plot_h - (y - ymin) / y_span * plot_h

    parts = [_svg_header(title), _axes(plot_w, plot_h)]

    for t in _nice_ticks(ymin, ymax):
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue
        y = sy(t)
        parts.append(
            f'<line x1="{_MARGIN_L - 4}" y1="{y:.2f}" x2="{_MARGIN_L}" y2="{y:.2f}" '
            f'stroke="#333333" stroke-width="1"/>\n'
            f'<text x="{_MARGIN_L - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" '
            f'fill="#111111">{_fmt(t)}</text>\n'
        )

    if xlog:
        exp_lo = math.floor(math.log10(xmin))
        exp_hi = math.ceil(math.log10(xmax))
        xticks = [10.0**e for e in range(int(exp_lo), int(exp_hi) + 1)]
    else:
        xticks = _nice_ticks(xmin, xmax)
    for t in xticks:
        if t < xmin - 1e-9 or t > xmax + 1e-9:
            continue
        x = sx(t)
        parts.append(
            f'<line x1="{x:.2f}" y1="{_MARGIN_T + plot_h}" x2="{x:.2f}" '
            f'y2="{_MARGIN_T + plot_h + 4}" stroke="#333333" stroke-width="1"/>\n'
            f'<text x="{x:.2f}" y="{_MARGIN_T + plot_h + 18}" font-size="11" '
            f'text-anchor="middle" fill="#111111">{_fmt(t)}</text>\n'
        )

    parts.append(_x_axis_label(plot_w, xlabel))
    parts.append(_y_axis_label(plot_h, ylabel))

    for i, (name, pts) in enumerate(series.items()):
        color = _PALETTE[i % len(_PALETTE)]
        pts_sorted = sorted(pts, key=lambda p: p[0])
        if pts_sorted:
            d_parts = []
            prev_y = None
            for j, (x, y) in enumerate(pts_sorted):
                xx, yy = sx(x), sy(y)
                if j == 0:
                    d_parts.append(f"M {xx:.2f} {yy:.2f}")
                else:
                    d_parts.append(f"L {xx:.2f} {prev_y:.2f} L {xx:.2f} {yy:.2f}")
                prev_y = yy
            d = " ".join(d_parts)
        else:
            # Still emit a (degenerate, invisible) path so every series
            # contributes exactly one <path> element.
            d = f"M {_MARGIN_L:.2f} {sy(ymin):.2f}"
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>\n')

        ly = _MARGIN_T + i * 18 + 10
        lx = _MARGIN_L + plot_w + 15
        parts.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" '
            f'stroke-width="3"/>\n'
            f'<text x="{lx + 24}" y="{ly + 4}" font-size="11" fill="#111111">'
            f'{_esc(name)}</text>\n'
        )

    parts.append("</svg>\n")
    Path(path).write_text("".join(parts))


def box_chart(
    groups: dict[str, list[float]],
    title: str,
    ylabel: str,
    path: str | Path,
) -> None:
    """Write a box-and-whisker chart (one box per `groups` entry, in dict
    order; min/max whiskers, IQR box, median line) to `path`."""
    # Local import: keeps `stats.py` free of any dependency on this module.
    from benchmarks.gomea.stats import median_iqr

    names = list(groups.keys())
    plot_w = _WIDTH - _MARGIN_L - _MARGIN_R
    plot_h = _HEIGHT - _MARGIN_T - _MARGIN_B

    all_vals = [v for vs in groups.values() for v in vs]
    if not all_vals:
        all_vals = [0.0, 1.0]
    ymin, ymax = min(all_vals), max(all_vals)
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    y_span = ymax - ymin

    def sy(y: float) -> float:
        return _MARGIN_T + plot_h - (y - ymin) / y_span * plot_h

    n = max(len(names), 1)
    slot = plot_w / n
    box_w = slot * 0.5

    parts = [_svg_header(title), _axes(plot_w, plot_h)]
    for t in _nice_ticks(ymin, ymax):
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue
        y = sy(t)
        parts.append(
            f'<line x1="{_MARGIN_L - 4}" y1="{y:.2f}" x2="{_MARGIN_L}" y2="{y:.2f}" '
            f'stroke="#333333" stroke-width="1"/>\n'
            f'<text x="{_MARGIN_L - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" '
            f'fill="#111111">{_fmt(t)}</text>\n'
        )
    parts.append(_y_axis_label(plot_h, ylabel))

    for i, name in enumerate(names):
        vals = groups[name]
        color = _PALETTE[i % len(_PALETTE)]
        cx = _MARGIN_L + slot * i + slot / 2
        if vals:
            med, q1, q3 = median_iqr(vals)
            lo, hi = min(vals), max(vals)
            y_med, y_q1, y_q3 = sy(med), sy(q1), sy(q3)
            y_lo, y_hi = sy(lo), sy(hi)
            x0, x1 = cx - box_w / 2, cx + box_w / 2
            box_top, box_bottom = min(y_q1, y_q3), max(y_q1, y_q3)
            parts.append(
                f'<line x1="{cx:.2f}" y1="{y_hi:.2f}" x2="{cx:.2f}" y2="{box_top:.2f}" '
                f'stroke="{color}" stroke-width="1.5"/>\n'
                f'<line x1="{cx:.2f}" y1="{box_bottom:.2f}" x2="{cx:.2f}" y2="{y_lo:.2f}" '
                f'stroke="{color}" stroke-width="1.5"/>\n'
                f'<rect x="{x0:.2f}" y="{box_top:.2f}" width="{box_w:.2f}" '
                f'height="{max(box_bottom - box_top, 0.5):.2f}" fill="{color}" '
                f'fill-opacity="0.35" stroke="{color}" stroke-width="1.5"/>\n'
                f'<line x1="{x0:.2f}" y1="{y_med:.2f}" x2="{x1:.2f}" y2="{y_med:.2f}" '
                f'stroke="{color}" stroke-width="2"/>\n'
            )
        parts.append(
            f'<text x="{cx:.2f}" y="{_MARGIN_T + plot_h + 18}" font-size="11" '
            f'text-anchor="middle" fill="#111111">{_esc(name)}</text>\n'
        )

    parts.append("</svg>\n")
    Path(path).write_text("".join(parts))
