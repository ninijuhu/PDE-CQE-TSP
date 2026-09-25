"""
Fig. 2 — Mechanism isolation of PDE and CQE (two panels).

(a) Component-level degradation relative to Full PDE–CQE
    (blue: PDE-only; grey: PDE replacement (2-opt)).
(b) PDE-only vs matched CDE: paired connected dots per instance
    (blue filled: PDE-only; grey hollow: CDE; Full PDE–CQE = 0%).
    Off-scale CDE values use an upward arrow + value annotation (no clipped marker).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

YLABEL = "Relative path-length increase vs Full PDE–CQE (%)"
XTICK_KW = dict(rotation=45, ha="right", fontsize=8.5)

# (a) bar colors: blue = PDE-only, grey = PDE replacement (2-opt)
C_BAR_PDE = "#4C78A8"
C_BAR_REPLACE = "#A0A0A0"

# (b) markers: blue filled = PDE-only, grey hollow = matched CDE
C_PDE = "#4C78A8"
C_CDE = "#7F7F7F"
C_LINK = "#B8B8B8"
C_FULL = "#333333"
C_GRID = "#C8C8C8"
C_SPINE = "#333333"

OUT_DIR = Path(r"D:\postgraduate\路径规划\路径规划")
EXCEL_PATH = OUT_DIR / "总的消融.xlsx"

PAPER_DATA = pd.DataFrame(
    [
        ("Ulysses-16", 78.57, 78.84, 78.63, 79.14),
        ("Ulysses-22", 79.05, 80.17, 79.84, 79.67),
        ("Eil-51", 504.03, 506.52, 507.46, 508.03),
        ("Eil-76", 620.02, 638.03, 637.53, 634.11),
        ("Berlin-52", 8623.66, 8957.31, 8937.16, 8767.95),
        ("St-70", 764.05, 772.86, 776.70, 772.63),
        ("Random-30", 10444.97, 10647.37, 10754.08, 10589.23),
        ("Random-50", 14252.80, 15102.10, 15308.47, 17924.45),
        ("Random-70", 15808.77, 16021.68, 15868.02, 15959.96),
        ("Random-100", 18593.20, 19163.62, 19347.01, 18909.28),
    ],
    columns=["dataset", "Full", "PDE", "w/o PDE", "CDE"],
)


def load_data() -> pd.DataFrame:
    df = PAPER_DATA.copy()
    if not EXCEL_PATH.is_file():
        print(f"[info] Excel not found ({EXCEL_PATH.name}); using paper Tables 2–3.")
        return df

    try:
        raw = pd.read_excel(EXCEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to read Excel ({exc}); using paper Tables 2–3.")
        return df

    colmap = {c.lower().strip(): c for c in raw.columns}

    def pick(*names):
        for name in names:
            if name.lower() in colmap:
                return raw[colmap[name.lower()]]
        return None

    ds = pick("dataset", "Dataset")
    full = pick("Full", "Full Model", "full_model", "Full PDE-CQE", "PDE_CQE", "Grover_QGA")
    pde = pick("PDE", "PDE-only", "w/o CQE", "no_cqe", "no_Grover")
    no_pde = pick("w/o PDE", "no_pde", "PDE replacement (2-opt)", "no_QGA")
    cde = pick("CDE", "cde", "Classical")

    if ds is None or full is None or pde is None or no_pde is None:
        print("[warn] Excel missing expected columns; using paper Tables 2–3.")
        return df

    out = pd.DataFrame(
        {
            "dataset": ds.astype(str).tolist(),
            "Full": np.asarray(full, dtype=float),
            "PDE": np.asarray(pde, dtype=float),
            "w/o PDE": np.asarray(no_pde, dtype=float),
            "CDE": (
                np.asarray(cde, dtype=float)
                if cde is not None
                else np.full(len(ds), np.nan)
            ),
        }
    )

    if cde is None or np.isnan(out["CDE"]).any():
        paper_by_key = {
            "".join(ch for ch in s.lower() if ch.isalnum()): v
            for s, v in zip(PAPER_DATA["dataset"], PAPER_DATA["CDE"])
        }
        filled = []
        for name, val in zip(out["dataset"], out["CDE"]):
            if np.isfinite(val):
                filled.append(val)
            else:
                key = "".join(ch for ch in str(name).lower() if ch.isalnum())
                filled.append(paper_by_key.get(key, np.nan))
        out["CDE"] = filled
        print("[info] CDE filled from paper Table 3 where missing in Excel.")

    if np.isnan(out["CDE"]).any():
        print("[warn] CDE incomplete after merge; using paper Tables 2–3.")
        return df

    print(f"[info] Loaded from {EXCEL_PATH.name}; CDE merged.")
    return out


def relative_increase(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (values - ref) / ref * 100.0


def style_axes(ax, *, remove_top_right: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color=C_GRID, alpha=0.7)
    ax.xaxis.grid(False)
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_color(C_SPINE)
            spine.set_linewidth(0.9)
    ax.tick_params(colors=C_SPINE, length=3.5)
    ax.axhline(0.0, color="black", linewidth=0.85, zorder=2)


def main() -> None:
    df = load_data()
    datasets = df["dataset"].tolist()
    full = df["Full"].to_numpy(dtype=float)
    pde = df["PDE"].to_numpy(dtype=float)
    no_pde = df["w/o PDE"].to_numpy(dtype=float)
    cde = df["CDE"].to_numpy(dtype=float)

    deg_wo_cqe = relative_increase(pde, full)
    deg_wo_pde = relative_increase(no_pde, full)
    deg_pde = relative_increase(pde, full)
    deg_cde = relative_increase(cde, full)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), constrained_layout=True
    )
    x = np.arange(len(datasets))
    width = 0.36

    # ---- (a) Component-level degradation (grouped bars) ----
    ax1.bar(
        x - width / 2,
        deg_wo_cqe,
        width,
        label="PDE-only",
        color=C_BAR_PDE,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax1.bar(
        x + width / 2,
        deg_wo_pde,
        width,
        label="PDE replacement (2-opt)",
        color=C_BAR_REPLACE,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    style_axes(ax1)
    ax1.set_title(
        "(a) Component-level degradation relative to Full PDE–CQE",
        loc="left",
        pad=8,
    )
    ax1.set_ylabel(YLABEL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, **XTICK_KW)
    ymin_a = min(0.0, float(np.min([deg_wo_cqe.min(), deg_wo_pde.min()])) - 0.4)
    ymax_a = float(np.max([deg_wo_cqe.max(), deg_wo_pde.max()])) * 1.15
    ax1.set_ylim(ymin_a, ymax_a)
    ax1.yaxis.set_major_locator(MultipleLocator(1.0))
    ax1.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.95,
    )

    # ---- (b) Paired connected dots; off-scale CDE uses arrow (not clipped marker) ----
    sorted_cde = np.sort(deg_cde)
    typical_max = float(
        np.max(
            [
                deg_pde.max(),
                sorted_cde[-2] if len(sorted_cde) > 1 else sorted_cde[-1],
            ]
        )
    )
    display_max = 8.0
    ylim_top = display_max * 1.05

    outlier_idx = [i for i, v in enumerate(deg_cde) if v > display_max]
    in_range = np.ones(len(datasets), dtype=bool)
    in_range[outlier_idx] = False

    for i in range(len(datasets)):
        y_top = deg_cde[i] if in_range[i] else display_max * 0.92
        ax2.plot(
            [x[i], x[i]],
            [y_top, deg_pde[i]],
            color=C_LINK,
            linewidth=1.2,
            zorder=2,
            solid_capstyle="round",
        )

    ax2.scatter(
        x,
        deg_pde,
        s=55,
        marker="o",
        facecolors=C_PDE,
        edgecolors=C_PDE,
        linewidths=1.2,
        zorder=4,
        label="PDE-only",
    )
    ax2.scatter(
        x[in_range],
        deg_cde[in_range],
        s=55,
        marker="s",
        facecolors="white",
        edgecolors=C_CDE,
        linewidths=1.8,
        zorder=4,
        label="CDE (matched)",
    )

    ax2.axhline(
        0.0,
        color=C_FULL,
        linewidth=1.4,
        linestyle="--",
        zorder=1,
        label="Full PDE–CQE = 0%",
    )

    for i in outlier_idx:
        ax2.annotate(
            "",
            xy=(x[i], display_max),
            xytext=(x[i], display_max - 0.85),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_CDE,
                lw=1.6,
                mutation_scale=12,
            ),
            zorder=5,
        )
        ax2.annotate(
            f"CDE = +{deg_cde[i]:.1f}%",
            xy=(x[i], display_max),
            xytext=(6, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color=C_CDE,
            fontweight="semibold",
            zorder=6,
        )

    style_axes(ax2)
    ax2.set_title("(b) PDE-only vs matched CDE", loc="left", pad=8)
    ax2.set_ylabel(YLABEL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, **XTICK_KW)
    ax2.set_ylim(-0.6, ylim_top)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C_PDE,
            markeredgecolor=C_PDE,
            markeredgewidth=1.2,
            markersize=8,
            label="PDE-only",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_CDE,
            markeredgewidth=1.8,
            markersize=8,
            label="CDE (matched)",
        ),
        Line2D(
            [0],
            [0],
            color=C_FULL,
            linestyle="--",
            linewidth=1.4,
            label="Full PDE–CQE = 0%",
        ),
    ]
    ax2.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.95,
    )

    out_png = OUT_DIR / "fig6_mechanism_isolation.png"
    out_pdf = OUT_DIR / "fig6_mechanism_isolation.pdf"
    legacy_png = OUT_DIR / "ablation_bar_chart1.png"
    legacy_pdf = OUT_DIR / "ablation_bar_chart1.pdf"

    for path in (out_png, legacy_png):
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    for path in (out_pdf, legacy_pdf):
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    for p in (out_png, out_pdf, legacy_png, legacy_pdf):
        print(f"  - {p}")
    if outlier_idx:
        for i in outlier_idx:
            print(
                f"[note] {datasets[i]} CDE off-scale: "
                f"+{deg_cde[i]:.1f}% (arrow + annotation; display max = {display_max:g}%)"
            )
    print(f"[info] In-range CDE/PDE max ≈ {typical_max:.2f}% (display_max = {display_max:g}%)")


if __name__ == "__main__":
    main()
