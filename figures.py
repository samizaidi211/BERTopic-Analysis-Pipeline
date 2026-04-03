"""
figures.py

Generate publication-grade and Nature-style figures from BERTopic exports.

Expected inputs in outputs/topics/:
    - topic_summary.csv
    - topic_<N>_docs.csv
    - document_metadata_enriched.csv

Expected columns in document_metadata_enriched.csv:
    - source_pdf OR title
    - publication_year
    - citation_count
    - optional: doi, journal, matched_title, matched_year, match_score

Outputs in outputs/figures/:
    - figure_panel_main.png
    - top_10_topics_all_time.png
    - hottest_topic_by_5year_period.png
    - average_citations_per_topic.png
    - fastest_growing_topics.png
    - latent_opportunity_topics.png
    - impact_vs_growth.png
    - topic_growth_table.csv
    - topic_opportunity_table.csv
    - hottest_topics_by_period_table.csv
    - average_citations_table.csv

Usage:
    python figures.py
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOPICS_DIR = Path("outputs/topics")
FIGURES_DIR = Path("outputs/figures")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def make_topic_label(top_words: str, max_words: int = 3) -> str:
    if pd.isna(top_words) or not str(top_words).strip():
        return "Unlabeled Topic"
    words = [w.strip() for w in str(top_words).split(",") if w.strip()]
    return ", ".join(words[:max_words]) if words else "Unlabeled Topic"


def normalize_title(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def load_topic_summary() -> pd.DataFrame:
    path = TOPICS_DIR / "topic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    df = df[df["Topic"] != -1].copy()

    required = {"Topic", "Count", "top_words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"topic_summary.csv missing columns: {missing}")

    if "topic_label" not in df.columns:
        df["topic_label"] = df["top_words"].apply(make_topic_label)

    return df


def load_topic_documents() -> pd.DataFrame:
    rows = []

    for path in TOPICS_DIR.glob("topic_*_docs.csv"):
        if path.name == "topic_outliers_docs.csv":
            continue

        match = re.match(r"topic_(\-?\d+)_docs\.csv", path.name)
        if not match:
            continue

        topic_id = int(match.group(1))
        df = pd.read_csv(path)
        df["Topic"] = topic_id
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_enriched_metadata() -> pd.DataFrame:
    path = TOPICS_DIR / "document_metadata_enriched.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "title" in df.columns:
        df["title_norm"] = df["title"].apply(normalize_title)
    if "source_pdf" in df.columns:
        df["source_pdf"] = df["source_pdf"].astype(str)

    return df


def attach_labels(doc_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty:
        return doc_df

    return doc_df.merge(
        summary_df[["Topic", "topic_label"]],
        on="Topic",
        how="left"
    )


def merge_enriched_metadata(doc_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty or meta_df.empty:
        return doc_df

    df = doc_df.copy()

    # Prefer source_pdf join if available in both
    if "source_pdf" in df.columns and "source_pdf" in meta_df.columns:
        merged = df.merge(
            meta_df,
            on="source_pdf",
            how="left",
            suffixes=("", "_meta")
        )
        return merged

    # Fallback to title join
    if "title" in df.columns and "title" in meta_df.columns:
        df["title_norm"] = df["title"].apply(normalize_title)
        merged = df.merge(
            meta_df.drop(columns=["title"], errors="ignore"),
            on="title_norm",
            how="left",
            suffixes=("", "_meta")
        )
        return merged

    return df


# ---------------------------------------------------------------------------
# Cleaning / feature engineering
# ---------------------------------------------------------------------------

def prepare_year_data(doc_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty:
        return doc_df

    df = doc_df.copy()

    if "publication_year" not in df.columns:
        return pd.DataFrame()

    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df[df["publication_year"].between(1950, 2035, inclusive="both")].copy()

    if df.empty:
        return df

    df["publication_year"] = df["publication_year"].astype(int)
    df["period_start"] = (df["publication_year"] // 5) * 5
    df["period_label"] = (
        df["period_start"].astype(str) + "–" + (df["period_start"] + 4).astype(str)
    )

    return df


def prepare_citation_data(doc_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty or "citation_count" not in doc_df.columns:
        return pd.DataFrame()

    df = doc_df.copy()
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")
    df = df.dropna(subset=["citation_count"]).copy()

    if df.empty:
        return df

    if "publication_year" in df.columns:
        current_year = pd.Timestamp.now().year
        age = (current_year - pd.to_numeric(df["publication_year"], errors="coerce") + 1).clip(lower=1)
        df["citations_per_year"] = df["citation_count"] / age

    return df


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------

def compute_hottest_topics_by_period(doc_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty or "period_label" not in doc_df.columns:
        return pd.DataFrame()

    counts = (
        doc_df.groupby(["period_label", "Topic", "topic_label"])
        .size()
        .reset_index(name="n_publications")
    )

    hottest = (
        counts.sort_values(["period_label", "n_publications"], ascending=[True, False])
        .groupby("period_label", as_index=False)
        .head(1)
        .sort_values("period_label")
        .reset_index(drop=True)
    )

    return hottest


def compute_growth_table(doc_df: pd.DataFrame) -> pd.DataFrame:
    if doc_df.empty or "publication_year" not in doc_df.columns:
        return pd.DataFrame()

    yearly = (
        doc_df.groupby(["publication_year", "Topic", "topic_label"])
        .size()
        .reset_index(name="n_publications")
    )

    rows = []
    for (topic, label), group in yearly.groupby(["Topic", "topic_label"]):
        group = group.sort_values("publication_year")
        if len(group) < 2:
            continue

        x = group["publication_year"].to_numpy()
        y = group["n_publications"].to_numpy()
        slope = np.polyfit(x, y, 1)[0]

        rows.append({
            "Topic": topic,
            "topic_label": label,
            "growth_rate": float(slope),
            "total_publications": int(y.sum()),
            "first_year": int(x.min()),
            "last_year": int(x.max()),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("growth_rate", ascending=False)


def compute_average_citations_table(citation_df: pd.DataFrame) -> pd.DataFrame:
    if citation_df.empty:
        return pd.DataFrame()

    agg = {"citation_count": "mean"}
    if "citations_per_year" in citation_df.columns:
        agg["citations_per_year"] = "mean"

    avg_df = (
        citation_df.groupby(["Topic", "topic_label"])
        .agg(agg)
        .reset_index()
        .rename(columns={"citation_count": "avg_citations"})
    )

    if "citations_per_year" in avg_df.columns:
        avg_df = avg_df.rename(columns={"citations_per_year": "avg_citations_per_year"})

    return avg_df.sort_values("avg_citations", ascending=False)


def compute_stagnation_table(
    citation_df: pd.DataFrame,
    growth_df: pd.DataFrame,
) -> pd.DataFrame:
    if citation_df.empty or growth_df.empty:
        return pd.DataFrame()

    impact_df = compute_average_citations_table(citation_df)
    if impact_df.empty:
        return pd.DataFrame()

    merged = impact_df.merge(
        growth_df[["Topic", "topic_label", "growth_rate", "total_publications"]],
        on=["Topic", "topic_label"],
        how="inner",
    )

    if merged.empty:
        return merged

    impact_col = "avg_citations_per_year" if "avg_citations_per_year" in merged.columns else "avg_citations"
    merged["impact_metric"] = merged[impact_col]

    # High impact + low/negative momentum = historically important but stagnant fields.
    # Clip growth_rate to (-inf, 0] so that only non-growing topics score highly;
    # fast-growing topics are excluded regardless of their impact.
    clamped_growth = merged["growth_rate"].clip(upper=0.0)
    merged["stagnation_score"] = merged["impact_metric"] / (clamped_growth.abs() + 0.05)

    merged["momentum_class"] = np.where(
        merged["growth_rate"] > 0.25,
        "Emerging",
        np.where(merged["growth_rate"] < -0.25, "Declining", "Stagnant/Flat")
    )

    return merged.sort_values("stagnation_score", ascending=False)

def compute_opportunity_table(
    citation_df: pd.DataFrame,
    growth_df: pd.DataFrame,
) -> pd.DataFrame:
    if citation_df.empty or growth_df.empty:
        return pd.DataFrame()

    impact_df = compute_average_citations_table(citation_df)
    if impact_df.empty:
        return pd.DataFrame()

    merged = impact_df.merge(
        growth_df[["Topic", "topic_label", "growth_rate", "total_publications"]],
        on=["Topic", "topic_label"],
        how="inner",
    )

    if merged.empty:
        return merged

    impact_col = "avg_citations_per_year" if "avg_citations_per_year" in merged.columns else "avg_citations"
    merged["impact_metric"] = merged[impact_col]

    # Opportunity = active growth × low citation impact.
    # Clip growth_rate to [0, inf) so only growing topics score;
    # the inverse of impact rewards under-cited topics.
    clamped_growth = merged["growth_rate"].clip(lower=0.0)
    merged["opportunity_score"] = clamped_growth / (merged["impact_metric"] + 0.05)

    merged["momentum_class"] = np.where(
        merged["growth_rate"] > 0.25,
        "Emerging",
        np.where(merged["growth_rate"] < -0.25, "Declining", "Stagnant/Flat")
    )

    return merged.sort_values("opportunity_score", ascending=False)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_topics_all_time(summary_df: pd.DataFrame) -> None:
    top = summary_df.sort_values("Count", ascending=False).head(10).sort_values("Count")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["topic_label"], top["Count"])
    ax.set_xlabel("Number of Publications")
    ax.set_ylabel("Topic")
    ax.set_title("Top 10 Topics Across the Full Corpus")

    save_figure(fig, FIGURES_DIR / "top_10_topics_all_time.png")


def plot_hottest_topic_by_period(hottest_df: pd.DataFrame) -> None:
    if hottest_df.empty:
        return

    df = hottest_df.copy()
    df["label"] = df["period_label"] + " | " + df["topic_label"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df["label"], df["n_publications"])
    ax.set_xlabel("Number of Publications")
    ax.set_ylabel("5-Year Interval | Hottest Topic")
    ax.set_title("Hottest Topic in Each 5-Year Period")

    save_figure(fig, FIGURES_DIR / "hottest_topic_by_5year_period.png")


def plot_average_citations(avg_citations_df: pd.DataFrame) -> None:
    if avg_citations_df.empty:
        return

    top = avg_citations_df.head(10).sort_values("avg_citations")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["topic_label"], top["avg_citations"])
    ax.set_xlabel("Average Citations per Publication")
    ax.set_ylabel("Topic")
    ax.set_title("Topics with the Highest Average Citations")

    save_figure(fig, FIGURES_DIR / "average_citations_per_topic.png")


def plot_fastest_growing_topics(growth_df: pd.DataFrame) -> None:
    if growth_df.empty:
        return

    top = growth_df.head(10).sort_values("growth_rate")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["topic_label"], top["growth_rate"])
    ax.set_xlabel("Growth Rate (Publications per Year)")
    ax.set_ylabel("Topic")
    ax.set_title("Fastest-Growing Topics")

    save_figure(fig, FIGURES_DIR / "fastest_growing_topics.png")


def plot_stagnant_topics(stagnation_df: pd.DataFrame) -> None:
    if stagnation_df.empty:
        return

    top = stagnation_df.head(10).sort_values("stagnation_score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["topic_label"], top["stagnation_score"])
    ax.set_xlabel("Stagnation Score")
    ax.set_ylabel("Topic")
    ax.set_title("High-Impact but Low-Momentum Topics")

    save_figure(fig, FIGURES_DIR / "stagnant_topics.png")

def plot_latent_opportunity_topics(opportunity_df: pd.DataFrame) -> None:
    if opportunity_df.empty:
        return

    top = opportunity_df.head(10).sort_values("opportunity_score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["topic_label"], top["opportunity_score"])
    ax.set_xlabel("Latent Opportunity Score")
    ax.set_ylabel("Topic")
    ax.set_title("Low-Impact but High-Momentum Topics")

    save_figure(fig, FIGURES_DIR / "latent_opportunity_topics.png")

def plot_impact_vs_growth(opportunity_df: pd.DataFrame) -> None:
    """
    Bubble scatter: X = growth_rate (symlog), Y = impact_metric, size = total_publications.

    Four quadrants split at medians:
        High growth + Low impact  → Opportunity  (green)
        High growth + High impact → Dominant     (blue)
        Low growth  + High impact → Stagnant     (orange)
        Low growth  + Low impact  → Fading       (grey)

    Labels: top-4 per quadrant by the quadrant's primary score, so every region
    of the plot gets annotated rather than piling all labels near x=0.
    X-axis uses symlog scale so the dense cluster near 0 is readable while
    outliers (e.g. chatgpt, fast-growing topics) remain visible.
    """
    if opportunity_df.empty:
        return

    df = opportunity_df.copy().reset_index(drop=True)

    x_mid = df["growth_rate"].median()
    y_mid = df["impact_metric"].median()

    # ── Bubble size ────────────────────────────────────────────────────────
    size_col = "total_publications" if "total_publications" in df.columns else None
    if size_col:
        raw = df[size_col].clip(lower=1)
        sizes = 35 + 350 * (raw - raw.min()) / (raw.max() - raw.min() + 1)
    else:
        sizes = 70

    # ── Quadrant membership ────────────────────────────────────────────────
    df["_hg"] = df["growth_rate"] >= x_mid
    df["_hi"] = df["impact_metric"] >= y_mid

    COLOR_MAP = {
        (True,  False): "#2ca02c",   # Opportunity
        (True,  True):  "#1f77b4",   # Dominant
        (False, True):  "#ff7f0e",   # Stagnant
        (False, False): "#aaaaaa",   # Fading
    }
    colors = df.apply(lambda r: COLOR_MAP[(r["_hg"], r["_hi"])], axis=1)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))

    # Symlog scale: linear region ±linthresh around 0, log outside.
    # linthresh ≈ 10th-percentile of |nonzero growth| keeps the dense cluster spread.
    nonzero = df["growth_rate"][df["growth_rate"].abs() > 1e-6].abs()
    linthresh = float(nonzero.quantile(0.10)) if not nonzero.empty else 0.1
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.5)

    # ── Axis limits with generous padding ─────────────────────────────────
    ylim_pad = (df["impact_metric"].max() - df["impact_metric"].min()) * 0.10
    ymin = df["impact_metric"].min() - ylim_pad
    ymax = df["impact_metric"].max() + ylim_pad * 1.5   # extra top headroom for labels

    # x limits: leave breathing room on both sides in log space
    xmin_raw = df["growth_rate"].min()
    xmax_raw = df["growth_rate"].max()
    xmin = xmin_raw * 1.4 if xmin_raw < 0 else xmin_raw - linthresh
    xmax = xmax_raw * 1.4 if xmax_raw > 0 else xmax_raw + linthresh

    # ── Quadrant shading (done in axes-fraction coords to avoid symlog issues) ─
    # We need x_mid in axes fraction.  Use a helper that maps data→display→axes.
    def _data_to_axes_x(val):
        disp = ax.transData.transform((val, 0))[0]
        return ax.transAxes.inverted().transform((disp, 0))[0]

    def _data_to_axes_y(val):
        disp = ax.transData.transform((0, val))[1]
        return ax.transAxes.inverted().transform((0, disp))[1]

    # Draw shading after setting limits so transform is accurate
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.canvas.draw()   # flush so transforms are live

    xf = np.clip(_data_to_axes_x(x_mid), 0.01, 0.99)
    yf = np.clip(_data_to_axes_y(y_mid), 0.01, 0.99)

    quad_alpha = 0.07
    # Stagnant (top-left)
    ax.axhspan(y_mid, ymax, xmin=0,  xmax=xf, color="#ff7f0e", alpha=quad_alpha)
    # Dominant (top-right)
    ax.axhspan(y_mid, ymax, xmin=xf, xmax=1,  color="#1f77b4", alpha=quad_alpha)
    # Fading (bottom-left)
    ax.axhspan(ymin,  y_mid, xmin=0, xmax=xf, color="#aaaaaa", alpha=quad_alpha)
    # Opportunity (bottom-right)
    ax.axhspan(ymin,  y_mid, xmin=xf, xmax=1, color="#2ca02c", alpha=quad_alpha)

    # ── Divider lines ─────────────────────────────────────────────────────
    ax.axvline(x_mid, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axhline(y_mid, color="black", linewidth=0.8, linestyle="--", alpha=0.45)

    # ── Quadrant corner labels ─────────────────────────────────────────────
    # Place in axes-fraction coords so they're immune to scale changes
    corner_kw = dict(fontsize=8.5, fontweight="bold", ha="center",
                     alpha=0.6, transform=ax.transAxes)
    ax.text((0 + xf) / 2, 0.97,
            "STAGNANT\n(high impact, low growth)",    color="#ff7f0e", va="top",  **corner_kw)
    ax.text((xf + 1)  / 2, 0.97,
            "DOMINANT\n(high impact, high growth)",   color="#1f77b4", va="top",  **corner_kw)
    ax.text((0 + xf) / 2, 0.03,
            "FADING\n(low impact, low growth)",        color="#888888", va="bottom", **corner_kw)
    ax.text((xf + 1)  / 2, 0.03,
            "OPPORTUNITY\n(low impact, high growth)", color="#2ca02c", va="bottom", **corner_kw)

    # ── Scatter ────────────────────────────────────────────────────────────
    ax.scatter(
        df["growth_rate"], df["impact_metric"],
        s=sizes, c=colors, alpha=0.82, linewidths=0.4, edgecolors="white", zorder=3,
    )

    # ── Per-quadrant label selection ───────────────────────────────────────
    # Pick top-N from each quadrant independently so every region gets labels.
    LABELS_PER_QUAD = 4
    label_idx: set[int] = set()

    quadrants = {
        "opportunity": df[ df["_hg"] & ~df["_hi"]],
        "dominant":    df[ df["_hg"] &  df["_hi"]],
        "stagnant":    df[~df["_hg"] &  df["_hi"]],
        "fading":      df[~df["_hg"] & ~df["_hi"]],
    }
    score_col = {
        "opportunity": "opportunity_score",
        "dominant":    "impact_metric",
        "stagnant":    "stagnation_score" if "stagnation_score" in df.columns else "impact_metric",
        "fading":      "total_publications" if size_col else "growth_rate",
    }
    for qname, qdf in quadrants.items():
        if qdf.empty:
            continue
        col = score_col[qname]
        if col not in qdf.columns:
            col = "impact_metric"
        label_idx.update(qdf[col].nlargest(LABELS_PER_QUAD).index.tolist())

    # ── Annotate with leader lines ────────────────────────────────────────
    # Alternate text offsets to reduce overlap in the dense cluster region
    _offset_cycle = [
        (12,  10), (-12,  10), (12, -14), (-12, -14),
        (20,   4), (-20,   4), (20, -18), (-20, -18),
        ( 6,  18), ( -6,  18), ( 6, -22), ( -6, -22),
    ]
    for i, idx in enumerate(sorted(label_idx)):
        row = df.loc[idx]
        ox, oy = _offset_cycle[i % len(_offset_cycle)]
        ax.annotate(
            row["topic_label"],
            xy=(row["growth_rate"], row["impact_metric"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=7.5, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, lw=0),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7),
            zorder=5,
        )

    # ── Size legend ────────────────────────────────────────────────────────
    if size_col:
        q_vals = df[size_col].quantile([0.25, 0.75]).astype(int)
        legend_handles = []
        for q, label_text in zip(q_vals, ["Small topic", "Large topic"]):
            s_q = float(35 + 350 * (q - df[size_col].min()) /
                        (df[size_col].max() - df[size_col].min() + 1))
            legend_handles.append(
                ax.scatter([], [], s=s_q, color="grey", alpha=0.6, label=f"{label_text} (~{q} pubs)")
            )
        ax.legend(handles=legend_handles, title="Bubble = total publications",
                  loc="lower right", fontsize=8, title_fontsize=8, framealpha=0.75)

    ax.set_xlabel("Growth Rate (Publications per Year)  —  symlog scale", fontsize=11)
    ax.set_ylabel("Impact Metric (Avg Citations per Year)", fontsize=11)
    ax.set_title("Topic Impact vs. Growth — Landscape Map", fontsize=13, fontweight="bold")

    save_figure(fig, FIGURES_DIR / "impact_vs_growth.png")


# ---------------------------------------------------------------------------
# Nature-style panel
# ---------------------------------------------------------------------------

def plot_main_figure_panel(
    summary_df: pd.DataFrame,
    hottest_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
    stagnation_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A
    top = summary_df.sort_values("Count", ascending=False).head(10).sort_values("Count")
    ax1.barh(top["topic_label"], top["Count"])
    ax1.set_xlabel("Number of Publications")
    ax1.set_ylabel("Topic")
    ax1.set_title("A. Top Themes Across the Full Corpus")

    # Panel B
    if not hottest_df.empty:
        temp = hottest_df.copy()
        temp["label"] = temp["period_label"] + " | " + temp["topic_label"]
        ax2.barh(temp["label"], temp["n_publications"])
        ax2.set_xlabel("Number of Publications")
        ax2.set_ylabel("5-Year Interval | Hottest Topic")
        ax2.set_title("B. Hottest Topic by 5-Year Period")
    else:
        ax2.text(0.5, 0.5, "No usable publication_year data", ha="center", va="center")
        ax2.set_title("B. Hottest Topic by 5-Year Period")
        ax2.axis("off")

    # Panel C — Impact vs Growth bubble chart with symlog x-axis + per-quadrant labels
    if not opportunity_df.empty:
        odf = opportunity_df.copy().reset_index(drop=True)
        x_mid = odf["growth_rate"].median()
        y_mid = odf["impact_metric"].median()

        odf["_hg"] = odf["growth_rate"] >= x_mid
        odf["_hi"] = odf["impact_metric"] >= y_mid
        CMAP = {(True,False):"#2ca02c",(True,True):"#1f77b4",
                (False,True):"#ff7f0e",(False,False):"#aaaaaa"}
        colors = odf.apply(lambda r: CMAP[(r["_hg"], r["_hi"])], axis=1)

        size_col = "total_publications" if "total_publications" in odf.columns else None
        if size_col:
            raw = odf[size_col].clip(lower=1)
            sizes = 25 + 250 * (raw - raw.min()) / (raw.max() - raw.min() + 1)
        else:
            sizes = 50

        nonzero = odf["growth_rate"][odf["growth_rate"].abs() > 1e-6].abs()
        linthresh = float(nonzero.quantile(0.10)) if not nonzero.empty else 0.1
        ax3.set_xscale("symlog", linthresh=linthresh, linscale=0.5)

        ax3.scatter(odf["growth_rate"], odf["impact_metric"],
                    s=sizes, c=colors, alpha=0.78, linewidths=0.3,
                    edgecolors="white", zorder=3)
        ax3.axvline(x_mid, linestyle="--", color="black", linewidth=0.7, alpha=0.45)
        ax3.axhline(y_mid, linestyle="--", color="black", linewidth=0.7, alpha=0.45)

        # Per-quadrant labels (2 per quadrant)
        c3_label_idx: set = set()
        c3_quads = {
            "opp":  odf[ odf["_hg"] & ~odf["_hi"]],
            "dom":  odf[ odf["_hg"] &  odf["_hi"]],
            "stag": odf[~odf["_hg"] &  odf["_hi"]],
            "fade": odf[~odf["_hg"] & ~odf["_hi"]],
        }
        c3_qscore = {
            "opp":  "opportunity_score",
            "dom":  "impact_metric",
            "stag": "stagnation_score" if "stagnation_score" in odf.columns else "impact_metric",
            "fade": "total_publications" if size_col else "growth_rate",
        }
        for qn, qdf in c3_quads.items():
            if qdf.empty: continue
            col = c3_qscore[qn] if c3_qscore[qn] in qdf.columns else "impact_metric"
            c3_label_idx.update(qdf[col].nlargest(2).index.tolist())

        c3_offsets = [(8,6),(-8,6),(8,-10),(-8,-10),(14,2),(-14,2),(6,14),(-6,14)]
        for i, idx in enumerate(sorted(c3_label_idx)):
            row = odf.loc[idx]
            ox, oy = c3_offsets[i % len(c3_offsets)]
            ax3.annotate(row["topic_label"],
                         xy=(row["growth_rate"], row["impact_metric"]),
                         xytext=(ox, oy), textcoords="offset points",
                         fontsize=6,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                         arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5),
                         zorder=5)

        ax3.set_xlabel("Growth Rate (symlog)", fontsize=8)
        ax3.set_ylabel("Impact (Avg Cit/Year)", fontsize=8)
        ax3.set_title("C. Topic Impact vs. Growth")
    else:
        ax3.text(0.5, 0.5, "No usable citation_count data", ha="center", va="center")
        ax3.set_title("C. Topic Impact Versus Growth")
        ax3.axis("off")

    # Panel D
    if not opportunity_df.empty:
        stagnant = opportunity_df.head(10).sort_values("opportunity_score")
        ax4.barh(stagnant["topic_label"], stagnant["opportunity_score"])
        ax4.set_xlabel("Latent Opportunity Score")
        ax4.set_ylabel("Topic")
        ax4.set_title("D. Low-Impact but High-Momentum Topics")
    else:
        ax4.text(0.5, 0.5, "No usable citation_count data", ha="center", va="center")
        ax4.set_title("D. Low-Impact but High-Momentum Topics")
        ax4.axis("off")

    save_figure(fig, FIGURES_DIR / "figure_panel_main.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = load_topic_summary()
    doc_df = load_topic_documents()
    doc_df = attach_labels(doc_df, summary_df)

    enriched_meta_df = load_enriched_metadata()
    doc_df = merge_enriched_metadata(doc_df, enriched_meta_df)

    year_df = prepare_year_data(doc_df)
    citation_df = prepare_citation_data(doc_df)

    hottest_df = compute_hottest_topics_by_period(year_df)
    growth_df = compute_growth_table(year_df)
    avg_citations_df = compute_average_citations_table(citation_df)
    stagnation_df = compute_stagnation_table(citation_df, growth_df)
    opportunity_df = compute_opportunity_table(citation_df, growth_df)



    # Save tables
    if not hottest_df.empty:
        hottest_df.to_csv(FIGURES_DIR / "hottest_topics_by_period_table.csv", index=False)
    if not growth_df.empty:
        growth_df.to_csv(FIGURES_DIR / "topic_growth_table.csv", index=False)
    if not avg_citations_df.empty:
        avg_citations_df.to_csv(FIGURES_DIR / "average_citations_table.csv", index=False)
    if not stagnation_df.empty:
        stagnation_df.to_csv(FIGURES_DIR / "topic_stagnation_table.csv", index=False)
    if not opportunity_df.empty:
        opportunity_df.to_csv(FIGURES_DIR / "topic_opportunity_table.csv", index=False)

    # Original planned figures
    plot_top_topics_all_time(summary_df)
    plot_hottest_topic_by_period(hottest_df)
    plot_average_citations(avg_citations_df)
    plot_fastest_growing_topics(growth_df)

    # Added higher-level figures
    plot_latent_opportunity_topics(opportunity_df)
    plot_stagnant_topics(stagnation_df)
    plot_impact_vs_growth(opportunity_df)

    # Nature-style main panel
    plot_main_figure_panel(summary_df, hottest_df, opportunity_df, stagnation_df)

    print("Saved figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()