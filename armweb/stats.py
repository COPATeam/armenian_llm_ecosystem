"""Corpus statistics + paper figures over pipeline artifacts.

Figures follow the dataviz method: single-hue marks (#2a78d6), recessive grids,
no dual axes (two panels instead), text in near-black ink on white surface.
"""
import json
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import iter_jsonl, write_stats, REPORTS

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(\+374|00374|0)\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.6, zorder=0)


def funnel_table(stage_reports: dict) -> list[dict]:
    """stage_reports: name -> {docs, bytes}; returns ordered funnel rows with deltas."""
    rows, prev = [], None
    for name, d in stage_reports.items():
        row = {"stage": name, "docs": d["docs"], "bytes": d["bytes"]}
        if prev:
            row["docs_pct_of_prev"] = round(100 * d["docs"] / prev["docs"], 2) if prev["docs"] else None
            row["bytes_pct_of_prev"] = (round(100 * d["bytes"] / prev["bytes"], 2) if (d.get("bytes") is not None and prev.get("bytes")) else None)
        rows.append(row)
        prev = d
    return rows


def cluster_histogram(dedup_stats: dict, survivors_path, out_png):
    """Log-log histogram of fuzzy-cluster sizes (from survivors' cluster_size field)."""
    sizes = Counter()
    for r in iter_jsonl(survivors_path):
        sizes[r.get("cluster_size", 1)] += 1
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=200, facecolor=SURFACE)
    _style(ax)
    xs = sorted(sizes)
    ax.plot(xs, [sizes[x] for x in xs], marker="o", markersize=3.5,
            linewidth=0, color=BLUE, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("duplicate-cluster size (docs merged)", color=INK, fontsize=10)
    ax.set_ylabel("number of clusters", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return {int(k): int(v) for k, v in sorted(sizes.items())}


def length_histogram(splits_paths: list, out_png):
    lengths = []
    for p in splits_paths:
        for r in iter_jsonl(p):
            lengths.append(len(r["text"]))
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.hist(lengths, bins=[10 ** (i / 10) for i in range(20, 55)],
            color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("document length (chars)", color=INK, fontsize=10)
    ax.set_ylabel("documents", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    n = len(lengths)
    lengths.sort()
    return {"docs": n, "median_chars": lengths[n // 2] if n else 0,
            "p90_chars": lengths[int(n * 0.9)] if n else 0,
            "total_chars": sum(lengths)}


def per_source_month(splits_paths: list, out_csv_source, out_csv_month, out_png):
    """Per-source and per-month doc counts + mean cluster size (syndication proxy)."""
    src = defaultdict(lambda: [0, 0])   # source -> [docs, sum_cluster]
    mon = defaultdict(lambda: [0, 0])
    for p in splits_paths:
        for r in iter_jsonl(p):
            cs = r.get("cluster_size", 1)
            s = r.get("source") or "unknown"
            src[s][0] += 1
            src[s][1] += cs
            d = r.get("post_date") or ""
            m = d[:7] if len(d) >= 7 and d[:7] >= "1998-01" else "unknown"
            mon[m][0] += 1
            mon[m][1] += cs
    with open(out_csv_source, "w", encoding="utf-8") as f:
        f.write("source,docs,mean_cluster_size\n")
        for s, (n, c) in sorted(src.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{s},{n},{c / n:.3f}\n")
    with open(out_csv_month, "w", encoding="utf-8") as f:
        f.write("month,docs,mean_cluster_size\n")
        for m, (n, c) in sorted(mon.items()):
            f.write(f"{m},{n},{c / n:.3f}\n")
    months = [m for m in sorted(mon) if m != "unknown"]
    if months:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), dpi=200,
                                       facecolor=SURFACE, sharex=True)
        for ax in (ax1, ax2):
            _style(ax)
        xs = range(len(months))
        ax1.bar(xs, [mon[m][0] for m in months], color=BLUE, width=0.9, zorder=3)
        ax1.set_ylabel("docs / month", color=INK, fontsize=9)
        ax2.bar(xs, [mon[m][1] / mon[m][0] for m in months], color=ORANGE,
                width=0.9, zorder=3)
        ax2.set_ylabel("mean cluster size", color=INK, fontsize=9)
        step = max(1, len(months) // 12)
        ax2.set_xticks(list(xs)[::step])
        ax2.set_xticklabels(months[::step], rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_png, facecolor=SURFACE)
        plt.close(fig)
    return {"sources": len(src), "months": len(months)}


def pii_scan(splits_paths: list) -> dict:
    emails = phones = docs_with_email = docs_with_phone = n = 0
    for p in splits_paths:
        for r in iter_jsonl(p):
            n += 1
            e = len(EMAIL_RE.findall(r["text"]))
            ph = len(PHONE_RE.findall(r["text"]))
            emails += e
            phones += ph
            docs_with_email += e > 0
            docs_with_phone += ph > 0
    return {"docs": n, "emails": emails, "docs_with_email": docs_with_email,
            "phones_matched": phones, "docs_with_phone": docs_with_phone}


def clusters_review_csv(clusters_sample_path, out_csv):
    rows = list(iter_jsonl(clusters_sample_path))
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("group,n_docs_sampled,category,example_source,example_text\n")
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ex = rs[0]
            text = ex["text"][:200].replace('"', "'").replace("\n", " ")
            f.write(f'{g},{len(rs)},,{ex.get("source", "")},"{text}"\n')
    return len(groups)
