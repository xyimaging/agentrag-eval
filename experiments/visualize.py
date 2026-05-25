"""
可视化脚本：生成三种 RAG Pipeline 的对比图表。

用法：
  python experiments/visualize.py

输出到 results/figures/
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import RESULTS_DIR

# ── 样式设置 ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "NaiveRAG":           "#4C72B0",
    "SelfReflectiveRAG":  "#DD8452",
    "AgenticRAG":         "#55A868",
}
PIPELINE_ORDER = ["NaiveRAG", "SelfReflectiveRAG", "AgenticRAG"]
PIPELINE_LABELS = {
    "NaiveRAG":          "Naive RAG",
    "SelfReflectiveRAG": "Self-Reflective RAG",
    "AgenticRAG":        "Agentic RAG",
}
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_latest_summaries():
    """加载每个 Pipeline 最新的 summary JSON（按时间戳排序，取最新）。"""
    summaries = {}
    for pipeline in ["naive", "self", "agentic"]:
        pattern = os.path.join(RESULTS_DIR, "tables", f"{pipeline}_*_summary.json")
        files = sorted(glob.glob(pattern))
        # 过滤掉 5 样本的 smoke test（n_samples < 50）
        for f in reversed(files):
            with open(f) as fp:
                s = json.load(fp)
            if s.get("n_samples", 0) >= 50:
                summaries[s["pipeline"]] = s
                break
    return summaries


def load_latest_raws():
    """加载每个 Pipeline 最新的 raw JSON。"""
    raws = {}
    for pipeline in ["naive", "self", "agentic"]:
        pattern = os.path.join(RESULTS_DIR, "raw", f"{pipeline}_*_raw.json")
        files = sorted(glob.glob(pattern))
        for f in reversed(files):
            with open(f) as fp:
                data = json.load(fp)
            if len(data) >= 50:
                # 找对应 summary 的 pipeline name
                key = {"naive": "NaiveRAG", "self": "SelfReflectiveRAG", "agentic": "AgenticRAG"}[pipeline]
                raws[key] = data
                break
    return raws


# ── 图 1：Overall EM / F1 对比柱状图 ─────────────────────────────────────────
def plot_overall_metrics(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Overall Performance: Naive vs Self-Reflective vs Agentic RAG\n(HotpotQA, n=100)",
                 fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes, ["overall_em", "overall_f1"], ["Exact Match (EM)", "Token F1"]):
        names = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER]
        values = [summaries[p][metric] for p in PIPELINE_ORDER]
        colors = [COLORS[p] for p in PIPELINE_ORDER]
        bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_overall_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# ── 图 2：Bridge vs Comparison 分层 EM 对比 ──────────────────────────────────
def plot_stratified_em(summaries):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Exact Match by Question Type\n(Bridge vs Comparison, n=100)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(PIPELINE_ORDER))
    width = 0.3
    bridge_vals = [summaries[p]["bridge_em"] for p in PIPELINE_ORDER]
    comp_vals   = [summaries[p]["comparison_em"] for p in PIPELINE_ORDER]

    bars1 = ax.bar(x - width/2, bridge_vals, width, label="Bridge",
                   color=[COLORS[p] for p in PIPELINE_ORDER], alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width/2, comp_vals, width, label="Comparison",
                   color=[COLORS[p] for p in PIPELINE_ORDER], alpha=0.5, edgecolor="white",
                   hatch="///")

    for bars, vals in [(bars1, bridge_vals), (bars2, comp_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS[p] for p in PIPELINE_ORDER], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact Match")

    solid_patch = mpatches.Patch(color="gray", alpha=0.9, label="Bridge (solid)")
    hatch_patch = mpatches.Patch(color="gray", alpha=0.5, hatch="///", label="Comparison (hatched)")
    ax.legend(handles=[solid_patch, hatch_patch], fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_stratified_em.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# ── 图 3：性能 vs 成本 散点图（核心图） ──────────────────────────────────────
def plot_cost_performance(summaries):
    fig, ax = plt.subplots(figsize=(9, 7))

    costs = [summaries[p]["est_cost_usd"] for p in PIPELINE_ORDER]
    f1s   = [summaries[p]["overall_f1"]   for p in PIPELINE_ORDER]

    # Dashed connector shows cost-performance trajectory
    ax.plot(costs, f1s, "--", color="#AAAAAA", linewidth=1.4, zorder=2, alpha=0.7)

    # Scatter points
    for pipeline in PIPELINE_ORDER:
        s = summaries[pipeline]
        ax.scatter(s["est_cost_usd"], s["overall_f1"],
                   color=COLORS[pipeline], s=320, zorder=5,
                   edgecolors="white", linewidth=2)

    # Labels with per-point offset arrows to avoid overlap
    label_offsets = {
        "NaiveRAG":          (-110, -22),
        "SelfReflectiveRAG": (  12,  16),
        "AgenticRAG":        (  12, -22),
    }
    for pipeline in PIPELINE_ORDER:
        s = summaries[pipeline]
        ox, oy = label_offsets[pipeline]
        ax.annotate(
            PIPELINE_LABELS[pipeline],
            xy=(s["est_cost_usd"], s["overall_f1"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=10.5, fontweight="bold", color=COLORS[pipeline],
            arrowprops=dict(arrowstyle="-|>", color=COLORS[pipeline],
                            lw=1.2, connectionstyle="arc3,rad=0.05"),
        )
        # Cost+F1 sub-label beneath each point
        ax.text(s["est_cost_usd"], s["overall_f1"] - 0.0025,
                f"  F1={s['overall_f1']:.3f} | ${s['est_cost_usd']:.3f}",
                ha="left", va="top", fontsize=8.5, color="#555555")

    # Axis: zoom to the relevant range with generous padding
    x_pad = max(costs) * 0.15
    y_pad = 0.007
    ax.set_xlim(-x_pad * 0.3, max(costs) + x_pad)
    ax.set_ylim(min(f1s) - 4 * y_pad, max(f1s) + 8 * y_pad)

    ax.set_xlabel("Estimated API Cost per 100 Questions (USD)", fontsize=11)
    ax.set_ylabel("Token F1 Score", fontsize=11)
    ax.set_title("Cost–Performance Trade-off\n(All three pipelines evaluated on identical data & model)",
                 fontsize=13, fontweight="bold", pad=14)

    # "Better value" arrow annotation — placed in lower-right, away from points
    ax.annotate(
        "← lower cost\n↑ better F1\n= ideal direction",
        xy=(0.97, 0.07), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color="#777777",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#CCCCCC", alpha=0.85),
    )

    # Horizontal reference line at Naive F1 to highlight degradation
    naive_f1 = summaries["NaiveRAG"]["overall_f1"]
    ax.axhline(naive_f1, color=COLORS["NaiveRAG"], linewidth=0.8,
               linestyle=":", alpha=0.5, zorder=1)
    ax.text(max(costs) + x_pad * 0.05, naive_f1 + y_pad * 0.6,
            "Naive F1 baseline", fontsize=8, color=COLORS["NaiveRAG"], va="bottom")

    plt.tight_layout(pad=1.8)
    path = os.path.join(FIGURES_DIR, "fig3_cost_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# ── 图 4：API 调用次数 vs 延迟 ────────────────────────────────────────────────
def plot_efficiency(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Efficiency Comparison\n(API Calls & Latency, n=100)",
                 fontsize=13, fontweight="bold")

    for ax, metric, ylabel, title in zip(
        axes,
        ["avg_api_calls", "avg_latency_s"],
        ["Avg API Calls per Question", "Avg Latency (s/question)"],
        ["API Call Count", "Response Latency"]
    ):
        names = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER]
        values = [summaries[p][metric] for p in PIPELINE_ORDER]
        colors = [COLORS[p] for p in PIPELINE_ORDER]
        bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# ── 图 5：F1 分布（重新设计：堆叠条 + 小提琴） ───────────────────────────────
def plot_f1_distribution(raws):
    from eval.metrics import f1_score as compute_f1

    # ── 收集每个 pipeline 的 F1 列表 + 三档统计 ─────────────────────────────
    f1_data   = {}
    bucket_data = {}   # {"zero": %, "partial": %, "perfect": %}
    for pipeline in PIPELINE_ORDER:
        if pipeline not in raws:
            continue
        f1s = [compute_f1(r["pred_answer"], r["gold_answer"]) for r in raws[pipeline]]
        f1_data[pipeline] = f1s
        n = len(f1s)
        bucket_data[pipeline] = {
            "No Credit\n(F1 = 0)":      sum(1 for x in f1s if x == 0.0) / n * 100,
            "Partial Credit\n(0 < F1 < 1)": sum(1 for x in f1s if 0.0 < x < 1.0) / n * 100,
            "Perfect Match\n(F1 = 1)":  sum(1 for x in f1s if x == 1.0) / n * 100,
        }

    fig, (ax_bar, ax_vio) = plt.subplots(1, 2, figsize=(13, 6),
                                          gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("Per-Question F1 Score Distribution  (n=100)",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── 左图：堆叠横向条形图（三档占比） ─────────────────────────────────────
    bucket_labels = ["No Credit\n(F1 = 0)", "Partial Credit\n(0 < F1 < 1)", "Perfect Match\n(F1 = 1)"]
    bucket_colors = ["#D9534F", "#F0AD4E", "#5CB85C"]   # red / amber / green
    pipeline_names = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER if p in f1_data]

    lefts = np.zeros(len(PIPELINE_ORDER))
    for bk, bc in zip(bucket_labels, bucket_colors):
        vals = [bucket_data[p][bk] for p in PIPELINE_ORDER if p in f1_data]
        bars = ax_bar.barh(pipeline_names, vals, left=lefts, color=bc, label=bk,
                           edgecolor="white", linewidth=0.8, height=0.5)
        # Percentage label inside each segment (only if wide enough)
        for bar, val, left in zip(bars, vals, lefts):
            if val >= 6:
                ax_bar.text(left + val / 2, bar.get_y() + bar.get_height() / 2,
                            f"{val:.0f}%", ha="center", va="center",
                            fontsize=9.5, fontweight="bold", color="white")
        lefts = lefts + np.array(vals)

    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Percentage of Questions (%)", fontsize=10.5)
    ax_bar.set_title("Score Bracket Breakdown", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.invert_yaxis()   # Naive on top

    # ── 右图：小提琴图 + 个体数据点（jitter） ───────────────────────────────
    positions = list(range(1, len(PIPELINE_ORDER) + 1))
    vio_data  = [f1_data[p] for p in PIPELINE_ORDER if p in f1_data]
    vio_colors = [COLORS[p] for p in PIPELINE_ORDER if p in f1_data]

    vp = ax_vio.violinplot(vio_data, positions=positions,
                           showmedians=False, showextrema=False,
                           widths=0.65)
    for body, color in zip(vp["bodies"], vio_colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
        body.set_linewidth(1.5)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for pos, pipeline, color in zip(positions, PIPELINE_ORDER, vio_colors):
        if pipeline not in f1_data:
            continue
        f1s = np.array(f1_data[pipeline])
        jitter = rng.uniform(-0.12, 0.12, size=len(f1s))
        ax_vio.scatter(pos + jitter, f1s, color=color, s=18, alpha=0.55, zorder=3)

    # Mean marker
    for pos, pipeline, color in zip(positions, PIPELINE_ORDER, vio_colors):
        if pipeline not in f1_data:
            continue
        mean_val = np.mean(f1_data[pipeline])
        ax_vio.scatter(pos, mean_val, color="white", s=60, zorder=5, edgecolors=color, linewidth=2)
        ax_vio.text(pos, mean_val + 0.04, f"{mean_val:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    ax_vio.set_xticks(positions)
    ax_vio.set_xticklabels([PIPELINE_LABELS[p] for p in PIPELINE_ORDER if p in f1_data],
                            fontsize=9.5)
    ax_vio.set_ylim(-0.08, 1.18)
    ax_vio.set_ylabel("F1 Score", fontsize=10.5)
    ax_vio.set_title("Distribution Shape + Individual Questions\n(○ = mean)", fontsize=11, fontweight="bold")
    ax_vio.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    plt.tight_layout(pad=1.5)
    path = os.path.join(FIGURES_DIR, "fig5_f1_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# ── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print("加载实验结果...")
    summaries = load_latest_summaries()
    raws      = load_latest_raws()

    print(f"  已加载 Pipeline: {list(summaries.keys())}")
    print("\n生成可视化图表...")

    plot_overall_metrics(summaries)
    plot_stratified_em(summaries)
    plot_cost_performance(summaries)
    plot_efficiency(summaries)
    plot_f1_distribution(raws)

    print(f"\n全部图表保存至: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
