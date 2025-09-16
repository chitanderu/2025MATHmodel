import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib
# 使用无界面后端，避免 Qt 依赖导致的错误
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


CORE_PATH = os.path.join("输出", "指标", "核心指标时序.csv")
DERIVED_PATH = os.path.join("输出", "指标", "衍生指标时序.csv")
OUT_DIR = os.path.join("输出", "图表")


def _ensure_font():
    # 尝试设置常见中文字体，若无则忽略
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Source Han Sans CN",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "SimSun",
    ]
    try:
        names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for f in candidates:
            if f in names:
                plt.rcParams["font.family"] = f
                plt.rcParams["font.sans-serif"] = candidates
                break
    except Exception:
        pass
    # 负号与中文兼容
    plt.rcParams["axes.unicode_minus"] = False


def read_indicator(file_path: str, key: str) -> Optional[pd.Series]:
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if "indicator_key" not in df.columns:
        return None
    part = df[df["indicator_key"] == key].copy()
    if part.empty:
        return None
    part = part.dropna(subset=["year"]).copy()
    part["year"] = part["year"].astype(int)
    s = part.set_index("year")["value"].astype(float).sort_index()
    return s


def load_series() -> Tuple[pd.Series, pd.Series]:
    pop = read_indicator(CORE_PATH, "population")
    pco2 = read_indicator(DERIVED_PATH, "co2_per_capita")

    # 回退：若衍生表没有人均CO2，则由核心表计算
    if pco2 is None:
        co2 = read_indicator(CORE_PATH, "co2")
        if pop is not None and co2 is not None:
            idx = sorted(set(pop.index) & set(co2.index))
            pco2 = (co2.reindex(idx) / pop.reindex(idx)).rename("value")

    if pop is None or pco2 is None:
        missing = [name for name, s in [("人口(population)", pop), ("人均CO2(co2_per_capita)", pco2)] if s is None]
        raise FileNotFoundError(f"缺少指标: {', '.join(missing)}。请先运行 build_indicators.py 生成核心与衍生指标。")

    # 对齐年份
    idx = sorted(set(pop.index) & set(pco2.index))
    return pop.reindex(idx), pco2.reindex(idx)


def _round_down(x: float, step: float) -> float:
    import math
    return math.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    import math
    return math.ceil(x / step) * step


def plot(pop: pd.Series, pco2: pd.Series, title: str = "人口与人均CO2排放量（年度）") -> str:
    _ensure_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    years = pop.index.tolist()
    fig, ax1 = plt.subplots(figsize=(11.5, 5.6), dpi=150)

    # 颜色方案（柔和绿）
    bar_face = "#a8d5a2"
    bar_edge = "#2b6f3e"
    line_color = "#1e5631"

    # 柱状图：人口（万人）
    bars = ax1.bar(years, pop.values, width=0.6, color=bar_face, edgecolor=bar_edge, linewidth=1.0, alpha=0.9, label="人口（万人）")
    ax1.set_ylabel("人口（万人）", color=bar_edge)
    ax1.tick_params(axis="y", labelcolor=bar_edge)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
    # 左轴范围：贴合数据
    y1_min = max(0, _round_down(pop.min() * 0.95, 100))
    y1_max = _round_up(pop.max() * 1.05, 100)
    if y1_max <= y1_min:
        y1_max = y1_min + 100
    ax1.set_ylim(y1_min, y1_max)
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # 折线：人均CO2（tCO2/人）
    ax2 = ax1.twinx()
    ax2.plot(years, pco2.values, color=line_color, marker="o", markersize=4.5, linewidth=2.2, label="人均碳排放量（tCO$_2$/人）")
    ax2.set_ylabel("人均碳排放量（tCO$_2$/人）", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # 右轴范围：非负并留白
    pmin = float(min(pco2.min(), 0))
    pmax = float(max(pco2.max(), 0))
    ymin2 = 0.0 if pmin >= 0 else _round_down(pmin * 1.05, 0.1)
    ymax2 = _round_up(pmax * 1.10, 0.1)
    if ymax2 <= ymin2:
        ymax2 = ymin2 + 0.1
    ax2.set_ylim(ymin2, ymax2)
    ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

    # X 轴样式
    ax1.set_xlabel("年份")
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=0)

    # 网格与边框
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # 合并图例（放在图内左上，避免与标题重叠）
    lines, labels = [], []
    for ax in [ax1, ax2]:
        lns, lbs = ax.get_legend_handles_labels()
        lines += lns
        labels += lbs
    ax1.legend(lines, labels, loc="upper left", ncol=1, frameon=False)

    # 标题与备注
    ax1.set_title(title, pad=14)

    # 在柱顶/折线点添加标签（可读且不拥挤）
    for rect in bars:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + (y1_max - y1_min) * 0.01, f"{h:,.0f}",
                 ha="center", va="bottom", fontsize=8, color=bar_edge, alpha=0.9)
    for x, y in zip(years, pco2.values):
        ax2.text(x, y + (ymax2 - ymin2) * 0.015, f"{y:.2f}", color=line_color, fontsize=8, ha="center", va="bottom")

    # 保留顶部空间给标题，避免与图例重叠
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    # 导出
    out_png = os.path.join(OUT_DIR, "人口与人均CO2排放_柱线图.png")
    out_csv = os.path.join(OUT_DIR, "人口与人均CO2排放_数据.csv")
    fig.savefig(out_png)

    pd.DataFrame({"year": years, "population_万人": pop.values, "人均CO2_tCO2每人": pco2.values}).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    plt.close(fig)
    return out_png


def main():
    pop, pco2 = load_series()
    out = plot(pop, pco2)
    print("图已导出:", out)


if __name__ == "__main__":
    # 运行: python plot_population_vs_percap_co2.py
    main()
