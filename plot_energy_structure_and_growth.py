import os
import re
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator


ENERGY_PATH = os.path.join("输出", "经济能源", "能源消费量.csv")
OUT_DIR = os.path.join("输出", "图表")


def _ensure_font():
    try:
        names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for f in [
            "Microsoft YaHei",
            "SimHei",
            "Source Han Sans CN",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "SimSun",
        ]:
            if f in names:
                plt.rcParams["font.family"] = f
                plt.rcParams["font.sans-serif"] = [
                    "Microsoft YaHei","SimHei","Source Han Sans CN","Noto Sans CJK SC","WenQuanYi Zen Hei","Arial Unicode MS","SimSun"
                ]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def _cleanup_label_text(text) -> str:
    if text is None:
        return ""
    s = str(text).strip().replace("（", "(").replace("）", ")")
    s = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+(?:,[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+)*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip().replace("（", "(").replace("）", ")") for c in df.columns]
    ycols = [c for c in df.columns if re.fullmatch(r"(19\d{2}|20\d{2})", str(c))]
    for c in ycols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    for col in ["主题", "项目", "子项", "单位", "细分项"]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].apply(_cleanup_label_text)
    for col in ["主题", "项目", "子项", "细分项"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df, sorted(ycols, key=int)


def _aggregate_by_categories(part: pd.DataFrame, ycols: List[str]) -> pd.DataFrame:
    cats = ["第一产业", "第二产业", "第三产业", "居民生活"]
    res = {}
    for cat in cats:
        sub = part[part["项目"].apply(_cleanup_label_text) == cat]
        if sub.empty:
            continue
        mask_total = sub["子项"].fillna("").apply(_cleanup_label_text) == "总量"
        if mask_total.any():
            row = sub[mask_total].iloc[0]
            vals = pd.to_numeric(row[ycols], errors="coerce")
        else:
            # 对二维 DataFrame 逐列转为数值后求和
            vals = sub[ycols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
        res[cat] = vals
    return pd.DataFrame(res).T if res else pd.DataFrame(columns=ycols, index=[])


def read_energy_structure() -> Tuple[pd.DataFrame, List[int]]:
    if not os.path.exists(ENERGY_PATH):
        raise FileNotFoundError(ENERGY_PATH)
    df = pd.read_csv(ENERGY_PATH, dtype=object)
    df, ycols = _normalize_df(df)
    part = df[df["主题"] == "能源消费量"].copy()
    if part.empty:
        raise ValueError("能源消费量.csv 中未找到 ‘主题=能源消费量’ 的记录")
    data = _aggregate_by_categories(part, ycols)
    if data.empty:
        raise ValueError("未能从能源消费量中聚合到四类（第一/第二/第三产业、居民生活）。")
    years = [int(y) for y in ycols]
    return data, years


def plot_energy_structure_and_growth():
    _ensure_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    data, years = read_energy_structure()
    order = ["第一产业", "第二产业", "第三产业", "居民生活"]
    present = [c for c in order if c in data.index]
    df_year = data.loc[present].T
    df_year.index = [int(y) for y in df_year.index]
    df_year = df_year.sort_index()

    total = df_year.sum(axis=1)
    growth = total.pct_change() * 100.0

    colors = {
        "第一产业": "#4C78A8",
        "第二产业": "#C0504D",
        "第三产业": "#9BBB59",
        "居民生活": "#8064A2",
    }
    line_color = "#2F5C8F"

    fig, ax1 = plt.subplots(figsize=(11.5, 6.2), dpi=150)

    bottom = None
    handles = []
    labels = []
    for col in present:
        vals = df_year[col].astype(float).values
        h = ax1.bar(df_year.index, vals, bottom=bottom, color=colors[col], edgecolor="white", linewidth=0.6, label=col)
        handles.append(h)
        labels.append(col)
        bottom = (vals if bottom is None else bottom + vals)

    ax1.set_ylabel("能源消费量（万tce）")
    ax1.set_xlabel("年份")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(df_year.index, growth.values, color=line_color, marker="o", linewidth=2.2, label="能源消费增长率")
    ax2.set_ylabel("能源消费增长率")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))

    line_handle = ax2.get_lines()[0]
    ax1.legend(handles + [line_handle], labels + ["能源消费增长率"], loc="upper left", frameon=False)

    ax1.set_title("能源消费量构成与增长率")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(OUT_DIR, "能源消费构成_与增长率.png")
    out_csv = os.path.join(OUT_DIR, "能源消费构成_与增长率_数据.csv")
    fig.savefig(out_png)
    long_rows = []
    for y in df_year.index:
        for c in present:
            long_rows.append({"year": int(y), "category": c, "value_万tce": float(df_year.loc[y, c])})
        long_rows.append({"year": int(y), "category": "增长率%", "value_万tce": float("nan"), "growth_pct": float(growth.loc[y]) if pd.notna(growth.loc[y]) else None})
    pd.DataFrame(long_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    plt.close(fig)
    print("图已导出:", out_png)


if __name__ == "__main__":
    plot_energy_structure_and_growth()
