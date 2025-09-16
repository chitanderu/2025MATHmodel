import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib

# 非交互后端，避免 Qt 依赖
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator


CARBON_PATH = os.path.join("输出", "碳排放", "碳排放量.csv")
CARBON_SUMMARY_PATH = os.path.join("输出", "碳排放", "碳排放_四部分汇总.csv")
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
    # 识别年份列
    year_cols = [c for c in df.columns if re.fullmatch(r"(19\d{2}|20\d{2})", str(c))]
    # 数值清洗
    for c in year_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    # 文本清洗
    for col in ["主题", "项目", "子项", "单位", "细分项"]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].apply(_cleanup_label_text)
    # 处理合并单元格带来的空白：前向填充
    for col in ["主题", "项目", "子项", "细分项"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df, sorted(year_cols, key=int)


def _read_and_pick(path: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    if not os.path.exists(path):
        return None, []
    df = pd.read_csv(path, dtype=object)
    df, ycols = _normalize_df(df)
    part = df[df["主题"] == "碳排放量"].copy()
    if part.empty:
        return None, []
    return part, ycols


def _map_category_from_cols(row: pd.Series) -> Optional[str]:
    # 使用 项目/子项/细分项 综合判断，优先匹配明确类目
    p = _cleanup_label_text(row.get("项目", ""))
    s = _cleanup_label_text(row.get("子项", ""))
    d = _cleanup_label_text(row.get("细分项", ""))
    t = " ".join([p, s, d])

    # 先排除总计性的描述
    if any(x in {p, s, d} for x in ["碳排放量", "总量", "合计"]):
        # 交给外层聚合，不直接映射
        pass

    # 居民生活优先（避免被“三”等字样误判）
    if any(k in t for k in ["居民", "居民生活", "生活消费"]):
        return "居民生活"

    # 一二三产业
    if any(k in t for k in ["第一产业", "第一", "一产", "农林牧渔", "农业", "林业", "牧业", "渔业"]):
        return "第一产业"
    if any(k in t for k in ["第二产业", "第二", "二产", "工业", "制造", "采矿", "能源供应部门", "工业消费部门"]):
        return "第二产业"
    if any(k in t for k in ["第三产业", "第三", "三产", "服务", "交通消费部门", "建筑消费部门", "批发", "零售", "住宿", "餐饮"]):
        return "第三产业"

    return None


def _aggregate_by_categories(part: pd.DataFrame, ycols: List[str]) -> pd.DataFrame:
    # 针对每个大类：若存在“子项=总量”则取该行；否则对该类所有子项求和
    categories = ["第一产业", "第二产业", "第三产业", "居民生活"]
    res = {}
    for cat in categories:
        sub = part[part["项目"].apply(_cleanup_label_text) == cat]
        if sub.empty:
            continue
        mask_total = sub["子项"].fillna("").apply(_cleanup_label_text) == "总量"
        if mask_total.any():
            row = sub[mask_total].iloc[0]
            vals = pd.to_numeric(row[ycols], errors="coerce")  # Series 可直接使用
        else:
            # DataFrame 需逐列转换为数值再求和
            vals = sub[ycols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
        res[cat] = vals
    if not res:
        return pd.DataFrame(columns=ycols, index=[])
    df = pd.DataFrame(res).T
    df.columns = ycols
    return df


def read_carbon_structure() -> Tuple[pd.DataFrame, List[int]]:
    # 优先从汇总文件读取，若失败则回退到单表
    part, ycols = _read_and_pick(CARBON_SUMMARY_PATH)
    if part is None:
        part, ycols = _read_and_pick(CARBON_PATH)
        if part is None:
            raise FileNotFoundError("未找到碳排放量数据，请检查输出/碳排放 目录下的文件。")

    # 仅保留‘万tCO2’等总量单位，排除因子行（如 tCO2/tce、tCO2/kWh）
    if "单位" in part.columns:
        unit_str = part["单位"].astype(str).str.lower()
        mask_total_unit = unit_str.str.contains("万tco2") | unit_str.str.contains("万吨co2") | unit_str.str.contains("mtco2")
        if mask_total_unit.any():
            part = part[mask_total_unit].copy()

    # 以“项目”列为主进行聚合
    data = _aggregate_by_categories(part, ycols)
    if data.empty:
        # 回退：尝试更宽松的关键词映射
        part["分类"] = part.apply(_map_category_from_cols, axis=1)
        part = part[part["分类"].notna()].copy()
        if part.empty:
            raise ValueError("未能从碳排放数据中识别到 第一/第二/第三产业 或 居民生活 分类。")
        agg = part.groupby("分类", as_index=False)[ycols].sum()
        data = agg.set_index("分类")[ycols]
    years = [int(y) for y in ycols]
    return data, years


def plot_structure_and_growth():
    _ensure_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    data, years = read_carbon_structure()
    # 确保四类列顺序
    cats = ["第一产业", "第二产业", "第三产业", "居民生活"]
    present = [c for c in cats if c in data.index]
    missing = [c for c in cats if c not in data.index]
    if missing:
        print("提示: 以下分类在数据中未找到，将忽略:", missing)

    # 按年转置为列
    df_year = data.loc[present].T  # 行=年份，列=分类
    df_year.index = [int(y) for y in df_year.index]
    df_year = df_year.sort_index()
    # 合计与增长率
    total = df_year.sum(axis=1)
    growth = total.pct_change() * 100.0

    # 颜色方案（四分类 + 线）
    # 颜色按需求：第一产业(蓝) -> 第二产业(红) -> 第三产业(绿) -> 居民生活(紫)
    colors = {
        "第一产业": "#4C78A8",  # 蓝
        "第二产业": "#C0504D",  # 红（更接近办公配色）
        "第三产业": "#9BBB59",  # 绿
        "居民生活": "#8064A2",  # 紫
    }
    line_color = "#2F5C8F"

    # 绘图
    fig, ax1 = plt.subplots(figsize=(11.5, 6.2), dpi=150)

    # 堆叠柱
    bottom = None
    bar_handles = []
    for col in present:  # 顺序：第一->第二->第三->居民
        vals = df_year[col].astype(float).values
        h = ax1.bar(df_year.index, vals, bottom=bottom, color=colors[col], edgecolor="white", linewidth=0.6, label=col)
        bar_handles.append((col, h))
        bottom = (vals if bottom is None else bottom + vals)

    ax1.set_ylabel("碳排放量（万tCO$_2$）")
    ax1.set_xlabel("年份")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax1.set_axisbelow(True)

    # 增长率折线（右轴）
    ax2 = ax1.twinx()
    ax2.plot(df_year.index, growth.values, color=line_color, marker="o", linewidth=2.2, label="碳排放增长率")
    ax2.set_ylabel("碳排放增长率")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))

    # 图例
    # legend 顺序严格使用 present，然后添加增长率
    line_handle = ax2.get_lines()[0]
    handles = [bh[1] for bh in bar_handles]
    labels = [bh[0] for bh in bar_handles]
    handles.append(line_handle)
    labels.append("碳排放增长率")
    ax1.legend(handles, labels, loc="upper left", frameon=False, ncol=1)

    # 标题与边距
    ax1.set_title("碳排放量构成与碳排放增长率")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 导出
    out_png = os.path.join(OUT_DIR, "碳排放构成_与增长率.png")
    out_csv = os.path.join(OUT_DIR, "碳排放构成_与增长率_数据.csv")
    fig.savefig(out_png)
    # 数据导出（长表）
    long_rows = []
    for y in df_year.index:
        for c in present:
            long_rows.append({"year": int(y), "category": c, "value_万tCO2": float(df_year.loc[y, c])})
        long_rows.append({"year": int(y), "category": "增长率%", "value_万tCO2": float("nan"), "growth_pct": float(growth.loc[y]) if pd.notna(growth.loc[y]) else None})
    pd.DataFrame(long_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.close(fig)
    print("图已导出:", out_png)


if __name__ == "__main__":
    plot_structure_and_growth()
