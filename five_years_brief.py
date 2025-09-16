import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib

# 使用非交互后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# 文件路径
CARBON_PATH = os.path.join("输出", "碳排放", "碳排放量.csv")
ECON_GDP_PATH = os.path.join("输出", "经济能源", "生产总值.csv")
ECON_ENERGY_PATH = os.path.join("输出", "经济能源", "能源消费量.csv")
ECON_SUMMARY_PATH = os.path.join("输出", "经济能源", "经济与能源_四部分汇总.csv")
CORE_INDICATORS = os.path.join("输出", "指标", "核心指标时序.csv")

OUT_DIR = os.path.join("输出", "分析", "阶段评估")


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


def _is_year(c: str) -> bool:
    return bool(re.fullmatch(r"(19\d{2}|20\d{2})", str(c)))


def _normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    df = df.copy()
    df.columns = [str(c).strip().replace("（", "(").replace("）", ")") for c in df.columns]
    ycols = [c for c in df.columns if _is_year(str(c))]
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
    return df, sorted([int(c) for c in ycols])


def _read_series_from_core(key: str, unit_fallback: str) -> Optional[Tuple[pd.Series, str]]:
    if not os.path.exists(CORE_INDICATORS):
        return None
    df = pd.read_csv(CORE_INDICATORS)
    if "indicator_key" not in df.columns:
        return None
    part = df[df["indicator_key"] == key].copy()
    if part.empty:
        return None
    part["year"] = part["year"].astype(int)
    s = part.set_index("year")["value"].astype(float).sort_index()
    unit = part["unit"].dropna().iloc[0] if part["unit"].notna().any() else unit_fallback
    return s, unit


def _series_from_row(row: pd.Series, years: List[int]) -> pd.Series:
    s = row[[str(y) for y in years]]
    s.index = [int(i) for i in years]
    return pd.to_numeric(s, errors="coerce")


def _find_total_row(df: pd.DataFrame, topic: str, prefer_item_keywords: List[str]) -> Optional[pd.Series]:
    cand = df[df["主题"] == topic].copy()
    if cand.empty:
        return None
    cand["项目"] = cand["项目"].apply(_cleanup_label_text)
    cand["子项"] = cand["子项"].apply(_cleanup_label_text)
    mask_kw = cand["项目"].fillna("").apply(lambda x: any(k in str(x) for k in prefer_item_keywords))
    mask_total = cand["子项"].fillna("").isin(["总量", "-", "", "——", "—", "— —"]) | cand["子项"].fillna("").str.fullmatch("总量")
    pref = cand[mask_kw & mask_total]
    if not pref.empty:
        return pref.iloc[0]
    pref = cand[cand["子项"].fillna("") == "总量"]
    if not pref.empty:
        return pref.iloc[0]
    return None


def _read_co2_series() -> Tuple[pd.Series, str]:
    co2 = _read_series_from_core("co2", "万tCO2")
    if co2 is not None:
        return co2
    if not os.path.exists(CARBON_PATH):
        raise FileNotFoundError(CARBON_PATH)
    df = pd.read_csv(CARBON_PATH, dtype=object)
    df, years = _normalize_df(df)
    row = _find_total_row(df, "碳排放量", ["碳排放量", "总量"]) 
    if row is None:
        part = df[df["主题"] == "碳排放量"].copy()
        sub = part[part["子项"].fillna("") == "总量"]
        if not sub.empty:
            s = pd.to_numeric(sub[[str(y) for y in years]], errors="coerce").sum(axis=0)
        else:
            s = pd.Series(dtype=float)
        s.index = [int(i) for i in s.index]
        return s, "万tCO2"
    return _series_from_row(row, years), row.get("单位", "万tCO2")


def _read_energy_series() -> Tuple[pd.Series, str]:
    en = _read_series_from_core("energy", "万tce")
    if en is not None:
        return en
    if not os.path.exists(ECON_ENERGY_PATH):
        raise FileNotFoundError(ECON_ENERGY_PATH)
    df = pd.read_csv(ECON_ENERGY_PATH, dtype=object)
    df, years = _normalize_df(df)
    row = _find_total_row(df, "能源消费量", ["能源消费", "总量"]) 
    if row is None:
        part = df[df["主题"] == "能源消费量"].copy()
        sub = part[part["子项"].fillna("") == "总量"]
        s = pd.to_numeric(sub[[str(y) for y in years]], errors="coerce").sum(axis=0)
        s.index = [int(i) for i in s.index]
        return s, "万tce"
    return _series_from_row(row, years), row.get("单位", "万tce")


def _read_gdp_series() -> Tuple[pd.Series, str]:
    gdp = _read_series_from_core("gdp", "亿元")
    if gdp is not None:
        return gdp
    if not os.path.exists(ECON_GDP_PATH):
        raise FileNotFoundError(ECON_GDP_PATH)
    df = pd.read_csv(ECON_GDP_PATH, dtype=object)
    df, years = _normalize_df(df)
    row = _find_total_row(df, "生产总值", ["GDP", "生产总值", "地区生产总值"]) 
    if row is None:
        part = df[df["主题"] == "生产总值"].copy()
        sub = part[part["子项"].fillna("") == "总量"]
        s = pd.to_numeric(sub[[str(y) for y in years]], errors="coerce").sum(axis=0)
        s.index = [int(i) for i in s.index]
        return s, "亿元"
    return _series_from_row(row, years), row.get("单位", "亿元")


def _read_population_series() -> Tuple[pd.Series, str]:
    pop = _read_series_from_core("population", "万人")
    if pop is not None:
        return pop
    if not os.path.exists(ECON_SUMMARY_PATH):
        raise FileNotFoundError(ECON_SUMMARY_PATH)
    df = pd.read_csv(ECON_SUMMARY_PATH, dtype=object)
    df, years = _normalize_df(df)
    part = df[df["主题"] == "人口"].copy()
    row = _find_total_row(part, "人口", ["常住人口", "总人口", "人口"]) 
    if row is None:
        row = part.iloc[0]
    return _series_from_row(row, years), row.get("单位", "万人")


def align_series(series: Dict[str, Tuple[pd.Series, str]]) -> Tuple[List[int], Dict[str, Tuple[pd.Series, str]]]:
    sets = [set(s.index) for s, _ in series.values() if s is not None and len(s) > 0]
    if not sets:
        return [], series
    years = sorted(set.intersection(*sets))
    for k, (s, u) in series.items():
        series[k] = (s.reindex(years), u)
    return years, series


@dataclass
class StageStat:
    name: str
    unit: str
    stage: str
    start_year: int
    end_year: int
    base_2010: Optional[float]
    start_value: Optional[float]
    end_value: Optional[float]
    net_change: Optional[float]
    sum_value: Optional[float]
    cagr: Optional[float]
    yoy_mean: Optional[float]
    yoy_std: Optional[float]


def stage_statistics(name: str, unit: str, s: pd.Series, stage: Tuple[int, int]) -> StageStat:
    y0, y1 = stage
    sub = s.loc[y0:y1]
    if sub.dropna().empty:
        return StageStat(name, unit, f"{y0}-{y1}", y0, y1, s.get(2010, np.nan), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    start_val = float(sub.iloc[0]) if pd.notna(sub.iloc[0]) else np.nan
    end_val = float(sub.iloc[-1]) if pd.notna(sub.iloc[-1]) else np.nan
    net = end_val - start_val if (not np.isnan(end_val) and not np.isnan(start_val)) else np.nan
    sum_val = float(sub.sum()) if not sub.dropna().empty else np.nan
    n = y1 - y0
    cagr = ((end_val / start_val) ** (1 / n) - 1) if (n > 0 and start_val and not np.isnan(end_val) and not np.isnan(start_val) and start_val != 0) else np.nan
    yoy = s.pct_change().loc[y0:y1]
    return StageStat(
        name,
        unit,
        f"{y0}-{y1}",
        y0,
        y1,
        float(s.get(2010, np.nan)) if 2010 in s.index else np.nan,
        start_val,
        end_val,
        net,
        sum_val,
        float(cagr) if not np.isnan(cagr) else np.nan,
        float(yoy.dropna().mean()) if not yoy.dropna().empty else np.nan,
        float(yoy.dropna().std()) if not yoy.dropna().empty else np.nan,
    )


def export_summary(stats: List[StageStat]):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for st in stats:
        rows.append({
            "指标": st.name,
            "单位": st.unit,
            "阶段": st.stage,
            "阶段起止": f"{st.start_year}-{st.end_year}",
            "2010基期值": st.base_2010,
            "期初值": st.start_value,
            "期末值": st.end_value,
            "净增": st.net_change,
            "阶段累计": st.sum_value,
            "年均增速CAGR": st.cagr,
            "同比均值": st.yoy_mean,
            "同比波动(Std)": st.yoy_std,
        })
    out_csv = os.path.join(OUT_DIR, "阶段评估摘要.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("已导出:", out_csv)


def plot_overview(series: Dict[str, Tuple[pd.Series, str]], years: List[int]):
    _ensure_font()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not years:
        print("无重叠年份，跳过绘图")
        return
    def idx100(s: pd.Series) -> pd.Series:
        base = s.get(2010, np.nan)
        if pd.isna(base) or base == 0:
            return s / s.dropna().iloc[0] * 100
        return s / base * 100

    co2, _ = series["co2"]
    gdp, _ = series["gdp"]
    energy, _ = series["energy"]
    pop, _ = series["population"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.2), dpi=150, gridspec_kw={"height_ratios": [2.2, 1]})

    x = years
    ax1.plot(x, idx100(co2.reindex(years)), label="碳排放量指数(2010=100)", color="#C0504D", linewidth=2.0)
    ax1.plot(x, idx100(gdp.reindex(years)), label="GDP指数(2010=100)", color="#4C78A8", linewidth=2.0)
    ax1.plot(x, idx100(energy.reindex(years)), label="能源消费指数(2010=100)", color="#9BBB59", linewidth=2.0)
    ax1.plot(x, idx100(pop.reindex(years)), label="人口指数(2010=100)", color="#8064A2", linewidth=2.0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylabel("指数（2010=100）")
    ax1.set_xticks(x)
    ax1.legend(loc="upper left", ncol=2, frameon=False)
    ax1.set_title("2010为基期的主要指标指数与阶段增速对比")

    def cagr(s: pd.Series, y0: int, y1: int) -> float:
        try:
            start = s.loc[y0]
            end = s.loc[y1]
            n = y1 - y0
            return (end / start) ** (1 / n) - 1
        except Exception:
            return np.nan

    labels = ["十二五(2011-2015)", "十三五(2016-2020)"]
    x_pos = np.arange(len(labels))
    width = 0.25
    cagr_sets = {
        "碳排放": [cagr(co2, 2011, 2015), cagr(co2, 2016, 2020)],
        "GDP": [cagr(gdp, 2011, 2015), cagr(gdp, 2016, 2020)],
        "能源": [cagr(energy, 2011, 2015), cagr(energy, 2016, 2020)],
    }
    colors = {"碳排放": "#C0504D", "GDP": "#4C78A8", "能源": "#9BBB59"}
    for i, (k, vals) in enumerate(cagr_sets.items()):
        ax2.bar(x_pos + i * width - width, np.array(vals) * 100, width=width, label=k, color=colors[k])
        for xi, v in zip(x_pos + i * width - width, vals):
            if pd.notna(v):
                ax2.text(xi, v * 100 + 0.15, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("年均增速 CAGR")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT_DIR, "阶段评估_总览.png")
    fig.savefig(out_png)
    plt.close(fig)
    print("已导出图表:", out_png)


@dataclass
class IndicatorSeries:
    key: str
    name: str
    unit: str
    series: pd.Series


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    co2, unit_co2 = _read_co2_series()
    gdp, unit_gdp = _read_gdp_series()
    energy, unit_energy = _read_energy_series()
    population, unit_pop = _read_population_series()

    series = {
        "co2": (co2, unit_co2),
        "gdp": (gdp, unit_gdp),
        "energy": (energy, unit_energy),
        "population": (population, unit_pop),
    }
    years, series = align_series(series)
    if not years:
        raise RuntimeError("无法在四个指标之间找到共同年份区间，请检查源CSV年列是否一致。")

    stages = [(2011, 2015), (2016, 2020)]
    name_map = {
        "co2": ("碳排放量", unit_co2),
        "gdp": ("地区生产总值", unit_gdp),
        "energy": ("能源消费量", unit_energy),
        "population": ("常住人口", unit_pop),
    }
    stats: List[StageStat] = []
    for key, (ser, unit) in series.items():
        nm, un = name_map[key]
        for st in stages:
            stats.append(stage_statistics(nm, un, ser, st))

    export_summary(stats)
    plot_overview(series, years)


if __name__ == "__main__":
    _ensure_font()
    main()
