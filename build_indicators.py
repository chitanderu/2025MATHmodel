import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# 数据来源路径（基于之前的导出目录）
ECON_DIR = os.path.join("输出", "经济能源")
CARBON_DIR = os.path.join("输出", "碳排放")
EXCEL_SOURCE = "数据_区域双碳目标与路径规划研究（含拆分数据表）.xlsx"


@dataclass
class SeriesInfo:
    name: str
    unit: str
    values: pd.Series  # 索引为年份(int)，值为 float


def _is_year(c: str) -> bool:
    return bool(re.fullmatch(r"(19\d{2}|20\d{2})", str(c)))


def _normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip().replace("（", "(").replace("）", ")") for c in df.columns]
    year_cols = [c for c in df.columns if _is_year(str(c))]
    # 数值清洗
    for c in year_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    # 标准标签列
    for col in ["主题", "项目", "子项", "单位", "细分项"]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].astype(str).str.strip()
    # 清理脚注上标与尾随数字（如 “总量1” -> “总量”）
    for col in ["主题", "项目", "子项", "细分项"]:
        if col in df.columns:
            df[col] = df[col].apply(_cleanup_label_text)
    return df, sorted(year_cols, key=int)


def _read_csv(path: str) -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=object)
    return _normalize_df(df)


def _find_total_row(df: pd.DataFrame, topic: str, prefer_item_keywords: List[str]) -> Optional[pd.Series]:
    """在主题下寻找‘总量’行；若不存在，则返回 None。"""
    cand = df[df["主题"] == topic].copy()
    if cand.empty:
        return None
    # 优先：项目包含关键字 / 子项为“总量”或为空
    cand["项目"] = cand["项目"].apply(_cleanup_label_text)
    cand["子项"] = cand["子项"].apply(_cleanup_label_text)
    mask_kw = cand["项目"].fillna("").apply(lambda x: any(k in str(x) for k in prefer_item_keywords))
    mask_total = cand["子项"].fillna("").isin(["总量", "-", "", "——", "—", "— —"]) | cand["子项"].fillna("").str.fullmatch("总量")
    pref = cand[mask_kw & mask_total]
    if not pref.empty:
        return pref.iloc[0]
    # 其次：任何‘子项=总量’的行
    pref = cand[cand["子项"].fillna("") == "总量"]
    if not pref.empty:
        return pref.iloc[0]
    return None


def _sum_by_project(df: pd.DataFrame, topic: str, year_cols: List[str]) -> pd.DataFrame:
    """对同一主题下按项目求和（当没有‘总量’行时用于聚合）。"""
    part = df[df["主题"] == topic]
    if part.empty:
        return pd.DataFrame(columns=["项目"] + year_cols)
    num = part[["项目"] + year_cols].copy()
    num["项目"] = num["项目"].apply(_cleanup_label_text)
    num["项目"] = num["项目"].fillna("未命名")
    num[year_cols] = num[year_cols].apply(pd.to_numeric, errors="coerce")
    return num.groupby("项目", as_index=False)[year_cols].sum()


def _series_from_row(row: pd.Series, year_cols: List[str]) -> pd.Series:
    s = row[year_cols]
    s.index = s.index.astype(int)
    return pd.to_numeric(s, errors="coerce")


def read_core_series() -> Dict[str, SeriesInfo]:
    """读取核心时序：GDP、人口、能源消费总量、碳排放总量。"""
    res: Dict[str, SeriesInfo] = {}

    # 生产总值
    gdp_df, ycols_gdp = _read_csv(os.path.join(ECON_DIR, "生产总值.csv"))
    gdp_row = _find_total_row(gdp_df, topic="生产总值", prefer_item_keywords=["GDP", "生产总值", "地区生产总值"]) 
    if gdp_row is not None:
        res["gdp"] = SeriesInfo("地区生产总值", "亿元", _series_from_row(gdp_row, ycols_gdp))
    else:
        # 求和一二三产业
        proj_sum = _sum_by_project(gdp_df, "生产总值", ycols_gdp)
        mask = proj_sum["项目"].isin(["第一产业", "第二产业", "第三产业"]) 
        if mask.any():
            s = proj_sum.loc[mask, ycols_gdp].sum()
            s.index = s.index.astype(int)
            res["gdp"] = SeriesInfo("地区生产总值", "亿元", s)

    # 能源消费量
    e_df, ycols_e = _read_csv(os.path.join(ECON_DIR, "能源消费量.csv"))
    e_row = _find_total_row(e_df, topic="能源消费量", prefer_item_keywords=["能源消费", "总量"]) 
    if e_row is not None:
        res["energy"] = SeriesInfo("能源消费量", "万tce", _series_from_row(e_row, ycols_e))
    else:
        # 项目聚合（第一/二/三产业 + 居民生活）
        proj = _sum_by_project(e_df, "能源消费量", ycols_e)
        mask = proj["项目"].isin(["第一产业", "第二产业", "第三产业", "居民生活"]) | proj["项目"].str.contains("能源消费|总量", na=False)
        s = proj.loc[mask, ycols_e].sum()
        s.index = s.index.astype(int)
        res["energy"] = SeriesInfo("能源消费量", "万tce", s)

    # 碳排放量
    c_df, ycols_c = _read_csv(os.path.join(CARBON_DIR, "碳排放量.csv"))
    c_row = _find_total_row(c_df, topic="碳排放量", prefer_item_keywords=["碳排放量", "排放量", "总量"]) 
    if c_row is not None:
        res["co2"] = SeriesInfo("碳排放量", "万tCO2", _series_from_row(c_row, ycols_c))
    else:
        proj = _sum_by_project(c_df, "碳排放量", ycols_c)
        mask = proj["项目"].isin(["第一产业", "第二产业", "第三产业", "居民生活"]) | proj["项目"].str.contains("碳排放|总量", na=False)
        s = proj.loc[mask, ycols_c].sum()
        s.index = s.index.astype(int)
        res["co2"] = SeriesInfo("碳排放量", "万tCO2", s)

    # 人口（可能未导出到经济能源目录，因此从源 Excel 搜索）
    pop = try_read_population_from_outputs()
    if pop is None:
        pop = extract_population_from_excel(EXCEL_SOURCE)
    if pop is not None:
        res["population"] = pop
    else:
        print("警告: 未找到人口时序，将无法计算人均类与强度类部分指标。")

    return res


def try_read_population_from_outputs() -> Optional[SeriesInfo]:
    # 1) 若未来导出有人口.csv，优先读取
    path1 = os.path.join(ECON_DIR, "人口.csv")
    if os.path.exists(path1):
        df, ycols = _read_csv(path1)
        row = _find_total_row(df, topic="人口", prefer_item_keywords=["常住人口", "总人口", "人口"]) 
        if row is not None:
            return SeriesInfo("常住人口", "万人", _series_from_row(row, ycols))

    # 2) 尝试从经济与能源_四部分汇总.csv中提取“人口-常住人口-总量”
    path2 = os.path.join(ECON_DIR, "经济与能源_四部分汇总.csv")
    if os.path.exists(path2):
        df, ycols = _read_csv(path2)
        df["主题"] = df["主题"].apply(_cleanup_label_text)
        df["项目"] = df["项目"].apply(_cleanup_label_text)
        df["子项"] = df["子项"].apply(_cleanup_label_text)
        part = df[df["主题"] == "人口"]
        if not part.empty:
            row = _find_total_row(part, topic="人口", prefer_item_keywords=["常住人口", "人口", "总量"]) 
            if row is not None:
                unit = str(row.get("单位", "万人"))
                return SeriesInfo("常住人口", unit, _series_from_row(row, ycols))
    return None


def extract_population_from_excel(file_path: str) -> Optional[SeriesInfo]:
    if not os.path.exists(file_path):
        return None
    xl = pd.ExcelFile(file_path)
    # 优先选择名称含“经济/能源”的表
    sheets = [s for s in xl.sheet_names if any(k in str(s) for k in ["经济", "能源"]) ] or xl.sheet_names
    # 预读每个表，尝试检测“主题/项目/子项/单位 + 年份列”结构
    for sheet in sheets:
        try:
            df = xl.parse(sheet, header=None, dtype=object)
        except Exception:
            continue
        header_row = _find_header_row_for_excel(df)
        try:
            df2 = xl.parse(sheet, header=header_row, dtype=object)
        except Exception:
            continue
        df2, ycols = _normalize_df(df2)
        if not {"主题", "项目", "子项", "单位"}.issubset(set(df2.columns)):
            continue
        part = df2[df2["主题"].astype(str).str.contains("人口", na=False)]
        if part.empty:
            continue
        # 倾向选择“常住人口-总量”行（支持‘总量1’等脚注）
        topic_value = part.iloc[0]["主题"]
        row = _find_total_row(part, topic=topic_value, prefer_item_keywords=["常住人口", "人口", "总量"]) 
        if row is None:
            continue
        s = _series_from_row(row, ycols)
        unit = str(row.get("单位", "万人"))
        return SeriesInfo("常住人口", unit, s)
    return None


def _find_header_row_for_excel(df0: pd.DataFrame) -> int:
    max_rows = min(30, len(df0))
    keys = ["主题", "项目", "子项", "单位"]
    for i in range(max_rows):
        row = df0.iloc[i].astype(str)
        hits = sum(1 for v in row if any(k in str(v) for k in keys))
        if hits >= 2 and any("主题" in str(v) for v in row):
            return i
    return 0


def _cleanup_label_text(text) -> str:
    if text is None:
        return ""
    s = str(text).strip().replace("（", "(").replace("）", ")")
    # 去除尾部脚注数字/上标以及逗号分隔的脚注，如 “总量1” “其他2,3”
    s = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+(?:,[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+)*$", "", s)
    # 多余空白
    s = re.sub(r"\s+", " ", s)
    return s


def _align_index(series_dict: Dict[str, SeriesInfo]) -> List[int]:
    # si.values 是一个 pandas Series；不要再取 .values（那是 numpy 数组，无法 dropna）
    years_sets = [set(s.dropna().index.astype(int)) for s in (si.values for si in series_dict.values()) if s is not None]
    if not years_sets:
        return []
    years = sorted(set.union(*years_sets))
    return years


def compute_derived(series: Dict[str, SeriesInfo]) -> Dict[str, SeriesInfo]:
    res: Dict[str, SeriesInfo] = {}
    years = _align_index(series)
    if not years:
        return res

    def get(name: str) -> Optional[pd.Series]:
        return series[name].values.reindex(years) if name in series else None

    gdp = get("gdp")  # 亿元
    pop = get("population")  # 万人
    energy = get("energy")  # 万tce
    co2 = get("co2")  # 万tCO2

    # 人均 GDP（万元/人）：(亿元/万人) -> 万元/人
    if gdp is not None and pop is not None:
        s = gdp / pop
        res["gdp_per_capita"] = SeriesInfo("人均GDP", "万元/人", s)

    # 人均能耗（tce/人）：(万tce/万人) -> tce/人
    if energy is not None and pop is not None:
        s = energy / pop
        res["energy_per_capita"] = SeriesInfo("人均能源消费量", "tce/人", s)

    # 人均排放（tCO2/人）：(万tCO2/万人) -> tCO2/人
    if co2 is not None and pop is not None:
        s = co2 / pop
        res["co2_per_capita"] = SeriesInfo("人均碳排放量", "tCO2/人", s)

    # 能耗强度（tce/万元）：(万tce/亿元) -> tce/万元
    if energy is not None and gdp is not None:
        s = energy / gdp
        res["energy_intensity"] = SeriesInfo("能源强度(能耗/GDP)", "tce/万元", s)

    # 碳强度（tCO2/万元）：(万tCO2/亿元) -> tCO2/万元
    if co2 is not None and gdp is not None:
        s = co2 / gdp
        res["carbon_intensity"] = SeriesInfo("碳强度(碳排放/GDP)", "tCO2/万元", s)

    # 组合因子（tCO2/tce）：(万tCO2/万tce)
    if co2 is not None and energy is not None:
        s = co2 / energy
        res["emission_factor_observed"] = SeriesInfo("观测排放因子(碳/能)", "tCO2/tce", s)

    return res


def yoy(series: pd.Series) -> pd.Series:
    return series.pct_change()


def cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    n = s.index.max() - s.index.min()
    if n <= 0:
        return float("nan")
    return (s.iloc[-1] / s.iloc[0]) ** (1 / n) - 1


def export_series_dict(series_dict: Dict[str, SeriesInfo], out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, info in series_dict.items():
        s = info.values
        for year, val in s.items():
            rows.append({
                "indicator_key": key,
                "indicator_name": info.name,
                "unit": info.unit,
                "year": int(year),
                "value": float(val) if pd.notna(val) else None,
            })
    pd.DataFrame(rows).sort_values(["indicator_key", "year"]).to_csv(
        os.path.join(out_dir, filename), index=False, encoding="utf-8-sig"
    )


def extract_structures(out_dir: str) -> None:
    # GDP 分产业结构
    gdp_df, ycols_gdp = _read_csv(os.path.join(ECON_DIR, "生产总值.csv"))
    part = gdp_df[gdp_df["主题"] == "生产总值"].copy()
    # 优先选择“子项=总量”的行
    has_total = (part["子项"].fillna("") == "总量").any()
    if has_total:
        struct = part[part["子项"].fillna("") == "总量"][ ["项目"] + ycols_gdp ]
    else:
        struct = part.groupby("项目", as_index=False)[ycols_gdp].sum()
    struct_long = struct.melt(id_vars=["项目"], var_name="year", value_name="value").dropna()
    struct_pivot = struct_long.pivot_table(index="year", values="value", aggfunc="sum")
    struct_long["share"] = struct_long.apply(lambda r: r["value"] / struct_pivot.loc[r["year"], "value"] if struct_pivot.loc[r["year"], "value"] else None, axis=1)
    struct_long["year"] = struct_long["year"].astype(int)
    struct_long.rename(columns={"项目": "sector"}, inplace=True)
    os.makedirs(out_dir, exist_ok=True)
    struct_long.to_csv(os.path.join(out_dir, "结构_生产总值.csv"), index=False, encoding="utf-8-sig")

    # 能源消费量结构
    e_df, ycols_e = _read_csv(os.path.join(ECON_DIR, "能源消费量.csv"))
    part = e_df[e_df["主题"] == "能源消费量"].copy()
    has_total = (part["子项"].fillna("") == "总量").any()
    if has_total:
        struct = part[part["子项"].fillna("") == "总量"][ ["项目"] + ycols_e ]
    else:
        struct = part.groupby("项目", as_index=False)[ycols_e].sum()
    struct_long = struct.melt(id_vars=["项目"], var_name="year", value_name="value").dropna()
    base = struct_long.pivot_table(index="year", values="value", aggfunc="sum")
    struct_long["share"] = struct_long.apply(lambda r: r["value"] / base.loc[r["year"], "value"] if base.loc[r["year"], "value"] else None, axis=1)
    struct_long["year"] = struct_long["year"].astype(int)
    struct_long.rename(columns={"项目": "sector"}, inplace=True)
    struct_long.to_csv(os.path.join(out_dir, "结构_能源消费量.csv"), index=False, encoding="utf-8-sig")

    # 碳排放结构
    c_df, ycols_c = _read_csv(os.path.join(CARBON_DIR, "碳排放量.csv"))
    part = c_df[c_df["主题"] == "碳排放量"].copy()
    has_total = (part["子项"].fillna("") == "总量").any()
    if has_total:
        struct = part[part["子项"].fillna("") == "总量"][ ["项目"] + ycols_c ]
    else:
        struct = part.groupby("项目", as_index=False)[ycols_c].sum()
    struct_long = struct.melt(id_vars=["项目"], var_name="year", value_name="value").dropna()
    base = struct_long.pivot_table(index="year", values="value", aggfunc="sum")
    struct_long["share"] = struct_long.apply(lambda r: r["value"] / base.loc[r["year"], "value"] if base.loc[r["year"], "value"] else None, axis=1)
    struct_long["year"] = struct_long["year"].astype(int)
    struct_long.rename(columns={"项目": "sector"}, inplace=True)
    struct_long.to_csv(os.path.join(out_dir, "结构_碳排放量.csv"), index=False, encoding="utf-8-sig")


def main(out_dir: str = os.path.join("输出", "指标")) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 读取核心
    core = read_core_series()
    export_series_dict(core, out_dir, "核心指标时序.csv")

    # 衍生指标
    derived = compute_derived(core)
    export_series_dict(derived, out_dir, "衍生指标时序.csv")

    # 同比与 CAGR 简报
    if core:
        rows = []
        for key, info in {**core, **derived}.items():
            s = info.values.sort_index()
            yoy_last = yoy(s).iloc[-1] if len(s) > 1 else float("nan")
            rows.append({
                "indicator_key": key,
                "indicator_name": info.name,
                "unit": info.unit,
                "year_last": int(s.index.max()),
                "value_last": float(s.iloc[-1]) if pd.notna(s.iloc[-1]) else None,
                "yoy_last": float(yoy_last) if pd.notna(yoy_last) else None,
                "cagr": float(cagr(s)),
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "指标简报.csv"), index=False, encoding="utf-8-sig")

    # 结构指标
    extract_structures(out_dir)

    print("已导出: 核心指标时序、衍生指标时序、指标简报、结构三表 到:", out_dir)


if __name__ == "__main__":
    # 运行: python build_indicators.py
    main()
