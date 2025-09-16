import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# 使用无界面后端，避免 Qt 依赖
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 数据路径
CARBON_PATH = os.path.join("输出", "碳排放", "碳排放量.csv")
ECON_GDP_PATH = os.path.join("输出", "经济能源", "生产总值.csv")
ECON_ENERGY_PATH = os.path.join("输出", "经济能源", "能源消费量.csv")
ECON_VARE_PATH = os.path.join("输出", "经济能源", "能耗品种结构.csv")
CARBON_GRID_FACTOR = os.path.join("输出", "碳排放", "外地调入电力碳排放因子.csv")
SUPPLY_FACTOR = os.path.join("输出", "碳排放", "能源供应部门碳排放因子.csv")
ECON_SUMMARY_PATH = os.path.join("输出", "经济能源", "经济与能源_四部分汇总.csv")
# 原始工作簿（用于从“经济与能源”子表回读人口）
EXCEL_SOURCE = "数据_区域双碳目标与路径规划研究（含拆分数据表）.xlsx"

OUT_DIR = os.path.join("输出", "分析", "关系模型")


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


# ---------- 通用清洗 ----------

def _cleanup_text(s) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace("（", "(").replace("）", ")")
    t = re.sub(r"\s+", " ", t)
    # 去尾部脚注数字/上标
    t = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+(?:,[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+)*$", "", t)
    return t


def _is_year(col: str) -> bool:
    return bool(re.fullmatch(r"(19\d{2}|20\d{2})", str(col)))


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
            df[col] = df[col].apply(_cleanup_text)
    for col in ["主题", "项目", "子项", "细分项"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df, sorted([int(c) for c in ycols])


def _find_total_row(df: pd.DataFrame, topic: str, prefer_item_keywords: List[str]) -> Optional[pd.Series]:
    cand = df[df["主题"] == topic].copy()
    if cand.empty:
        return None
    cand["项目"] = cand["项目"].apply(_cleanup_text)
    cand["子项"] = cand["子项"].apply(_cleanup_text)
    mask_kw = cand["项目"].fillna("").apply(lambda x: any(k in str(x) for k in prefer_item_keywords))
    mask_total = cand["子项"].fillna("").isin(["总量", "-", "", "——", "—", "— —"]) | cand["子项"].fillna("").str.fullmatch("总量")
    pref = cand[mask_kw & mask_total]
    if not pref.empty:
        return pref.iloc[0]
    pref = cand[cand["子项"].fillna("") == "总量"]
    if not pref.empty:
        return pref.iloc[0]
    return None


# ---------- 读取核心时序 ----------

def read_co2_series() -> pd.Series:
    if not os.path.exists(CARBON_PATH):
        raise FileNotFoundError(CARBON_PATH)
    df = pd.read_csv(CARBON_PATH, dtype=object)
    df, years = _normalize_df(df)
    row = _find_total_row(df, "碳排放量", ["碳排放量", "总量"]) 
    if row is not None:
        s = row[[str(y) for y in years]]
        s.index = [int(i) for i in years]
        return pd.to_numeric(s, errors="coerce")
    # 汇总子类总量
    part = df[df["主题"] == "碳排放量"].copy()
    sub = part[part["子项"].fillna("") == "总量"]
    s = pd.to_numeric(sub[[str(y) for y in years]], errors="coerce").sum(axis=0)
    s.index = [int(i) for i in s.index]
    return s


def read_gdp_series() -> pd.Series:
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
        return s
    s = row[[str(y) for y in years]]
    s.index = [int(i) for i in years]
    return pd.to_numeric(s, errors="coerce")


def read_energy_series() -> pd.Series:
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
        return s
    s = row[[str(y) for y in years]]
    s.index = [int(i) for i in years]
    return pd.to_numeric(s, errors="coerce")


def read_population_series() -> pd.Series:
    # 优先：从已导出的经济与能源汇总读取
    if os.path.exists(ECON_SUMMARY_PATH):
        df = pd.read_csv(ECON_SUMMARY_PATH, dtype=object)
        df, years = _normalize_df(df)
        part = df[df["主题"] == "人口"].copy()
        if not part.empty:
            row = _find_total_row(part, "人口", ["常住人口", "总人口", "人口"]) 
            if row is None:
                row = part.iloc[0]
            s = row[[str(y) for y in years]]
            s.index = [int(i) for i in years]
            return pd.to_numeric(s, errors="coerce")

    # 回退：直接从原始 Excel 的“经济/能源”相关工作表寻找人口行
    if os.path.exists(EXCEL_SOURCE):
        s = _read_population_from_excel(EXCEL_SOURCE)
        if s is not None:
            return s

    raise ValueError("未在经济与能源汇总或原始Excel中识别到‘人口-常住人口-总量’时序")


def _find_header_row_for_excel(df0: pd.DataFrame) -> int:
    max_rows = min(30, len(df0))
    keys = ["主题", "项目", "子项", "单位"]
    for i in range(max_rows):
        row = df0.iloc[i].astype(str)
        hits = sum(1 for v in row if any(k in str(v) for k in keys))
        if hits >= 2 and any("主题" in str(v) for v in row):
            return i
    return 0


def _read_population_from_excel(file_path: str) -> Optional[pd.Series]:
    try:
        xl = pd.ExcelFile(file_path)
    except Exception:
        return None
    # 优先选择名称含“经济/能源/人口”的表
    sheets = [s for s in xl.sheet_names if any(k in str(s) for k in ["经济", "能源", "人口"]) ] or xl.sheet_names
    for sheet in sheets:
        try:
            preview = xl.parse(sheet, header=None, dtype=object, nrows=30)
        except Exception:
            continue
        h = _find_header_row_for_excel(preview)
        try:
            df = xl.parse(sheet, header=h, dtype=object)
        except Exception:
            continue
        df, years = _normalize_df(df)
        if not {"主题", "项目", "子项", "单位"}.issubset(set(df.columns)):
            continue
        part = df[df["主题"].astype(str).str.contains("人口", na=False)]
        if part.empty:
            continue
        row = _find_total_row(part, "人口", ["常住人口", "人口", "总量"]) 
        if row is None:
            row = part.iloc[0]
        s = pd.to_numeric(row[[str(y) for y in years]], errors="coerce")
        s.index = [int(i) for i in years]
        return s
    return None


# ---------- 二级指标：非化石占比 ----------

NON_FOSSIL_KEYS = ["水电", "核电", "风电", "太阳", "光伏", "生物", "地热", "非化石", "可再生"]
ELECTRICITY_KEYS = ["电力", "电能"]


def read_non_fossil_share(energy_total: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """返回 (非化石占比, 电力占终端比重)，若无法从品种结构直接识别，返回近似估计和电力份额(若可)。"""
    nf_share = pd.Series(dtype=float)
    power_share = pd.Series(dtype=float)
    if os.path.exists(ECON_VARE_PATH):
        df = pd.read_csv(ECON_VARE_PATH, dtype=object)
        df, years = _normalize_df(df)
        part = df[df["主题"].str.contains("能耗品种结构|品种结构|能源品种", na=False)].copy()
        if part.empty:
            part = df  # 兜底
        # 标记非化石与电力
        def mark(row: pd.Series) -> Tuple[bool, bool]:
            txt = " ".join([str(row.get("项目", "")), str(row.get("子项", "")), str(row.get("细分项", ""))])
            txt = _cleanup_text(txt)
            is_nf = any(k in txt for k in NON_FOSSIL_KEYS)
            is_power = any(k in txt for k in ELECTRICITY_KEYS)
            return is_nf, is_power

        flags = part.apply(mark, axis=1, result_type="expand")
        part["is_nf"] = flags[0]
        part["is_power"] = flags[1]
        year_cols = [c for c in part.columns if _is_year(c)]
        # 非化石分量求和
        if part["is_nf"].any():
            nf_mat = part.loc[part["is_nf"], year_cols]
            nf_vals = nf_mat.apply(pd.to_numeric, errors="coerce").sum(axis=0)
            nf_vals.index = nf_vals.index.astype(int)
            nf_share = nf_vals.reindex(energy_total.index) / energy_total
        # 电力份额
        if part["is_power"].any():
            p_mat = part.loc[part["is_power"], year_cols]
            p_vals = p_mat.apply(pd.to_numeric, errors="coerce").sum(axis=0)
            p_vals.index = p_vals.index.astype(int)
            power_share = p_vals.reindex(energy_total.index) / energy_total

    # 如未得到非化石占比，尝试用电网因子近似
    if nf_share.empty:
        grid = read_grid_factor()
        if grid is not None:
            # 参考火电因子（本地口径可调整）
            EF_fossil = 0.85  # tCO2/MWh 的典型值
            s_power_nf = 1 - grid / EF_fossil
            s_power_nf = s_power_nf.clip(lower=0, upper=1)
            if power_share.empty:
                nf_share = s_power_nf  # 作为“电力非化石占比”近似输出
            else:
                nf_share = (s_power_nf * power_share).reindex(energy_total.index)
    return nf_share.reindex(energy_total.index), power_share.reindex(energy_total.index)


def read_grid_factor() -> Optional[pd.Series]:
    # 读取电网/外购电排放因子
    path = CARBON_GRID_FACTOR if os.path.exists(CARBON_GRID_FACTOR) else SUPPLY_FACTOR
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=object)
    df, years = _normalize_df(df)
    # 取“外地调入电力碳排放因子”或“电力/供电”相关行
    def is_grid_row(row: pd.Series) -> bool:
        t = " ".join([str(row.get("项目", "")), str(row.get("子项", "")), str(row.get("细分项", ""))])
        t = _cleanup_text(t)
        return any(k in t for k in ["电力", "供电", "电网", "外购电", "外来电"]) and any(k in t for k in ["因子", "/kwh", "kwh", "/mwh", "tco2"]) 

    sub = df[df.apply(is_grid_row, axis=1)]
    if sub.empty:
        return None
    # 取单位为 tCO2/kWh 或 tCO2/MWh 的行，若是每MWh，将换算到每kWh
    row = sub.iloc[0]
    s = pd.to_numeric(row[[str(y) for y in years]], errors="coerce")
    s.index = [int(i) for i in years]
    unit = str(row.get("单位", "")).lower()
    if "mwh" in unit and s.max() > 1:  # 单位 tCO2/MWh 值较大，需要换算到每kWh
        s = s / 1000.0
    return s


# ---------- 主数据表构建 ----------

def build_master_table() -> pd.DataFrame:
    co2 = read_co2_series()  # 万tCO2
    gdp = read_gdp_series()  # 亿元
    energy = read_energy_series()  # 万tce
    pop = read_population_series()  # 万人
    idx = sorted(set(co2.index) & set(gdp.index) & set(energy.index) & set(pop.index))
    co2 = co2.reindex(idx)
    gdp = gdp.reindex(idx)
    energy = energy.reindex(idx)
    pop = pop.reindex(idx)

    df = pd.DataFrame({
        "co2_万tCO2": co2,
        "gdp_亿元": gdp,
        "energy_万tce": energy,
        "population_万人": pop,
    }, index=idx)
    # 派生
    df["人均CO2_tCO2_人"] = df["co2_万tCO2"] / df["population_万人"]
    df["碳强度_tCO2_万元"] = df["co2_万tCO2"] / df["gdp_亿元"]
    df["能源强度_tce_万元"] = df["energy_万tce"] / df["gdp_亿元"]
    df["观测排放因子_tCO2_tce"] = df["co2_万tCO2"] / df["energy_万tce"]
    df["能源效率_GDP_能源"] = df["gdp_亿元"] / df["energy_万tce"]

    # 非化石占比
    nf, pshare = read_non_fossil_share(df["energy_万tce"])  # 可能为空
    if not nf.empty:
        df["非化石占比"] = nf
    if not pshare.empty:
        df["电力占比"] = pshare
    return df


def yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change()


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="spearman")


# ---------- LMDI（加法，逐年累加） ----------

def _log_mean(a: float, b: float) -> float:
    if a == b:
        return a
    if a > 0 and b > 0 and a != b:
        return (a - b) / (np.log(a) - np.log(b))
    # 处理零或负值（理论上这些指标应>0），退化为均值
    return (a + b) / 2.0


def lmdi_decompose(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """对阶段 [start, end] 做 LMDI 加法分解。列依赖 df 中的 P,GDP,E,CO2 四项。"""
    sub = df.loc[start:end, ["population_万人", "gdp_亿元", "energy_万tce", "co2_万tCO2"]].dropna()
    if len(sub) < 2:
        return pd.DataFrame()
    rows = []
    for t in range(sub.index.min() + 1, sub.index.max() + 1):
        if t not in sub.index or (t - 1) not in sub.index:
            continue
        P1, P0 = sub.loc[t, "population_万人"], sub.loc[t - 1, "population_万人"]
        GDP1, GDP0 = sub.loc[t, "gdp_亿元"], sub.loc[t - 1, "gdp_亿元"]
        E1, E0 = sub.loc[t, "energy_万tce"], sub.loc[t - 1, "energy_万tce"]
        C1, C0 = sub.loc[t, "co2_万tCO2"], sub.loc[t - 1, "co2_万tCO2"]
        L = _log_mean(C1, C0)
        # 因素
        A1, A0 = GDP1 / P1, GDP0 / P0
        I_e1, I_e0 = E1 / GDP1, E0 / GDP0
        I_c1, I_c0 = C1 / E1, C0 / E0
        dP = L * np.log(P1 / P0)
        dA = L * np.log(A1 / A0)
        dIe = L * np.log(I_e1 / I_e0)
        dIc = L * np.log(I_c1 / I_c0)
        rows.append({"year": t, "人口": dP, "活动(GDP/人)": dA, "能效(E/GDP)": dIe, "碳因子(CO2/E)": dIc})
    out = pd.DataFrame(rows).set_index("year")
    out.loc["合计"] = out.sum()
    return out


# ---------- 回归（最小二乘） ----------

def _ols(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    # 添加常数项
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    yhat = X1 @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return beta, r2


def regression_loglinear(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
    sub = df.copy()
    if start is not None and end is not None:
        sub = sub.loc[start:end]
    sub = sub.dropna(subset=["co2_万tCO2", "gdp_亿元", "population_万人", "能源强度_tce_万元", "观测排放因子_tCO2_tce"])  # 关键变量
    if len(sub) < 3:
        return pd.DataFrame()
    y = np.log(sub["co2_万tCO2"].values.astype(float))
    X = np.column_stack([
        np.log(sub["gdp_亿元"].values.astype(float)),
        np.log(sub["population_万人"].values.astype(float)),
        np.log(sub["能源强度_tce_万元"].values.astype(float)),
        np.log(sub["观测排放因子_tCO2_tce"].values.astype(float)),
    ])
    beta, r2 = _ols(y, X)
    out = pd.DataFrame({
        "变量": ["常数", "ln(GDP)", "ln(人口)", "ln(能耗强度)", "ln(碳因子)"],
        "系数": beta,
    })
    out.loc[len(out)] = ["R2", r2]
    return out


def regression_growth(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
    sub = df.copy()
    if start is not None and end is not None:
        sub = sub.loc[start:end]
    g = np.log(sub).diff().dropna()
    g = g.dropna(subset=["co2_万tCO2", "gdp_亿元", "population_万人", "能源强度_tce_万元", "观测排放因子_tCO2_tce"])  # 关键变量
    if len(g) < 3:
        return pd.DataFrame()
    y = g["co2_万tCO2"].values.astype(float)
    X = g[["gdp_亿元", "population_万人", "能源强度_tce_万元", "观测排放因子_tCO2_tce"]].values.astype(float)
    beta, r2 = _ols(y, X)
    out = pd.DataFrame({
        "变量": ["常数", "Δln(GDP)", "Δln(人口)", "Δln(能耗强度)", "Δln(碳因子)"],
        "系数": beta,
    })
    out.loc[len(out)] = ["R2", r2]
    return out


# ---------- 主流程 ----------

def run():
    _ensure_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 构建主表
    master = build_master_table()
    master.index.name = "year"
    master_out = os.path.join(OUT_DIR, "主指标_年度时序.csv")
    master.to_csv(master_out, encoding="utf-8-sig")

    # 2) 增长率与相关
    growth = yoy_growth(master)
    growth_out = os.path.join(OUT_DIR, "主指标_同比增长率.csv")
    growth.to_csv(growth_out, encoding="utf-8-sig")

    level_corr = spearman_matrix(master)
    growth_corr = spearman_matrix(growth.dropna(how="all"))
    level_corr.to_csv(os.path.join(OUT_DIR, "Spearman_水平相关矩阵.csv"), encoding="utf-8-sig")
    growth_corr.to_csv(os.path.join(OUT_DIR, "Spearman_增长率相关矩阵.csv"), encoding="utf-8-sig")

    # 3) LMDI 分解（十二五/十三五）
    lmdi_12 = lmdi_decompose(master, 2011, 2015)
    lmdi_13 = lmdi_decompose(master, 2016, 2020)
    if not lmdi_12.empty:
        lmdi_12.to_csv(os.path.join(OUT_DIR, "LMDI_2011_2015.csv"), encoding="utf-8-sig")
    if not lmdi_13.empty:
        lmdi_13.to_csv(os.path.join(OUT_DIR, "LMDI_2016_2020.csv"), encoding="utf-8-sig")

    # 4) 回归模型（全样本与分阶段）
    reg1_full = regression_loglinear(master)
    reg1_12 = regression_loglinear(master, 2011, 2015)
    reg1_13 = regression_loglinear(master, 2016, 2020)
    if not reg1_full.empty:
        reg1_full.to_csv(os.path.join(OUT_DIR, "回归_对数线性_全样本.csv"), index=False, encoding="utf-8-sig")
    if not reg1_12.empty:
        reg1_12.to_csv(os.path.join(OUT_DIR, "回归_对数线性_2011_2015.csv"), index=False, encoding="utf-8-sig")
    if not reg1_13.empty:
        reg1_13.to_csv(os.path.join(OUT_DIR, "回归_对数线性_2016_2020.csv"), index=False, encoding="utf-8-sig")

    reg2_full = regression_growth(master)
    reg2_12 = regression_growth(master, 2011, 2015)
    reg2_13 = regression_growth(master, 2016, 2020)
    if not reg2_full.empty:
        reg2_full.to_csv(os.path.join(OUT_DIR, "回归_增长率_全样本.csv"), index=False, encoding="utf-8-sig")
    if not reg2_12.empty:
        reg2_12.to_csv(os.path.join(OUT_DIR, "回归_增长率_2011_2015.csv"), index=False, encoding="utf-8-sig")
    if not reg2_13.empty:
        reg2_13.to_csv(os.path.join(OUT_DIR, "回归_增长率_2016_2020.csv"), index=False, encoding="utf-8-sig")

    # 5) 参数建议（效率/非化石占比）
    params_rows = []
    def cagr(series: pd.Series, y0: int, y1: int) -> float:
        series = series.dropna()
        if y0 not in series.index or y1 not in series.index:
            return np.nan
        n = y1 - y0
        if n <= 0:
            return np.nan
        start = series.loc[y0]
        end = series.loc[y1]
        if start in [0, np.nan] or pd.isna(start) or pd.isna(end) or start == 0:
            return np.nan
        return (end / start) ** (1 / n) - 1

    eta_13 = -cagr(master["能源强度_tce_万元"], 2016, 2020)  # 效率提升=强度下降率
    sNF_13 = cagr(master.get("非化石占比", pd.Series(dtype=float)), 2016, 2020)
    params_rows.append({"参数": "效率提升率η", "阶段": "十三五", "基线(年均)": eta_13})
    params_rows.append({"参数": "非化石占比提升", "阶段": "十三五", "基线(年均)": sNF_13})
    # 情景系数（可调整倍数）
    for mul, tag in [(1.25, "中性+25%"), (1.5, "加速+50%")]:
        params_rows.append({"参数": "效率提升率η", "阶段": tag, "基线(年均)": eta_13 * mul if pd.notna(eta_13) else np.nan})
        params_rows.append({"参数": "非化石占比提升", "阶段": tag, "基线(年均)": sNF_13 * mul if pd.notna(sNF_13) else np.nan})
    pd.DataFrame(params_rows).to_csv(os.path.join(OUT_DIR, "预测参数_建议.csv"), index=False, encoding="utf-8-sig")

    print("已输出关系模型分析到:", OUT_DIR)


if __name__ == "__main__":
    run()
