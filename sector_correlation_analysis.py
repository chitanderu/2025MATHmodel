import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# 使用非交互式后端，避免 Qt 依赖
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 输入文件（来自前面步骤的导出）
CARBON_PATH = os.path.join("输出", "碳排放", "碳排放量.csv")
ENERGY_PATH = os.path.join("输出", "经济能源", "能源消费量.csv")
CORE_INDICATORS = os.path.join("输出", "指标", "核心指标时序.csv")
DERIVED_INDICATORS = os.path.join("输出", "指标", "衍生指标时序.csv")

# 输出目录
OUT_DIR = os.path.join("输出", "分析", "相关性")


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


SECTOR_LABELS_ORDER = [
    "农林消费部门",
    "能源供应部门",
    "工业消费部门",
    "交通消费部门",
    "建筑消费部门",
    "居民生活消费",
]

SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "农林消费部门": ["农林", "农业", "林业", "牧业", "渔业", "第一产业"],
    "能源供应部门": ["能源供应", "能源供给", "能源供", "电力", "供热", "热力", "发电"],
    "工业消费部门": ["工业消费", "工业", "制造", "采矿"],
    "交通消费部门": ["交通消费", "交通", "运输"],
    "建筑消费部门": ["建筑消费", "建筑", "建築"],
    "居民生活消费": ["居民生活", "生活消费", "居民"],
}


def _cleanup_label_text(text) -> str:
    if text is None:
        return ""
    s = str(text).strip().replace("（", "(").replace("）", ")")
    # 去脚注上标/尾随数字（如 总量1, 其他2,3）
    s = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+(?:,[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+)*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip().replace("（", "(").replace("）", ")") for c in df.columns]
    year_cols = [c for c in df.columns if re.fullmatch(r"(19\d{2}|20\d{2})", str(c))]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    for col in ["主题", "项目", "子项", "单位", "细分项"]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].apply(_cleanup_label_text)
    for col in ["主题", "项目", "子项", "细分项"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df, sorted(year_cols, key=int)


def _match_sector(row: pd.Series, prefer_subitem: bool = True) -> Optional[str]:
    fields = {
        "项目": _cleanup_label_text(row.get("项目", "")),
        "子项": _cleanup_label_text(row.get("子项", "")),
        "细分项": _cleanup_label_text(row.get("细分项", "")),
    }
    # 优先使用子项来判定，避免“第一/第二/第三产业 总量”被误映射
    search_order = ["子项", "细分项", "项目"] if prefer_subitem else ["项目", "子项", "细分项"]
    for key in SECTOR_LABELS_ORDER:
        kws = SECTOR_KEYWORDS[key]
        for fld in search_order:
            text = fields[fld]
            if any(k in text for k in kws):
                return key
    return None


def extract_sector_timeseries_from_carbon() -> Tuple[pd.DataFrame, List[int]]:
    if not os.path.exists(CARBON_PATH):
        raise FileNotFoundError(CARBON_PATH)
    df = pd.read_csv(CARBON_PATH, dtype=object)
    df, ycols = _normalize_df(df)
    part = df[df["主题"].str.contains("碳排放", na=False)].copy()
    if part.empty:
        raise ValueError("碳排放量.csv 中未找到‘主题=碳排放量’记录")
    # 只保留总量单位（万tCO2/万吨CO2/MtCO2），排除因子行
    unit = part["单位"].astype(str).str.lower()
    mask_total = unit.str.contains("万tco2") | unit.str.contains("万吨co2") | unit.str.contains("mtco2") | unit.str.fullmatch("tco2")
    if mask_total.any():
        part = part[mask_total]

    # 若存在“子项=总量”，这是部门合计（如 第一产业-总量），不直接映射部门；仅用于总量核对
    # 真正的部门数据在子项/细分项中
    # 部门聚合：对匹配到的部门按年合并
    rows = []
    for i, r in part.iterrows():
        sec = _match_sector(r, prefer_subitem=True)
        if not sec:
            continue
        s = pd.to_numeric(r[ycols], errors="coerce")
        rows.append((sec, s))
    if not rows:
        raise ValueError("未能在碳排放量.csv 中从‘子项/细分项’识别到六类部门，请检查列命名。")

    agg: Dict[str, pd.Series] = {}
    for sec, s in rows:
        agg[sec] = (agg[sec] + s) if sec in agg else s
    data = pd.DataFrame(agg)
    data.index = data.index.astype(int)
    data = data.sort_index()
    # 只保留定义顺序中的列
    cols = [c for c in SECTOR_LABELS_ORDER if c in data.columns]
    data = data[cols]
    years = sorted(data.index.tolist())
    return data, years


def extract_sector_timeseries_from_energy() -> Tuple[pd.DataFrame, List[int]]:
    if not os.path.exists(ENERGY_PATH):
        raise FileNotFoundError(ENERGY_PATH)
    df = pd.read_csv(ENERGY_PATH, dtype=object)
    df, ycols = _normalize_df(df)
    part = df[df["主题"].str.contains("能源消费量", na=False)].copy()
    if part.empty:
        raise ValueError("能源消费量.csv 中未找到‘主题=能源消费量’记录")

    rows = []
    for i, r in part.iterrows():
        sec = _match_sector(r, prefer_subitem=True)
        if not sec:
            continue
        s = pd.to_numeric(r[ycols], errors="coerce")
        rows.append((sec, s))

    if not rows:
        raise ValueError("未能在能源消费量.csv 中从‘子项/细分项’识别到六类部门，请检查列命名。")

    agg: Dict[str, pd.Series] = {}
    for sec, s in rows:
        agg[sec] = (agg[sec] + s) if sec in agg else s
    data = pd.DataFrame(agg)
    data.index = data.index.astype(int)
    data = data.sort_index()
    cols = [c for c in SECTOR_LABELS_ORDER if c in data.columns]
    data = data[cols]
    years = sorted(data.index.tolist())
    return data, years


def _align_two(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = sorted(set(a.index) & set(b.index))
    return a.reindex(idx), b.reindex(idx)


def corr_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change()  # 第一行为空


def save_csv(df: pd.DataFrame, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, name), encoding="utf-8-sig")


def plot_heatmap(mat: pd.DataFrame, title: str, filename: str, cmap: str = "RdBu_r", vmin: float = -1.0, vmax: float = 1.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    _ensure_font()
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相关系数")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)


def plot_scatter_pairs(co2: pd.DataFrame, energy: pd.DataFrame):
    _ensure_font()
    # 六个部门的 排放 vs 能耗 散点 + 拟合线
    os.makedirs(OUT_DIR, exist_ok=True)
    co2, energy = _align_two(co2, energy)
    cols = [c for c in SECTOR_LABELS_ORDER if c in co2.columns and c in energy.columns]
    if not cols:
        return
    n = len(cols)
    rows = 2
    cols_per_row = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(13, 7), dpi=150)
    axes = np.array(axes).reshape(rows, cols_per_row)
    for idx, sec in enumerate(cols):
        r = idx // cols_per_row
        c = idx % cols_per_row
        ax = axes[r, c]
        x = energy[sec]
        y = co2[sec]
        mask = x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], s=22, alpha=0.8, color="#4C78A8")
        # 线性拟合
        if mask.sum() >= 2:
            coef = np.polyfit(x[mask], y[mask], 1)
            xv = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xv, coef[0] * xv + coef[1], color="#C0504D")
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.set_title(f"{sec} (r={r:.2f})", fontsize=10)
        else:
            ax.set_title(f"{sec}", fontsize=10)
        ax.set_xlabel("能源消费量（万tce）", fontsize=8)
        ax.set_ylabel("碳排放量（万tCO$_2$）", fontsize=8)
    # 清理多余子图
    for k in range(n, rows * cols_per_row):
        fig.delaxes(axes.flatten()[k])
    fig.suptitle("部门碳排放 vs 能源消费（散点与拟合）")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "部门_排放_vs_能耗_散点.png"))
    plt.close(fig)


def merge_core_and_derived() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    core = None
    derived = None
    if os.path.exists(CORE_INDICATORS):
        core = pd.read_csv(CORE_INDICATORS)
    if os.path.exists(DERIVED_INDICATORS):
        derived = pd.read_csv(DERIVED_INDICATORS)
    return core, derived


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    _ensure_font()

    _ensure_font()
    # 1) 提取部门排放与能耗时序
    co2, years1 = extract_sector_timeseries_from_carbon()
    energy, years2 = extract_sector_timeseries_from_energy()
    co2, energy = _align_two(co2, energy)

    # 导出原始时序
    save_csv(co2, "部门_碳排放_时序.csv")
    save_csv(energy, "部门_能耗_时序.csv")

    # 2) 计算增长率（同比）
    co2_g = growth_rate(co2)
    energy_g = growth_rate(energy)
    save_csv(co2_g, "部门_碳排放_同比增长率.csv")
    save_csv(energy_g, "部门_能耗_同比增长率.csv")

    # 3) 相关性矩阵（水平值 + 增长率）
    pearson_level = corr_matrix(co2, method="pearson")
    spearman_level = corr_matrix(co2, method="spearman")
    pearson_growth = corr_matrix(co2_g.dropna(how="all"), method="pearson")
    spearman_growth = corr_matrix(co2_g.dropna(how="all"), method="spearman")
    save_csv(pearson_level, "相关矩阵_碳排放_水平_Pearson.csv")
    save_csv(spearman_level, "相关矩阵_碳排放_水平_Spearman.csv")
    save_csv(pearson_growth, "相关矩阵_碳排放_增长率_Pearson.csv")
    save_csv(spearman_growth, "相关矩阵_碳排放_增长率_Spearman.csv")
    plot_heatmap(pearson_level, "部门碳排放相关性（Pearson，水平）", "heatmap_排放_Pearson_水平.png")
    plot_heatmap(pearson_growth, "部门碳排放相关性（Pearson，增长率）", "heatmap_排放_Pearson_增长率.png")

    # 4) 排放与能耗的相关性（同部门）
    corr_rows = []
    for sec in SECTOR_LABELS_ORDER:
        if sec in co2.columns and sec in energy.columns:
            s1 = co2[sec]
            s2 = energy[sec]
            mask = s1.notna() & s2.notna()
            if mask.sum() >= 2:
                rp = s1[mask].corr(s2[mask], method="pearson")
                rs = s1[mask].corr(s2[mask], method="spearman")
            else:
                rp, rs = np.nan, np.nan
            # 增长率相关
            g1 = s1.pct_change()
            g2 = s2.pct_change()
            maskg = g1.notna() & g2.notna()
            if maskg.sum() >= 2:
                rp_g = g1[maskg].corr(g2[maskg], method="pearson")
                rs_g = g1[maskg].corr(g2[maskg], method="spearman")
            else:
                rp_g, rs_g = np.nan, np.nan
            corr_rows.append({
                "sector": sec,
                "pearson_level": rp,
                "spearman_level": rs,
                "pearson_growth": rp_g,
                "spearman_growth": rs_g,
            })
    corr_df = pd.DataFrame(corr_rows)
    save_csv(corr_df, "相关性_同部门_排放_vs能耗.csv")

    # 5) 绘制散点矩阵（排放 vs 能耗）
    plot_scatter_pairs(co2, energy)

    # 6) 与宏观指标的关系（若存在）
    core, derived = merge_core_and_derived()
    if core is not None:
        try:
            gdp = core[core["indicator_key"] == "gdp"].copy()
            gdp = gdp.set_index(gdp["year"].astype(int))["value"].astype(float)
            # 取总排放
            co2_total = co2.sum(axis=1)
            idx = sorted(set(gdp.index) & set(co2_total.index))
            r_pg = gdp.reindex(idx).corr(co2_total.reindex(idx))
            pd.DataFrame({"metric": ["GDP vs 总碳排放"], "pearson": [r_pg]}).to_csv(
                os.path.join(OUT_DIR, "相关性_GDP_vs_总碳排放.csv"), index=False, encoding="utf-8-sig"
            )
        except Exception:
            pass

    print("相关性分析文件已输出到:", OUT_DIR)


if __name__ == "__main__":
    run()
