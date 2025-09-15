import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


FILE_PATH = "数据_区域双碳目标与路径规划研究（含拆分数据表）.xlsx"
OUTPUT_DIR = os.path.join("输出", "经济能源")


# 目标四个部分（按“主题”列筛选）
TARGET_TOPICS = [
    "生产总值",
    "能源消费量",
    "产业能耗结构",
    "能耗品种结构",
]

# 关键列名关键词（用于柔性匹配不同表头写法）
COLUMN_KEYS: Dict[str, List[str]] = {
    "topic": ["主题"],
    "item": ["项目", "指标"],
    "subitem": ["子项", "分项", "行业", "部门"],
    "unit": ["单位", "计量单位"],
    "detail": ["细分项", "细项", "细分", "用途", "环节"],
}


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def _find_header_row(df0: pd.DataFrame) -> int:
    """在前 30 行中寻找包含“主题/项目”等关键字的表头行索引。"""
    max_rows = min(30, len(df0))
    expect = set(sum(COLUMN_KEYS.values(), []))  # 扁平化关键词
    for i in range(max_rows):
        row = df0.iloc[i].astype(str).fillna("")
        texts = [_normalize_text(x) for x in row]
        hit = sum(any(k in t for k in expect) for t in texts)
        if hit >= 2 and any("主题" in t for t in texts):
            return i
    # 兜底：返回第 0 行
    return 0


def _read_sheet_with_header(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    # 预读找表头
    preview = xl.parse(sheet, header=None, nrows=30, dtype=object)
    h = _find_header_row(preview)
    df = xl.parse(sheet, header=h, dtype=object)
    # 丢弃全空行/列
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # 标准化列名
    cols = []
    for c in df.columns:
        c = _normalize_text(c)
        c = re.sub(r"Unnamed: \d+_level_\d+", "", c)
        cols.append(c)
    df.columns = cols
    return df


def _find_column(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = list(df.columns)
    # 直接包含
    for c in cols:
        t = _normalize_text(c)
        if any(k == t for k in keys) or any(k in t for k in keys):
            return c
    return None


def _year_like(c: str) -> bool:
    c = _normalize_text(c)
    return bool(re.fullmatch(r"(19\d{2}|20\d{2})(?:年)?", c))


def _extract_year(c: str) -> Optional[int]:
    m = re.search(r"(19\d{2}|20\d{2})", _normalize_text(c))
    return int(m.group(1)) if m else None


def _clean_value(x):
    s = _to_str(x).strip()
    if s in {"-", "—", "— —", "", "nan", "None"}:
        return pd.NA
    # 去除千分位
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return pd.NA


def _cleanup_labels(s: str) -> str:
    # 去脚注上标，如 “总量¹” -> “总量” ； “其他转换²,³” -> “其他转换”
    s = _normalize_text(s)
    s = re.sub(r"[\d]+(?:,[\d]+)*$", "", s)
    s = re.sub(r"[①-⑳\^\d]+$", "", s)
    return s


def _ensure_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    # 定位关键列
    col_topic = _find_column(df, COLUMN_KEYS["topic"]) or "主题"
    col_item = _find_column(df, COLUMN_KEYS["item"]) or "项目"
    col_subitem = _find_column(df, COLUMN_KEYS["subitem"]) or "子项"
    col_unit = _find_column(df, COLUMN_KEYS["unit"]) or "单位"
    col_detail = _find_column(df, COLUMN_KEYS["detail"]) or "细分项"

    # 若缺列则新增空列，避免后续错误
    for c in [col_topic, col_item, col_subitem, col_unit, col_detail]:
        if c not in df.columns:
            df[c] = pd.NA

    # 年份列
    year_cols = [c for c in df.columns if _year_like(str(c))]
    # 有些列名形如 “2010 年”，做一次精确年份映射
    year_map = {c: _extract_year(str(c)) for c in year_cols}
    # 重命名为纯数字年，避免重复
    rename_map = {}
    for c, y in year_map.items():
        if y is not None:
            rename_map[c] = str(y)
    df = df.rename(columns=rename_map)
    year_cols = sorted({str(_extract_year(c)) for c in df.columns if _extract_year(str(c))}, key=int)

    cols_meta = {
        "topic": col_topic,
        "item": col_item,
        "subitem": col_subitem,
        "unit": col_unit,
        "detail": col_detail,
    }
    return df, cols_meta, year_cols


def normalize_econ_energy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """将一张“经济与能源”类表标准化为统一宽表结构。"""
    df = df.copy()

    # 关键列与年份列
    df, cols, year_cols = _ensure_columns(df)

    # 前向填充合并单元格导致的空白
    for c in [cols["topic"], cols["item"], cols["subitem"], cols["unit" ]] + ([cols["detail"]] if cols["detail"] in df.columns else []):
        if c in df.columns:
            df[c] = df[c].ffill()

    # 清洗标签
    for c in [cols["item"], cols["subitem"], cols["detail"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_cleanup_labels)

    # 仅保留四大主题
    df = df[df[cols["topic"]].astype(str).isin(TARGET_TOPICS)]

    # 数值清洗
    for yc in year_cols:
        if yc in df.columns:
            df[yc] = df[yc].map(_clean_value)

    # 输出统一列序
    ordered_cols = [cols["topic"], cols["item"], cols["subitem"], cols["unit"], cols["detail"]] + year_cols
    # 去重，仅保留存在的列
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols].copy()

    # 规范列名为标准中文
    rename_final = {
        cols["topic"]: "主题",
        cols["item"]: "项目",
        cols["subitem"]: "子项",
        cols["unit"]: "单位",
        cols["detail"]: "细分项",
    }
    df = df.rename(columns=rename_final)
    # 列顺序（标准列 + 年份）
    fixed = [c for c in ["主题", "项目", "子项", "单位", "细分项"] if c in df.columns]
    year_cols = [c for c in df.columns if re.fullmatch(r"(19\d{2}|20\d{2})", str(c))]
    df = df[fixed + year_cols]

    # 丢弃全空年份行
    if year_cols:
        df = df.dropna(subset=year_cols, how="all")

    return df.reset_index(drop=True)


def select_sections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """按主题拆分四个部分。"""
    sections: Dict[str, pd.DataFrame] = {}
    for topic in TARGET_TOPICS:
        part = df[df["主题"] == topic].copy()
        if not part.empty:
            sections[topic] = part.reset_index(drop=True)
    return sections


def find_candidate_sheets(xl: pd.ExcelFile) -> List[str]:
    """优先选择名字里包含“经济/能源/能耗/结构”的工作表。若没命中，返回全部。"""
    sn = xl.sheet_names
    preferred = [
        s for s in sn if any(k in str(s) for k in ["经济", "能源", "能耗", "结构", "经济与能源", "经济能源"])
    ]
    return preferred if preferred else sn


def main(file_path: str = FILE_PATH, output_dir: str = OUTPUT_DIR) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    xl = pd.ExcelFile(file_path)
    print("工作表:", xl.sheet_names)

    combined: List[pd.DataFrame] = []
    used_sheets: List[str] = []

    for sheet in find_candidate_sheets(xl):
        try:
            df_raw = _read_sheet_with_header(xl, sheet)
            df_norm = normalize_econ_energy_sheet(df_raw)
        except Exception as e:
            print(f"解析失败（{sheet}）: {e}")
            continue

        if not df_norm.empty:
            combined.append(df_norm)
            used_sheets.append(sheet)

    if not combined:
        print("未在任何工作表中解析到目标结构（主题/项目/子项/单位/细分项 + 年份列）。")
        return

    all_data = pd.concat(combined, ignore_index=True)
    # 如果多表重复，按行去重
    all_data = all_data.drop_duplicates()

    sections = select_sections(all_data)

    # 导出分表
    for topic, df in sections.items():
        out = os.path.join(output_dir, f"{topic}.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print("导出:", out, "行数:", len(df))

    # 总表
    total_out = os.path.join(output_dir, "经济与能源_四部分汇总.csv")
    all_data.to_csv(total_out, index=False, encoding="utf-8-sig")
    print("\n使用的工作表:", used_sheets)
    print("汇总导出:", total_out)
    print("示例预览:")
    print(all_data.head(20))
    


if __name__ == "__main__":
    # 运行: python econ_energy_extractor.py
    main()

