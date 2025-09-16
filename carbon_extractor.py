import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# 源文件（放在与脚本同级目录）
FILE_PATH = "数据_区域双碳目标与路径规划研究（含拆分数据表）.xlsx"

# 输出目录：保持与其他脚本一致的结构
OUTPUT_DIR = os.path.join("输出", "碳排放")


# 目标四个部分（标准名）
TARGET_TOPICS = [
    "碳排放量",
    "能源消费部门碳排放因子",
    "能源供应部门碳排放因子",
    "外地调入电力碳排放因子",
]

# 主题别名映射（将别名归一到标准名）
TOPIC_ALIASES: Dict[str, str] = {}
for alias in ["碳排放量", "CO2排放量", "二氧化碳排放量", "总碳排放量", "碳排放"]:
    TOPIC_ALIASES[alias.lower()] = "碳排放量"
for alias in [
    "能源消费部门碳排放因子",
    "消费部门碳排放因子",
    "终端消费部门碳排放因子",
    "终端部门碳排放因子",
    "终端碳排放因子",
    "消费部门因子",
]:
    TOPIC_ALIASES[alias.lower()] = "能源消费部门碳排放因子"
for alias in [
    "能源供应部门碳排放因子",
    "供应部门碳排放因子",
    "能源生产部门碳排放因子",
    "生产部门碳排放因子",
]:
    TOPIC_ALIASES[alias.lower()] = "能源供应部门碳排放因子"
for alias in [
    "外地调入电力碳排放因子",
    "外来电力碳排放因子",
    "外购电碳排放因子",
    "外送电碳排放因子",
    "外部电力碳排放因子",
]:
    TOPIC_ALIASES[alias.lower()] = "外地调入电力碳排放因子"


COLUMN_KEYS: Dict[str, List[str]] = {
    "topic": ["主题", "类别", "分类"],
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


def _cleanup_labels(s: str) -> str:
    s = _normalize_text(s)
    # 移除脚注上标（1、2、3 或带逗号的上标；罗马或圈号数字）
    s = re.sub(r"[①-⑳\^\d]+(?:,[\d]+)*$", "", s)
    # 去掉结尾的注释括号，如 “(万tCO2)” 不在标签层面处理
    return s


def _to_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def _find_header_row(df0: pd.DataFrame) -> int:
    # 在前 30 行里寻找包含主题/项目等的表头
    max_rows = min(30, len(df0))
    expect = set(sum(COLUMN_KEYS.values(), []))
    for i in range(max_rows):
        row = df0.iloc[i].astype(str).fillna("")
        texts = [_normalize_text(x) for x in row]
        hit = sum(any(k in t for k in expect) for t in texts)
        if hit >= 2 and any("主题" in t or "项目" in t for t in texts):
            return i
    return 0


def _read_sheet_with_header(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    preview = xl.parse(sheet, header=None, nrows=30, dtype=object)
    h = _find_header_row(preview)
    df = xl.parse(sheet, header=h, dtype=object)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # 标准化列名
    df.columns = [_normalize_text(c) for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for c in df.columns:
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
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return pd.NA


def _ensure_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    col_topic = _find_column(df, COLUMN_KEYS["topic"]) or "主题"
    col_item = _find_column(df, COLUMN_KEYS["item"]) or "项目"
    col_subitem = _find_column(df, COLUMN_KEYS["subitem"]) or "子项"
    col_unit = _find_column(df, COLUMN_KEYS["unit"]) or "单位"
    col_detail = _find_column(df, COLUMN_KEYS["detail"]) or "细分项"

    for c in [col_topic, col_item, col_subitem, col_unit, col_detail]:
        if c not in df.columns:
            df[c] = pd.NA

    # 年份列统一
    year_cols = [c for c in df.columns if _year_like(str(c))]
    year_map = {c: _extract_year(str(c)) for c in year_cols}
    rename_map = {c: str(y) for c, y in year_map.items() if y is not None}
    if rename_map:
        df = df.rename(columns=rename_map)
    year_cols = sorted({str(_extract_year(c)) for c in df.columns if _extract_year(str(c))}, key=int)

    return df, {
        "topic": col_topic,
        "item": col_item,
        "subitem": col_subitem,
        "unit": col_unit,
        "detail": col_detail,
    }, year_cols


def _map_topic(val: str) -> Optional[str]:
    t = _cleanup_labels(_normalize_text(val)).lower()
    # 去掉尾随“1/2/3/4”等脚注
    t = re.sub(r"[\d]+$", "", t)
    # 直接匹配
    if t in TOPIC_ALIASES:
        return TOPIC_ALIASES[t]
    # 包含式匹配（如“终端消费部门碳排放因子（含XX）”）
    for alias, canon in TOPIC_ALIASES.items():
        if alias in t:
            return canon
    return None


def normalize_carbon_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df, cols, year_cols = _ensure_columns(df)

    # 前向填充合并单元格导致的空白
    for c in [cols["topic"], cols["item"], cols["subitem"], cols["unit"], cols["detail"]]:
        if c in df.columns:
            df[c] = df[c].ffill()

    # 主题归一
    df[cols["topic"]] = df[cols["topic"]].map(lambda x: _map_topic(str(x)) if pd.notna(x) else None)
    df = df[df[cols["topic"]].isin(TARGET_TOPICS)]

    # 标签清洗
    for c in [cols["item"], cols["subitem"], cols["detail"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_cleanup_labels)

    # 数值清洗
    for yc in year_cols:
        if yc in df.columns:
            df[yc] = df[yc].map(_clean_value)

    # 丢弃全年份皆空的行
    if year_cols:
        df = df.dropna(subset=year_cols, how="all")

    # 输出统一列名与顺序
    rename_final = {
        cols["topic"]: "主题",
        cols["item"]: "项目",
        cols["subitem"]: "子项",
        cols["unit"]: "单位",
        cols["detail"]: "细分项",
    }
    df = df.rename(columns=rename_final)
    ordered = [c for c in ["主题", "项目", "子项", "单位", "细分项"] if c in df.columns]
    df = df[ordered + year_cols]
    return df.reset_index(drop=True)


def select_sections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sections: Dict[str, pd.DataFrame] = {}
    for topic in TARGET_TOPICS:
        part = df[df["主题"] == topic].copy()
        if not part.empty:
            sections[topic] = part.reset_index(drop=True)
    return sections


def find_candidate_sheets(xl: pd.ExcelFile) -> List[str]:
    names = xl.sheet_names
    preferred = [
        s for s in names if any(k in str(s) for k in ["碳", "CO2", "排放", "因子", "电力"])
    ]
    return preferred if preferred else names


def main(file_path: str = FILE_PATH, output_dir: str = OUTPUT_DIR) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    xl = pd.ExcelFile(file_path)
    print("工作表:", xl.sheet_names)

    used = []
    combined = []
    for sheet in find_candidate_sheets(xl):
        try:
            raw = _read_sheet_with_header(xl, sheet)
            norm = normalize_carbon_sheet(raw)
        except Exception as e:
            print(f"解析失败（{sheet}）: {e}")
            continue
        if not norm.empty:
            combined.append(norm)
            used.append(sheet)

    if not combined:
        print("未在工作簿中解析到目标‘碳排放’结构（主题/项目/子项/单位/细分项 + 年份列）。")
        return

    all_data = pd.concat(combined, ignore_index=True).drop_duplicates()

    # 分主题导出
    sections = select_sections(all_data)
    for topic, df in sections.items():
        out = os.path.join(output_dir, f"{topic}.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print("导出:", out, "行数:", len(df))

    # 汇总导出
    total_out = os.path.join(output_dir, "碳排放_四部分汇总.csv")
    all_data.to_csv(total_out, index=False, encoding="utf-8-sig")
    print("\n使用的工作表:", used)
    print("汇总导出:", total_out)
    print("示例预览:")
    print(all_data.head(20))


if __name__ == "__main__":
    # 运行: python carbon_extractor.py
    main()

