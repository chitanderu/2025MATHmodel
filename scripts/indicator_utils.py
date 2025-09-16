import re
import math
from typing import List, Dict, Optional, Any, Iterable


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    # unify spaces and punctuation
    s = s.strip()
    # remove common noise tokens
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s


def match_first(columns: Iterable[str], keywords: List[str]) -> Optional[str]:
    cols = [normalize_text(c) for c in columns]
    for kw in keywords:
        kw_norm = normalize_text(kw)
        for c in cols:
            if kw_norm and kw_norm in c:
                # return the original column matching this normalized one
                idx = cols.index(c)
                return list(columns)[idx]
    return None


def find_columns(columns: Iterable[str], patterns: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for key, kws in patterns.items():
        result[key] = match_first(columns, kws)
    return result


def coerce_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none", "-"}:
        return None
    # remove thousands separators and spaces
    s = s.replace(",", "").replace(" ", "")
    # detect Chinese unit suffix like 万、亿 before any explicit unit
    mult = 1.0
    if s.endswith("亿"):
        mult *= 1e8
        s = s[:-1]
    elif s.endswith("万"):
        mult *= 1e4
        s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        # extract first numeric segment (e.g., '123.4万吨' -> 123.4)
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if not m:
            return None
        try:
            return float(m.group(0)) * mult
        except Exception:
            return None


def apply_unit_multiplier(value: Optional[float], unit: Optional[str], multipliers: Dict[str, float]) -> Optional[float]:
    if value is None:
        return None
    if not unit:
        return value
    u = normalize_text(unit)
    return value * multipliers.get(u, 1.0)


def pct_share(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom in (None, 0):
        return None
    return numer / denom


def pct_change(series: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    prev: Optional[float] = None
    for i, v in enumerate(series):
        if i == 0:
            prev = v
            continue
        if v is None or prev in (None, 0):
            out.append(None)
        else:
            out.append((v / prev) - 1.0)
        prev = v
    return out

