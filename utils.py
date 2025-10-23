import pandas as pd, numpy as np, re

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def to_metric_unit_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a generic Nsight CSV table, extract a 3-column DataFrame with columns
    ['metric','unit','value'] by detecting header names like 'Metric Name',
    'Metric Unit', 'Metric Value' (case-insensitive, space/punct agnostic).
    Returns empty DataFrame if required columns not found.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    mcol = norm_map.get('metricname') or norm_map.get('name') or norm_map.get('metric')
    ucol = norm_map.get('metricunit') or norm_map.get('unit')
    vcol = norm_map.get('metricvalue') or norm_map.get('value')
    # Avoid confusing with 'kernelname'
    if mcol and _norm(mcol) == 'name' and any(_norm(c) == 'kernelname' for c in cols):
        mcol = norm_map.get('metricname') or mcol
    if not mcol or not vcol:
        return pd.DataFrame()
    out = pd.DataFrame({
        'metric': df[mcol].astype(str),
        'unit': df[ucol].astype(str) if ucol in df.columns else '',
        'value': df[vcol],
    })
    return out

def extract_rule_value(df: pd.DataFrame, rule_name: str):
    """
    From a metrics-id CSV DataFrame, extract the numeric 'Estimated Speedup' for a given 'Rule Name'.
    Returns np.nan if not found.
    """
    import numpy as np
    if df is None or df.empty:
        return np.nan
    # Normalize column names
    cols = {_norm(c): c for c in df.columns}
    rname = cols.get('rulename')
    rval  = cols.get('estimatedspeedup')
    if not rname or not rval:
        return np.nan
    sub = df[df[rname].astype(str).str.fullmatch(rule_name, case=False)]
    if sub.empty:
        sub = df[df[rname].astype(str).str.contains(rule_name, case=False, regex=False)]
    if sub.empty:
        return np.nan
    try:
        return pd.to_numeric(str(sub.iloc[0][rval]), errors='coerce')
    except Exception:
        return np.nan

def pick_value(df, candidates):
    """
    Pick first metric matched by regex or substring from a standard 3-col CSV: metric, unit, value.
    """
    if df is None or df.empty:
        return (np.nan, "", "")
    cols = [str(c).lower() for c in df.columns]
    mcol = 'metric' if 'metric' in cols else df.columns[0]
    ucol = 'unit' if 'unit' in cols else (df.columns[1] if len(df.columns) > 1 else None)
    vcol = 'value' if 'value' in cols else (df.columns[2] if len(df.columns) > 2 else None)
    series = df[mcol].astype(str)
    for cand in candidates:
        try:
            mask = series.str.fullmatch(cand) | series.str.contains(cand, regex=True)
            sub = df.loc[mask]
            if not sub.empty:
                val = pd.to_numeric(str(sub.iloc[0][vcol]).replace('%',''), errors='coerce') if vcol else np.nan
                unit = str(sub.iloc[0][ucol]) if ucol else ""
                return (val, unit, str(sub.iloc[0][mcol]))
        except Exception:
            continue
    return (np.nan, "", "")
