import pandas as pd, numpy as np, re

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def pick_value(df, candidates):
    """
    Pick first metric matched by regex or substring from a standard 3-col CSV: metric, unit, value.
    """
    if df is None or df.empty:
        return (np.nan, "", "")
    cols = [c.lower() for c in df.columns]
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
