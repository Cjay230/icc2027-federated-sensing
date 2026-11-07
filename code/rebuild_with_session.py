# Rebuild the 60s windows carrying the journey identifier.
#
# Why: the cached windows in rebuilt_cache/*.csv have no session column, so every
# train/test split so far has been random over windows. Windows are built per
# `source_file`, and one source_file is one recording journey, so source_file IS
# the journey ID -- it was simply dropped on the way out.
#
# Features are computed by importing window_features() from rebuild_windows_fixed.py
# unchanged, so the new files must reproduce the old ones exactly, plus two columns.
# Verified at the end by comparing window counts and feature values per city.
#
# Memory: the raw CSVs total ~1.1 GB (kaust alone is 746 MB), so the header is read
# first and only the needed columns are loaded.
import os, sys, importlib.util
import numpy as np
import pandas as pd

RAW = r"C:\Users\user\Desktop\electives\VIP\kaust data\full data"
WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
SRC = os.path.join(WIN, "rebuild_windows_fixed.py")
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
WINDOW_SEC = 60
MIN_ROWS = 10

# import the original feature code so features are identical, not reimplemented
spec = importlib.util.spec_from_file_location("rwf", SRC)
rwf = importlib.util.module_from_spec(spec)
sys.modules["rwf"] = rwf
spec.loader.exec_module(rwf)
find_cols = rwf.find_cols
window_features = rwf.window_features

KEYWORDS = ['RSRP/antenna', 'E-UTRAN carrier RSSI', 'RSRQ/antenna',
            'CQI for codeword 0', 'CQI for codeword 1', 'Rank indication -', 'Velocity']
SINGLES = ['longitude', 'latitude', 'timing advance',
           'physical cell identity (lte pcell)', 'pdsch scheduled', 'pathloss']


def needed_columns(path):
    head = pd.read_csv(path, nrows=0)
    cols = set(['source_file', 'Time', 'label'])
    for kw in KEYWORDS:
        cols.update(find_cols(head, [kw]))
    for s in SINGLES:
        c = next((c for c in head.columns if s in c.lower()), None)
        if c:
            cols.add(c)
    return [c for c in head.columns if c in cols]


def build(city):
    path = os.path.join(RAW, city + ".csv")
    use = needed_columns(path)
    print("  %-8s loading %d of the file's columns" % (city, len(use)), flush=True)
    df = pd.read_csv(path, low_memory=False, usecols=use)

    # time parsing, identical to the original
    _t = df['Time'].astype(str).str.strip()
    df['time_parsed'] = pd.to_datetime(_t, format='%H:%M:%S.%f', errors='coerce')
    _still = df['time_parsed'].isna() & _t.ne('') & _t.ne('nan')
    if _still.any():
        df.loc[_still, 'time_parsed'] = pd.to_datetime(_t[_still], format='mixed', errors='coerce')
    df = df.sort_values(['source_file', 'time_parsed']).reset_index(drop=True)

    rsrp_cols = find_cols(df, ['RSRP/antenna'])
    rssi_cols = find_cols(df, ['E-UTRAN carrier RSSI'])
    rsrq_cols = find_cols(df, ['RSRQ/antenna'])
    cqi0_cols = find_cols(df, ['CQI for codeword 0'])
    cqi1_cols = find_cols(df, ['CQI for codeword 1'])
    rank_cols = find_cols(df, ['Rank indication -'])
    vel_cols = find_cols(df, ['Velocity'])
    lon_col = next((c for c in df.columns if 'longitude' in c.lower()), None)
    lat_col = next((c for c in df.columns if 'latitude' in c.lower()), None)
    ta_col = next((c for c in df.columns if 'timing advance' in c.lower()), None)
    cell_id_col = next((c for c in df.columns if 'physical cell identity (lte pcell)' in c.lower()), None)
    pdsch_col = next((c for c in df.columns if 'pdsch scheduled' in c.lower()), None)
    pathloss_col = next((c for c in df.columns if 'pathloss' in c.lower()), None)

    rows = []
    for src, grp in df.groupby('source_file'):
        grp = grp.dropna(subset=['time_parsed']).copy()
        if len(grp) < MIN_ROWS:
            continue
        _raw = grp['time_parsed']
        _sec = _raw.dt.hour * 3600 + _raw.dt.minute * 60 + _raw.dt.second + _raw.dt.microsecond / 1e6
        _sec = _sec.reset_index(drop=True)
        _day = (_sec.diff().fillna(0) < -3600).cumsum() * 86400.0
        _abs = (_sec + _day).values
        grp['t_sec'] = _abs - _abs.min()
        grp['win_id'] = (grp['t_sec'] // WINDOW_SEC).astype(int)
        for wid, win in grp.groupby('win_id'):
            if len(win) < MIN_ROWS:
                continue
            f = window_features(win, rsrp_cols, rssi_cols, rsrq_cols, cqi0_cols, cqi1_cols,
                                rank_cols, vel_cols, lon_col, lat_col, ta_col,
                                cell_id_col, pdsch_col, pathloss_col, WINDOW_SEC)
            lbl = win['label'].mode()
            f['label'] = lbl.iloc[0] if len(lbl) > 0 else 'unknown'
            f['session'] = str(src)          # <-- the journey identifier
            f['win_id'] = int(wid)
            rows.append(f)
    out = pd.DataFrame(rows)
    dest = os.path.join(WIN, city + "_60s_sess.csv")
    out.to_csv(dest, index=False)
    del df
    return out, dest


print("Rebuilding 60s windows with journey IDs")
print("=" * 70)
summary = []
for city in CITIES:
    new, dest = build(city)
    old = pd.read_csv(os.path.join(WIN, city + "_60s.csv"))
    feats = [c for c in old.columns if c != 'label']
    same_n = (len(new) == len(old))
    ident = False
    if same_n:
        a = old[feats].fillna(-999.0).round(6).values.astype(float)
        b = new[feats].fillna(-999.0).round(6).values.astype(float)
        ident = bool(np.allclose(a, b, atol=1e-5))
    summary.append((city, len(old), len(new), same_n, ident))
    print("  %-8s old %5d  new %5d  counts_match=%s  features_identical=%s  journeys=%d"
          % (city, len(old), len(new), same_n, ident, new.session.nunique()), flush=True)

print()
print("=" * 70)
print("TOTAL  old %d  new %d" % (sum(s[1] for s in summary), sum(s[2] for s in summary)))
if all(s[3] and s[4] for s in summary):
    print("ALL CITIES: window counts and feature values reproduce the existing cache exactly.")
else:
    print("WARNING: a city differs from the existing cache -- inspect before use.")

print()
print("journeys per label, per city:")
for city in CITIES:
    d = pd.read_csv(os.path.join(WIN, city + "_60s_sess.csv"))
    g = d.groupby('label').agg(windows=('session', 'size'), journeys=('session', 'nunique'))
    print(" ", city)
    for lbl, r in g.sort_values('windows', ascending=False).iterrows():
        flag = "   <-- single journey, cannot be split" if r.journeys == 1 else ""
        print("    %-12s %5d windows  %3d journeys%s" % (lbl, r.windows, r.journeys, flag))
