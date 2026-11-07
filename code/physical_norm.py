# Does normalising handover features by the site's cell diameter improve cross-site transfer?
#
# The physics. Handover rate is not a motion feature. It is motion contaminated by site
# geometry: H = v/D + e, where D is the cell diameter and e is a fluctuation term that
# dominates for slow movers. Measured per city from vehicular windows:
#   Jeddah 180 m, KAUST 133 m, KZ 116 m, Mekkah 121 m  (a 1.55x spread)
# So two cities with identical traffic report different handover rates purely because
# their cells differ in size. A model trained on three cities reads the fourth on the
# wrong scale.
#
# The transform. Multiply handover-derived features by the site's own D:
#   handover_rate [events/s] * D [m]  ->  m/s, a speed proxy comparable across sites
#   unique_cells  [count]    * D [m]  ->  m, a distance proxy
#
# Protocol note on D. D is estimated inside each city from that city's own windows and
# only the single scalar is used. For a held-out city this is the proposed deployment
# protocol: the site computes its own D locally and declares one number. A real operator
# would read D from network planning data without needing velocity at all; estimating it
# from local velocity here is a stand-in for that lookup. No per-window data crosses a
# boundary in either case.
#
# Prediction being tested. If the mechanism is real, cross-city (LOCO) accuracy should
# rise while within-city accuracy stays flat or dips -- the same signature the position
# result showed. A rise in all three would more likely mean the transform is just adding
# scale information, not removing a site-specific nuisance.
#
# Random forest is used as a fast screen: it is more sensitive to feature scaling than a
# standardised MLP, so if there is no effect here there will be none federated.
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
OUT = r"C:\Users\user\Desktop\VIP 2\rebuilt_results"
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
LM = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3, 'bike': 4, 'motorcycle': 5, 'jog': 6, 'train': 7}
FAM = {'walk': 0, 'jog': 0, 'bike': 1, 'scooter': 1,
       'car': 2, 'bus': 2, 'motorcycle': 2, 'train': 3}
VEH = {'car', 'bus', 'motorcycle'}
GPS = ['longitude_mean', 'latitude_mean']
SEEDS = (42, 123, 456)
TEST_FRAC = 0.2
HO_FEATS = ['handover_rate', 'unique_cells']


def load():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv")); t['city'] = c; d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(LM)].reset_index(drop=True)
    D['y8'] = D.label.map(LM).astype(int)
    D['y4'] = D.label.map(FAM).astype(int)
    D['sid'] = D.city + "::" + D.session.astype(str)
    return D


def cell_diameters(D):
    """One scalar per city, from that city's own vehicular windows. D = v / (3.6 H), metres."""
    out = {}
    for c in CITIES:
        s = D[(D.city == c) & D.label.isin(VEH) &
              (D.velocity_mean > 0) & (D.handover_rate > 0)]
        out[c] = float(np.median(s.velocity_mean / (3.6 * s.handover_rate))) if len(s) >= 10 else np.nan
    return out


def apply_norm(D, Dm):
    E = D.copy()
    scale = E.city.map(Dm).to_numpy()
    for f in HO_FEATS:
        E[f] = E[f].to_numpy() * scale
    return E


def grouped_mask(D, seed):
    rng = np.random.RandomState(seed); test = set()
    for (city, lab), g in D.groupby(['city', 'label']):
        s = sorted(g.sid.unique())
        if len(s) < 2:
            continue
        rng.shuffle(s)
        k = min(max(1, int(round(TEST_FRAC * len(s)))), len(s) - 1)
        test.update(s[:k])
    return D.sid.isin(test).values


def evaluate(D, feats, ycol, labs):
    X = D[feats].fillna(0).to_numpy(dtype=np.float64)
    y = D[ycol].to_numpy()
    rnd, grp = [], []
    for sd in SEEDS:
        m = np.random.RandomState(sd).rand(len(D)) < TEST_FRAC
        rf = RandomForestClassifier(n_estimators=200, random_state=sd, n_jobs=2).fit(X[~m], y[~m])
        rnd.append(f1_score(y[m], rf.predict(X[m]), average='macro', labels=labs, zero_division=0))
        gm = grouped_mask(D, sd)
        rf2 = RandomForestClassifier(n_estimators=200, random_state=sd, n_jobs=2).fit(X[~gm], y[~gm])
        grp.append(f1_score(y[gm], rf2.predict(X[gm]), average='macro', labels=labs, zero_division=0))
    loco = []
    for ho in CITIES:
        tr = (D.city != ho).to_numpy()
        rf3 = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(X[tr], y[tr])
        loco.append(f1_score(y[~tr], rf3.predict(X[~tr]), average='macro', labels=labs, zero_division=0))
    return float(np.mean(rnd)), float(np.mean(grp)), float(np.mean(loco)), loco


if __name__ == "__main__":
    D = load()
    Dm = cell_diameters(D)
    print("implied cell diameter per city (metres), from that city's own vehicular windows:")
    for c in CITIES:
        print("   %-8s %.0f m" % (c, Dm[c]))
    print("   spread: %.2fx" % (max(Dm.values()) / min(Dm.values())))
    E = apply_norm(D, Dm)
    feats = [c for c in D.columns
             if c not in ('label', 'city', 'session', 'win_id', 'y8', 'y4', 'sid')]
    POL = {'A all features': feats, 'B no position': [c for c in feats if c not in GPS]}

    rows = []
    for tname, DF in [('raw', D), ('D-normalised', E)]:
        print("\n" + "=" * 92)
        print("TRANSFORM: %s" % tname)
        print("=" * 92, flush=True)
        for tgt, ycol, labs in [('8 modes', 'y8', list(range(8))),
                                ('4 families', 'y4', list(range(4)))]:
            for pname, fs in POL.items():
                r, g, l, lp = evaluate(DF, fs, ycol, labs)
                rows.append(dict(target=tgt, transform=tname, policy=pname,
                                 random=r, grouped=g, loco=l,
                                 **{('loco_' + c): v for c, v in zip(CITIES, lp)}))
                print("  %-11s %-16s random %.3f | grouped %.3f | LOCO %.3f  [%s]"
                      % (tgt, pname, r, g, l,
                         " ".join("%s %.2f" % (c[:4], v) for c, v in zip(CITIES, lp))),
                      flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "physical_norm.csv"), index=False)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("VERDICT: effect of D-normalisation")
    print("=" * 92)
    for tgt in ['8 modes', '4 families']:
        for pname in POL:
            a = R[(R.target == tgt) & (R.policy == pname) & (R.transform == 'raw')].iloc[0]
            b = R[(R.target == tgt) & (R.policy == pname) & (R.transform == 'D-normalised')].iloc[0]
            print("  %-11s %-16s  LOCO %+.4f   grouped %+.4f   random %+.4f"
                  % (tgt, pname, b.loco - a.loco, b.grouped - a.grouped, b.random - a.random))
    print()
    print("  The mechanism predicts LOCO up, grouped and random flat or down.")
    print("  All three rising would suggest added scale information, not a removed nuisance.")
    print("\nsaved physical_norm.csv")
