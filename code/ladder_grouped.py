# The disclosure ladder on JOURNEY-GROUPED splits, and its federated version.
#
# Two things are fixed relative to earlier runs:
#   1. No journey spans the train/test boundary. Previously the split was random over
#      windows, so windows of one trip sat on both sides.
#   2. The ladder is run with the actual federated methods, not only a random forest.
#
# Split rule (per city, per class):
#   a class with >= 2 journeys sends ~20% of its journeys (at least one, and never all)
#   to test; a class with exactly 1 journey sends that journey to TRAIN, so the client
#   still observes the class and the federated structure is unchanged -- it simply has
#   no test support for it in that city. KZ/walk is the only such case.
#
# Policies: A = all features; B = no coordinates; C = radio only; D = radio minus the
# campaign-distinguishing fields.
#
# Stage 1 (--stage rf):  random forest, three protocols, cheap.
# Stage 2 (--stage fed): FedAvg and FedObserve, policies A and B, grouped + leave-one-city-out.
import os, copy, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
OUT = r"C:\Users\user\Desktop\VIP 2\rebuilt_results"
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
LM = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3, 'bike': 4, 'motorcycle': 5, 'jog': 6, 'train': 7}
NC = 8; ALLC = list(range(NC))
FAM = {'walk': 0, 'jog': 0, 'bike': 1, 'scooter': 1, 'car': 2, 'bus': 2, 'motorcycle': 2, 'train': 3}
NFAM = 4
BS, LR, MOM = 32, 0.01, 0.9
EPOCHS, ROUNDS = 5, 30
SEEDS = (42, 123, 456)
TEST_FRAC = 0.2

GPS = ['longitude_mean', 'latitude_mean']
KIN = ['velocity_mean', 'velocity_max', 'velocity_var']
CAMPAIGN = ['timing_advance_mean', 'pathloss_mean', 'pathloss_max', 'pathloss_var',
            'rsrq_mean', 'rsrq_var', 'rsrq_max', 'rsrq_min', 'pdsch_mean']


def load():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv"))
        t['city'] = c
        d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(LM)].reset_index(drop=True)
    D['y'] = D.label.map(LM).astype(int)
    D['fam'] = D.label.map(FAM).astype(int)
    D['sid'] = D.city + "::" + D.session.astype(str)
    return D


def grouped_split(D, seed):
    """Boolean test mask. No journey appears on both sides."""
    rng = np.random.RandomState(seed)
    test_sids = set(); singles = []
    for (city, lab), g in D.groupby(['city', 'label']):
        sids = sorted(g.sid.unique())
        if len(sids) < 2:
            singles.append((city, lab, sids[0]))
            continue
        rng.shuffle(sids)
        k = max(1, int(round(TEST_FRAC * len(sids))))
        k = min(k, len(sids) - 1)
        test_sids.update(sids[:k])
    return D.sid.isin(test_sids).values, singles


def build_policies(feats):
    radio = [c for c in feats if c not in GPS + KIN]
    return {
        'A all features': feats,
        'B no position': [c for c in feats if c not in GPS],
        'C radio only': radio,
        'D radio, no campaign': [c for c in radio if c not in CAMPAIGN],
    }


# ----------------------------------------------------------------- stage 1: RF
def stage_rf(D, feats):
    POL = build_policies(feats)
    rows = []
    print("Stage 1: random forest, three protocols")
    print("=" * 80, flush=True)
    for tgt, ycol, labs in [('8 modes', 'y', ALLC), ('4 families', 'fam', list(range(NFAM)))]:
        print("\n--- target: %s ---" % tgt, flush=True)
        for pname, fs in POL.items():
            X = D[fs].fillna(0).to_numpy(dtype=np.float64)
            y = D[ycol].to_numpy()
            rnd, grp = [], []
            for sd in SEEDS:
                r = np.random.RandomState(sd)
                m = r.rand(len(D)) < TEST_FRAC
                rf = RandomForestClassifier(n_estimators=200, random_state=sd, n_jobs=2).fit(X[~m], y[~m])
                rnd.append(f1_score(y[m], rf.predict(X[m]), average='macro', labels=labs, zero_division=0))
                gm, _ = grouped_split(D, sd)
                rf2 = RandomForestClassifier(n_estimators=200, random_state=sd, n_jobs=2).fit(X[~gm], y[~gm])
                grp.append(f1_score(y[gm], rf2.predict(X[gm]), average='macro', labels=labs, zero_division=0))
            loco = []
            for ho in CITIES:
                tr = (D.city != ho).values
                rf3 = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(X[tr], y[tr])
                loco.append(f1_score(y[~tr], rf3.predict(X[~tr]), average='macro', labels=labs, zero_division=0))
            rows.append(dict(target=tgt, policy=pname, n_feat=len(fs),
                             random=np.mean(rnd), grouped=np.mean(grp),
                             grouped_sd=np.std(grp), loco=np.mean(loco),
                             **{('loco_' + c): v for c, v in zip(CITIES, loco)}))
            print("  %-22s n=%2d  random %.3f | GROUPED %.3f (sd %.3f) | LOCO %.3f  [%s]"
                  % (pname, len(fs), np.mean(rnd), np.mean(grp), np.std(grp), np.mean(loco),
                     " ".join("%s %.2f" % (c[:4], v) for c, v in zip(CITIES, loco))), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "ladder_grouped_rf.csv"), index=False)
    print("\nsaved ladder_grouped_rf.csv", flush=True)


# ------------------------------------------------------------ stage 2: federated
class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        s.fc_out = nn.Linear(256, NC)

    def forward(s, x):
        return s.fc_out(s.net(x))


def federate(Xtr, ytr, Xte, clients, nf, cw, restore, classwise, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(clients)
    counts = np.array([[int((ytr[ix] == k).sum()) for k in range(NC)] for ix in clients], float)
    sizes = counts.sum(1); pk = sizes / sizes.sum()
    cc = [set(int(v) for v in np.unique(ytr[ix])) for ix in clients]
    g = MLP(nf).state_dict()
    for _rnd in range(ROUNDS):
        locs = []
        for i, ix in enumerate(clients):
            m = MLP(nf); m.load_state_dict(copy.deepcopy(g))
            if cw:
                w = torch.zeros(NC); pres = sorted(cc[i]); n = len(ix)
                for k in pres:
                    if counts[i, k] > 0:
                        w[k] = n / (len(pres) * counts[i, k])
                crit = nn.CrossEntropyLoss(weight=w)
            else:
                crit = nn.CrossEntropyLoss()
            opt = optim.SGD(m.parameters(), lr=LR, momentum=MOM)
            dl = DataLoader(TensorDataset(torch.tensor(Xtr[ix]), torch.tensor(ytr[ix])),
                            batch_size=BS, shuffle=True)
            m.train()
            for _e in range(EPOCHS):
                for xb, yb in dl:
                    opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
            if restore:
                with torch.no_grad():
                    for k in [k for k in range(NC) if k not in cc[i]]:
                        m.fc_out.weight[k].copy_(g['fc_out.weight'][k])
                        m.fc_out.bias[k].copy_(g['fc_out.bias'][k])
            locs.append(m)
        if not classwise:
            g = {k: sum(pk[i] * locs[i].state_dict()[k].float() for i in range(K)) for k in g}
        else:
            ct = counts.sum(0)
            wc = np.where(ct > 0, counts / np.maximum(ct, 1e-12), 1.0 / K)
            new = {}
            for k in g:
                if 'fc_out' not in k:
                    new[k] = sum(pk[i] * locs[i].state_dict()[k].float() for i in range(K))
            ow = torch.zeros_like(g['fc_out.weight']); ob = torch.zeros_like(g['fc_out.bias'])
            for k in range(NC):
                for i in range(K):
                    ow[k] += wc[i, k] * locs[i].state_dict()['fc_out.weight'][k].float()
                    ob[k] += wc[i, k] * locs[i].state_dict()['fc_out.bias'][k].float()
            new['fc_out.weight'] = ow; new['fc_out.bias'] = ob
            g = new
    m = MLP(nf); m.load_state_dict(g); m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).argmax(1).numpy()


def prep(tr, te, fs):
    mu = tr[fs].fillna(0).mean(); sg = tr[fs].fillna(0).std().replace(0, 1)
    Xtr = ((tr[fs].fillna(0) - mu) / sg).to_numpy(np.float32)
    Xte = ((te[fs].fillna(0) - mu) / sg).to_numpy(np.float32)
    return Xtr, Xte


def stage_fed(D, feats):
    POL = build_policies(feats)
    use = {'A all features': POL['A all features'], 'B no position': POL['B no position']}
    METH = [('FedAvg', False, False, False), ('FedObserve', True, True, True)]
    rows = []; t0 = time.time()
    print("Stage 2: federated, policies A and B")
    print("=" * 80, flush=True)

    print("\n--- 2a: four-city federation, journey-grouped test set ---", flush=True)
    for pname, fs in use.items():
        for mname, cw, rest, clw in METH:
            sc, per, trn = [], {c: [] for c in CITIES}, []
            for sd in SEEDS:
                gm, _ = grouped_split(D, sd)
                tr, te = D[~gm], D[gm]
                Xtr, Xte = prep(tr, te, fs)
                ytr = tr.y.to_numpy(); yte = te.y.to_numpy()
                clients = [np.where(tr.city.to_numpy() == c)[0] for c in CITIES]
                yp = federate(Xtr, ytr, Xte, clients, len(fs), cw, rest, clw, sd)
                sc.append(f1_score(yte, yp, average='macro', labels=ALLC, zero_division=0))
                pc = f1_score(yte, yp, average=None, labels=ALLC, zero_division=0)
                trn.append(pc[7])
                for c in CITIES:
                    msk = te.city.to_numpy() == c
                    if msk.sum():
                        per[c].append(f1_score(yte[msk], yp[msk], average='macro',
                                               labels=ALLC, zero_division=0))
            rows.append(dict(stage='grouped', policy=pname, method=mname,
                             macro_f1=np.mean(sc), sd=np.std(sc), f1_train=np.mean(trn),
                             **{('city_' + c): (np.mean(v) if v else np.nan) for c, v in per.items()}))
            print("  %-16s %-11s grouped macro F1 %.4f (sd %.4f)  train %.3f  [%s]  (%.0fs)"
                  % (pname, mname, np.mean(sc), np.std(sc), np.mean(trn),
                     " ".join("%s %.2f" % (c[:4], np.mean(v) if v else float('nan'))
                              for c, v in per.items()), time.time() - t0), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "ladder_grouped_fed.csv"), index=False)

    print("\n--- 2b: leave-one-city-out, federate over three cities, test on the fourth ---",
          flush=True)
    for pname, fs in use.items():
        for mname, cw, rest, clw in METH:
            hold = {}
            for ho in CITIES:
                tr = D[D.city != ho]; te = D[D.city == ho]
                others = [c for c in CITIES if c != ho]
                Xtr, Xte = prep(tr, te, fs)
                ytr = tr.y.to_numpy(); yte = te.y.to_numpy()
                clients = [np.where(tr.city.to_numpy() == c)[0] for c in others]
                v = []
                for sd in SEEDS:
                    yp = federate(Xtr, ytr, Xte, clients, len(fs), cw, rest, clw, sd)
                    v.append(f1_score(yte, yp, average='macro', labels=ALLC, zero_division=0))
                hold[ho] = float(np.mean(v))
            rows.append(dict(stage='loco', policy=pname, method=mname,
                             macro_f1=float(np.mean(list(hold.values()))), sd=np.nan,
                             f1_train=np.nan, **{('city_' + c): hold[c] for c in CITIES}))
            print("  %-16s %-11s LOCO mean %.4f  [%s]  (%.0fs)"
                  % (pname, mname, np.mean(list(hold.values())),
                     " ".join("%s %.2f" % (c[:4], hold[c]) for c in CITIES),
                     time.time() - t0), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "ladder_grouped_fed.csv"), index=False)
    print("\nsaved ladder_grouped_fed.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rf", "fed"], required=True)
    a = ap.parse_args()
    D = load()
    feats = [c for c in D.columns
             if c not in ('label', 'city', 'session', 'win_id', 'y', 'fam', 'sid')]
    print("windows %d   features %d   journeys %d" % (len(D), len(feats), D.sid.nunique()))
    _, singles = grouped_split(D, 42)
    if singles:
        print("single-journey classes (kept in TRAIN, no test support in that city):")
        for city, lab, sid in singles:
            print("   %-8s %-12s %s" % (city, lab, sid))
    print()
    if a.stage == "rf":
        stage_rf(D, feats)
    else:
        stage_fed(D, feats)
