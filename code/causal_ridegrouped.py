# ISOLATION vs SCARCITY, on journey-grouped splits.
#
# This is the paper's headline result under the measurement framing, and it is the last
# major experiment still measured on random window splits (causal_multiclass_seeds.py uses
# train_test_split(random_state=42), so windows of one trip sat on both sides).
#
# Design unchanged from the original, only the split rule differs:
#   A  baseline  : the target class as it naturally occurs
#   B  isolated  : the class kept ONLY at its largest holder, removed elsewhere
#   C  rarity ctl: the class left distributed but subsampled to B's total training count
#
# B and C train on the SAME number of samples and differ only in concentration, so
#   (B - C) isolates concentration from count
#   (C - A) measures count alone
# If B collapses and C does not, structural isolation rather than scarcity is the cause.
#
# A, B and C share the same journey-grouped split and the same initial weights within a
# seed, so the three are paired. The test set is never modified by the conditions.
import os, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
OUT = r"C:\Users\user\Desktop\VIP 2\rebuilt_results"
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
LM = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3, 'bike': 4, 'motorcycle': 5, 'jog': 6, 'train': 7}
NC = 8; ALLC = list(range(NC))
ROUNDS, BS, LR, MOM, EPOCHS = 30, 32, 0.01, 0.9, 5
SEEDS = (42, 123, 456)
TEST_FRAC = 0.2
TARGETS = ['walk', 'car', 'bus']


class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        s.fc_out = nn.Linear(256, NC)

    def forward(s, x):
        return s.fc_out(s.net(x))


def ride_map():
    pairs = [tuple(p) for p in json.load(open(os.path.join(OUT, "confirmed_pairs.json")))]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return {f: find(f) for f in parent}


def load():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv")); t['city'] = c; d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(LM)].reset_index(drop=True)
    D['y'] = D.label.map(LM).astype(int)
    D['file'] = D.session.astype(str).str.replace('.csv', '', regex=False)
    rm = ride_map()
    D['ride'] = D.file.map(lambda f: rm.get(f, f))
    D['sid'] = D.city + "::" + D.ride
    return D


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


def conditions(tr, target, keep, rng):
    y = tr.y.to_numpy(); city = tr.city.to_numpy()
    idxA = np.arange(len(tr))
    dropB = (y == target) & (city != keep)
    idxB = np.where(~dropB)[0]
    n_keep = int(((y == target) & (city == keep)).sum())
    n_tot = int((y == target).sum())
    keep_mask = np.ones(len(tr), bool)
    for c in CITIES:
        ix = np.where((y == target) & (city == c))[0]
        if len(ix) == 0:
            continue
        n = max(1, int(round(len(ix) * n_keep / max(n_tot, 1))))
        drop = rng.permutation(ix)[n:]
        keep_mask[drop] = False
    idxC = np.where(keep_mask)[0]
    return {'A': idxA, 'B': idxB, 'C': idxC}


def fedavg(Xtr, ytr, cityarr, Xte, nf, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    clients = [np.where(cityarr == c)[0] for c in CITIES]
    clients = [ix for ix in clients if len(ix) >= BS]
    sizes = np.array([len(ix) for ix in clients], float); pk = sizes / sizes.sum()
    g = MLP(nf).state_dict()
    for _r in range(ROUNDS):
        locs = []
        for ix in clients:
            m = MLP(nf); m.load_state_dict(copy.deepcopy(g))
            opt = optim.SGD(m.parameters(), lr=LR, momentum=MOM)
            crit = nn.CrossEntropyLoss()
            dl = DataLoader(TensorDataset(torch.tensor(Xtr[ix]), torch.tensor(ytr[ix])),
                            batch_size=BS, shuffle=True)
            m.train()
            for _e in range(EPOCHS):
                for xb, yb in dl:
                    opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
            locs.append(m)
        g = {k: sum(pk[i] * locs[i].state_dict()[k].float() for i in range(len(clients)))
             for k in g}
    m = MLP(nf); m.load_state_dict(g); m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).argmax(1).numpy()


if __name__ == "__main__":
    D = load()
    feats = [c for c in D.columns
             if c not in ('label', 'city', 'session', 'win_id', 'y', 'sid', 'file', 'ride')]
    nf = len(feats)
    print("ISOLATION vs SCARCITY, journey-grouped splits.  windows %d  features %d" % (len(D), nf))
    print("=" * 86, flush=True)

    rows = []
    for tname in TARGETS:
        target = LM[tname]
        holders = D[D.label == tname].city.value_counts()
        keep = holders.index[0]
        print("\n--- target %s : held by %s, isolating at %s ---"
              % (tname, dict(holders), keep), flush=True)
        for sd in SEEDS:
            gm = grouped_mask(D, sd)
            tr, te = D[~gm].reset_index(drop=True), D[gm]
            mu = tr[feats].fillna(0).mean(); sg = tr[feats].fillna(0).std().replace(0, 1)
            Xtr_all = ((tr[feats].fillna(0) - mu) / sg).to_numpy(np.float32)
            Xte = ((te[feats].fillna(0) - mu) / sg).to_numpy(np.float32)
            ytr_all = tr.y.to_numpy(); yte = te.y.to_numpy()
            cityarr_all = tr.city.to_numpy()
            rng = np.random.RandomState(sd)
            conds = conditions(tr, target, keep, rng)
            line = []
            for cname, ix in conds.items():
                yp = fedavg(Xtr_all[ix], ytr_all[ix], cityarr_all[ix], Xte, nf, sd)
                pc = f1_score(yte, yp, average=None, labels=ALLC, zero_division=0)
                mac = f1_score(yte, yp, average='macro', labels=ALLC, zero_division=0)
                rows.append(dict(target_class=tname, condition=cname, seed=sd,
                                 n_train=len(ix), n_target=int((ytr_all[ix] == target).sum()),
                                 class_f1=float(pc[target]), macro_f1=float(mac)))
                line.append("%s %.3f (n=%d)" % (cname, pc[target], int((ytr_all[ix] == target).sum())))
            print("   seed %3d  %s" % (sd, "   ".join(line)), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "causal_ridegrouped.csv"), index=False)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print("VERDICT  (class F1 of the manipulated class)")
    print("=" * 86)
    print("  %-8s %8s %8s %8s   %10s %10s" % ('class', 'A', 'B', 'C', 'B-A', 'B-C'))
    for tname in TARGETS:
        d = R[R.target_class == tname]
        a = d[d.condition == 'A'].sort_values('seed').class_f1.to_numpy()
        b = d[d.condition == 'B'].sort_values('seed').class_f1.to_numpy()
        c = d[d.condition == 'C'].sort_values('seed').class_f1.to_numpy()
        print("  %-8s %8.3f %8.3f %8.3f   %+10.3f %+10.3f"
              % (tname, a.mean(), b.mean(), c.mean(), (b - a).mean(), (b - c).mean()))
    print()
    print("  B-A  isolating the class    (concentration AND count change)")
    print("  B-C  concentration alone    (count held equal)  <- the causal estimate")
    print("  Clearly negative B-C means isolation, not scarcity, does the damage.")
    print("\nsaved causal_ridegrouped.csv")
