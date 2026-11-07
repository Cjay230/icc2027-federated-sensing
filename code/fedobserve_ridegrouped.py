# THE PAPER'S REPORTED CONFIGURATION, RIDE-GROUPED.
#
# Table III of icc.measurement.tex reports FedObserve WITH class-wise aggregation (alpha),
# from grouped_10seed.csv: 0.6214 macro F1 and 0.1414 on train. main_table_ridegrouped.py
# re-ran the alpha-dropped variant instead, so it does not speak to the reported table.
# This runs the reported configuration under ride grouping.
#
# Identical to grouped_10seed.py in seeds, hyperparameters, arms, aggregation and scoring.
# The only change is that sid is the RIDE: the twins confirmed in confirmed_pairs.json merge
# into one split unit, so a ride recorded by two phones cannot straddle the split.
import os, copy, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from scipy import stats as sps

WIN = "C:/Users/user/Desktop/VIP 2/rebuilt_cache"
OUT = "C:/Users/user/Desktop/VIP 2/rebuilt_results"
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
LM = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3, 'bike': 4, 'motorcycle': 5, 'jog': 6, 'train': 7}
NC = 8; ALLC = list(range(NC))
BS, LR, MOM, EPOCHS, ROUNDS = 32, 0.01, 0.9, 5, 30
SEEDS = (42, 123, 456, 7, 21, 99, 202, 303, 404, 505)
TEST_FRAC = 0.2


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


def grouped_split(D, seed):
    rng = np.random.RandomState(seed); test = set()
    for (city, lab), g in D.groupby(['city', 'label']):
        s = sorted(g.sid.unique())
        if len(s) < 2:
            continue
        rng.shuffle(s)
        k = min(max(1, int(round(TEST_FRAC * len(s)))), len(s) - 1)
        test.update(s[:k])
    return D.sid.isin(test).values


class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        s.fc_out = nn.Linear(256, NC)

    def forward(s, x):
        return s.fc_out(s.net(x))


def federate(Xtr, ytr, Xte_by_city, clients, nf, cw, restore, classwise, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(clients)
    counts = np.array([[int((ytr[ix] == k).sum()) for k in range(NC)] for ix in clients], float)
    pk = counts.sum(1) / counts.sum()
    cc = [set(int(v) for v in np.unique(ytr[ix])) for ix in clients]
    g = MLP(nf).state_dict()
    for _r in range(ROUNDS):
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
            ct = counts.sum(0); wc = np.where(ct > 0, counts / np.maximum(ct, 1e-12), 1.0 / K)
            new = {k: sum(pk[i] * locs[i].state_dict()[k].float() for i in range(K))
                   for k in g if 'fc_out' not in k}
            ow = torch.zeros_like(g['fc_out.weight']); ob = torch.zeros_like(g['fc_out.bias'])
            for k in range(NC):
                for i in range(K):
                    ow[k] += wc[i, k] * locs[i].state_dict()['fc_out.weight'][k].float()
                    ob[k] += wc[i, k] * locs[i].state_dict()['fc_out.bias'][k].float()
            new['fc_out.weight'] = ow; new['fc_out.bias'] = ob
            g = new
    m = MLP(nf); m.load_state_dict(g); m.eval()
    out = {}
    with torch.no_grad():
        for c in CITIES:
            out[c] = m(torch.tensor(Xte_by_city[c])).numpy()
    return out, {c: cc[i] for i, c in enumerate(CITIES)}


def score(te, lg, avail=None):
    allp, allt = [], []
    for c in CITIES:
        msk = te.city.to_numpy() == c
        if msk.sum() == 0:
            continue
        L = lg[c].copy()
        if avail is not None:
            L[:, [k for k in range(NC) if k not in avail[c]]] = -1e9
        allp.append(L.argmax(1)); allt.append(te.y.to_numpy()[msk])
    allp = np.concatenate(allp); allt = np.concatenate(allt)
    mac = f1_score(allt, allp, average='macro', labels=ALLC, zero_division=0)
    pc = f1_score(allt, allp, average=None, labels=ALLC, zero_division=0)
    return float(mac), float(pc[7])


if __name__ == "__main__":
    D = load()
    feats = [c for c in D.columns
             if c not in ('label', 'city', 'session', 'win_id', 'y', 'sid', 'file', 'ride')]
    nf = len(feats)
    print("FedObserve (WITH alpha), RIDE-GROUPED. %d features, %d windows, %d rides."
          % (nf, len(D), D.sid.nunique()))
    print("=" * 80, flush=True)
    rows = []; t0 = time.time()
    for sd in SEEDS:
        gm = grouped_split(D, sd)
        tr, te = D[~gm], D[gm]
        mu = tr[feats].fillna(0).mean(); sg = tr[feats].fillna(0).std().replace(0, 1)
        Xtr = ((tr[feats].fillna(0) - mu) / sg).to_numpy(np.float32)
        ytr = tr.y.to_numpy()
        clients = [np.where(tr.city.to_numpy() == c)[0] for c in CITIES]
        Xte = {c: ((te[te.city == c][feats].fillna(0) - mu) / sg).to_numpy(np.float32)
               for c in CITIES}
        for label, cw, rest, cwise in [('FedAvg', False, False, False),
                                       ('FedObserve', True, True, True)]:
            lg, avail = federate(Xtr, ytr, Xte, clients, nf, cw, rest, cwise, sd)
            for suffix, mk in [('', None), (' + mask', avail)]:
                mac, trf = score(te, lg, mk)
                rows.append(dict(method=label + suffix, seed=sd, macro_f1=mac, f1_train=trf))
        print("  seed %3d done (%.0fs)" % (sd, time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "fedobserve_ridegrouped.csv"), index=False)

    R = pd.DataFrame(rows)
    order = ['FedAvg', 'FedAvg + mask', 'FedObserve', 'FedObserve + mask']
    print()
    print("  %-24s %-24s %s" % ('method', 'macro F1', 'train F1'))
    for m in order:
        d = R[R['method'] == m]
        print("  %-24s %.4f +- %.4f        %.4f +- %.4f"
              % (m, d.macro_f1.mean(), d.macro_f1.std(), d.f1_train.mean(), d.f1_train.std()))

    print()
    print("  FAIR TEST (both masked), ride-grouped")
    a = R[R['method'] == 'FedAvg + mask'].sort_values('seed')
    b = R[R['method'] == 'FedObserve + mask'].sort_values('seed')
    for metric in ['macro_f1', 'f1_train']:
        x = a[metric].to_numpy(); y = b[metric].to_numpy()
        st, p = sps.wilcoxon(y, x, alternative='greater')
        print("    %-10s %.4f vs %.4f   wins %2d/10   median %+.4f   p = %.4f"
              % (metric, y.mean(), x.mean(), int((y > x).sum()),
                 float(np.median(y - x)), p))

    old = os.path.join(OUT, "grouped_10seed.csv")
    if os.path.exists(old):
        O = pd.read_csv(old)
        print()
        print("  AGAINST THE TABLE IN THE PAPER (file-grouped -> ride-grouped)")
        for m in ['FedAvg', 'FedObserve', 'FedObserve + mask']:
            if m not in set(O.method):
                continue
            aa = O[O['method'] == m].sort_values('seed')
            bb = R[R['method'] == m].sort_values('seed')
            print("    %-20s macro %.4f -> %.4f (%+.4f)   train %.4f -> %.4f (%+.4f)"
                  % (m, aa.macro_f1.mean(), bb.macro_f1.mean(),
                     bb.macro_f1.mean() - aa.macro_f1.mean(),
                     aa.f1_train.mean(), bb.f1_train.mean(),
                     bb.f1_train.mean() - aa.f1_train.mean()))
    print("\nsaved fedobserve_ridegrouped.csv")
