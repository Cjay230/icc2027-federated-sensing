# FACTORIAL ABLATION, journey-grouped. Answers reviewer comments 3 and 4.
#
# The paper reports a CUMULATIVE ladder (+0.020 / +0.008 / +0.002), which orders the three
# mechanisms but cannot isolate any of them. The reviewer asks for a factorial, and for the
# claim "most current gains come from local class weighting" to be tested directly.
#
# 2^3 arms: class-weighted loss (cw) x absent-row restoration (restore) x class-wise
# aggregation (classwise). Inference masking is a free re-scoring of each trained model,
# so 8 trainings give 16 results per seed.
#
# This also settles whether "one rule at three points" survives. The loss weight is
# inverse-frequency (lambda ~ 1/n) while the aggregation weight is proportional
# (alpha ~ n); if cw carries the gain alone and restore/classwise add little, the rule
# governs two points and the loss weighting is a separate standard component.
import os, copy, time, itertools
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
BS, LR, MOM, EPOCHS, ROUNDS = 32, 0.01, 0.9, 5, 30
SEEDS = (42, 123, 456, 7, 21)
TEST_FRAC = 0.2


def load():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv")); t['city'] = c; d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(LM)].reset_index(drop=True)
    D['y'] = D.label.map(LM).astype(int)
    D['sid'] = D.city + "::" + D.session.astype(str)
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
    feats = [c for c in D.columns if c not in ('label', 'city', 'session', 'win_id', 'y', 'sid')]
    nf = len(feats)
    print("FACTORIAL ABLATION, journey-grouped, %d seeds. windows %d" % (len(SEEDS), len(D)))
    print("=" * 80, flush=True)
    rows = []; t0 = time.time()
    for sd in SEEDS:
        gm = grouped_mask(D, sd)
        tr, te = D[~gm], D[gm]
        mu = tr[feats].fillna(0).mean(); sg = tr[feats].fillna(0).std().replace(0, 1)
        Xtr = ((tr[feats].fillna(0) - mu) / sg).to_numpy(np.float32)
        ytr = tr.y.to_numpy()
        clients = [np.where(tr.city.to_numpy() == c)[0] for c in CITIES]
        Xte = {c: ((te[te.city == c][feats].fillna(0) - mu) / sg).to_numpy(np.float32)
               for c in CITIES}
        for cw, rest, clw in itertools.product([0, 1], [0, 1], [0, 1]):
            lg, avail = federate(Xtr, ytr, Xte, clients, nf, bool(cw), bool(rest), bool(clw), sd)
            for mk in (0, 1):
                mac, trf = score(te, lg, avail if mk else None)
                rows.append(dict(cw=cw, restore=rest, classwise=clw, mask=mk, seed=sd,
                                 macro_f1=mac, f1_train=trf))
        print("  seed %3d done (%.0fs)" % (sd, time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "factorial.csv"), index=False)

    R = pd.DataFrame(rows)
    print()
    print("  cw rest clw mask |  macro F1          | train F1")
    print("  " + "-" * 58)
    for cw, rest, clw, mk in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
        d = R[(R.cw == cw) & (R.restore == rest) & (R.classwise == clw) & (R.mask == mk)]
        print("  %2d %4d %3d %4d | %.4f +- %.4f | %.4f" %
              (cw, rest, clw, mk, d.macro_f1.mean(), d.macro_f1.std(), d.f1_train.mean()))

    print()
    print("=" * 80)
    print("MAIN EFFECTS (mean over all settings of the other factors)")
    print("=" * 80)
    for f in ['cw', 'restore', 'classwise', 'mask']:
        off = R[R[f] == 0]; on = R[R[f] == 1]
        print("  %-10s macro F1 %+.4f   train F1 %+.4f"
              % (f, on.macro_f1.mean() - off.macro_f1.mean(),
                 on.f1_train.mean() - off.f1_train.mean()))
    print()
    print("  If cw dominates and restore/classwise are near zero, the 'one rule at three")
    print("  points' claim does not hold and must be restated.")
    print("\nsaved factorial.csv")
