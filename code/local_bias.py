# Can a per-site output BIAS recover FedPer's advantage while keeping one shared model?
#
# Background. masked_inference.py showed FedPer's edge over FedObserve is NOT mainly
# absent-class suppression: masking impossible classes closed only about a quarter of
# the gap, and FedPer still led on the classes each city does hold (jeddah 0.334 vs
# 0.304, mekkah 0.416 vs 0.361). So the remaining advantage is local adaptation to each
# city's feature distribution, not label availability.
#
# FedBN (ICLR 2021) is the established method for feature shift, but it keeps batch-norm
# parameters local and this architecture has no BN layers. The cheapest intervention that
# still adapts per site is to keep only the output layer's BIAS local: 8 numbers per
# client, never aggregated, while the full 256x8 weight matrix stays shared. The model is
# still one shared model plus a per-site offset.
#
# Arms (5 per protocol). Arms 2 and 4 are free re-scorings of 1 and 3, so three
# trainings per seed:
#   1 FedObserve                         (baseline)
#   2 FedObserve + masked inference      (re-score of 1)
#   3 FedObserve + local bias            (trained)
#   4 FedObserve + local bias + mask     (re-score of 3)
#   5 FedPer                             (target to beat)
#
# A negative result here is informative: if local bias does not move the per-city numbers,
# the gap needs the whole head and FedPer's cost is unavoidable.
import os, copy, time
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
BS, LR, MOM = 32, 0.01, 0.9
EPOCHS, ROUNDS = 5, 30
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


def random_mask(D, seed):
    return np.random.RandomState(seed).rand(len(D)) < TEST_FRAC


class MLP(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        s.fc_out = nn.Linear(256, NC)

    def forward(s, x):
        return s.fc_out(s.net(x))


def local_train(m, X, y, cc_i, counts_i, cw):
    if cw:
        w = torch.zeros(NC); pres = sorted(cc_i); n = len(y)
        for k in pres:
            if counts_i[k] > 0:
                w[k] = n / (len(pres) * counts_i[k])
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=LR, momentum=MOM)
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=BS, shuffle=True)
    m.train()
    for _e in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    return m


def train_fedobserve(Xtr, ytr, clients, nf, seed, local_bias):
    """FedObserve. If local_bias, fc_out.bias is never aggregated: 8 numbers stay per site."""
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(clients)
    counts = np.array([[int((ytr[ix] == k).sum()) for k in range(NC)] for ix in clients], float)
    pk = counts.sum(1) / counts.sum()
    cc = [set(int(v) for v in np.unique(ytr[ix])) for ix in clients]
    g = MLP(nf).state_dict()
    biases = [g['fc_out.bias'].clone() for _ in range(K)]
    for _r in range(ROUNDS):
        locs = []
        for i, ix in enumerate(clients):
            m = MLP(nf)
            sd = copy.deepcopy(g)
            if local_bias:
                sd['fc_out.bias'] = biases[i].clone()
            m.load_state_dict(sd)
            m = local_train(m, Xtr[ix], ytr[ix], cc[i], counts[i], True)
            with torch.no_grad():
                for k in [k for k in range(NC) if k not in cc[i]]:
                    m.fc_out.weight[k].copy_(g['fc_out.weight'][k])
                    if not local_bias:
                        m.fc_out.bias[k].copy_(g['fc_out.bias'][k])
            if local_bias:
                biases[i] = m.fc_out.bias.detach().clone()
            locs.append(m)
        ct = counts.sum(0); wc = np.where(ct > 0, counts / np.maximum(ct, 1e-12), 1.0 / K)
        new = {k: sum(pk[i] * locs[i].state_dict()[k].float() for i in range(K))
               for k in g if 'fc_out' not in k}
        ow = torch.zeros_like(g['fc_out.weight']); ob = torch.zeros_like(g['fc_out.bias'])
        for k in range(NC):
            for i in range(K):
                ow[k] += wc[i, k] * locs[i].state_dict()['fc_out.weight'][k].float()
                if not local_bias:
                    ob[k] += wc[i, k] * locs[i].state_dict()['fc_out.bias'][k].float()
        new['fc_out.weight'] = ow
        new['fc_out.bias'] = g['fc_out.bias'] if local_bias else ob
        g = new
    return g, (biases if local_bias else None), cc


def train_fedper(Xtr, ytr, clients, nf, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(clients)
    counts = np.array([[int((ytr[ix] == k).sum()) for k in range(NC)] for ix in clients], float)
    pk = counts.sum(1) / counts.sum()
    cc = [set(int(v) for v in np.unique(ytr[ix])) for ix in clients]
    g = MLP(nf).state_dict()
    heads = [{k: g[k].clone() for k in g if 'fc_out' in k} for _ in range(K)]
    for _r in range(ROUNDS):
        locs = []
        for i, ix in enumerate(clients):
            m = MLP(nf); sd = copy.deepcopy(g); sd.update(heads[i]); m.load_state_dict(sd)
            m = local_train(m, Xtr[ix], ytr[ix], cc[i], counts[i], False)
            heads[i] = {k: m.state_dict()[k].clone() for k in m.state_dict() if 'fc_out' in k}
            locs.append(m)
        g = {k: (g[k] if 'fc_out' in k
                 else sum(pk[i] * locs[i].state_dict()[k].float() for i in range(K))) for k in g}
    return g, heads, cc


def logits_of(state, X, nf):
    m = MLP(nf); m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(torch.tensor(X)).numpy()


def score(te, lg, avail=None):
    allp, allt, per = [], [], {}
    for c in CITIES:
        msk = te.city.to_numpy() == c
        if msk.sum() == 0:
            per[c] = np.nan; continue
        L = lg[c].copy()
        if avail is not None:
            L[:, [k for k in range(NC) if k not in avail[c]]] = -1e9
        yp = L.argmax(1); yt = te.y.to_numpy()[msk]
        per[c] = f1_score(yt, yp, average='macro', labels=ALLC, zero_division=0)
        allp.append(yp); allt.append(yt)
    allp = np.concatenate(allp); allt = np.concatenate(allt)
    pooled = f1_score(allt, allp, average='macro', labels=ALLC, zero_division=0)
    pc = f1_score(allt, allp, average=None, labels=ALLC, zero_division=0)
    return pooled, per, float(pc[7])


def run(D, feats, protocol):
    rows = []; t0 = time.time(); nf = len(feats)
    print("\n" + "=" * 90)
    print("PROTOCOL: %s" % protocol)
    print("=" * 90, flush=True)
    for sd in SEEDS:
        tm = grouped_mask(D, sd) if protocol == 'journey-grouped' else random_mask(D, sd)
        tr, te = D[~tm], D[tm]
        mu = tr[feats].fillna(0).mean(); sg = tr[feats].fillna(0).std().replace(0, 1)
        Xtr = ((tr[feats].fillna(0) - mu) / sg).to_numpy(np.float32)
        ytr = tr.y.to_numpy()
        clients = [np.where(tr.city.to_numpy() == c)[0] for c in CITIES]
        Xte = {c: ((te[te.city == c][feats].fillna(0) - mu) / sg).to_numpy(np.float32)
               for c in CITIES}

        def rec(name, triple):
            pooled, per, trf = triple
            rows.append(dict(protocol=protocol, method=name, seed=sd, pooled=pooled,
                             f1_train=trf, **{('city_' + c): per[c] for c in CITIES}))

        g, _, cc = train_fedobserve(Xtr, ytr, clients, nf, sd, local_bias=False)
        avail = {c: cc[i] for i, c in enumerate(CITIES)}
        lg = {c: logits_of(g, Xte[c], nf) for c in CITIES}
        rec('FedObserve', score(te, lg, None))
        rec('FedObserve + mask', score(te, lg, avail))

        g2, biases, cc2 = train_fedobserve(Xtr, ytr, clients, nf, sd, local_bias=True)
        avail2 = {c: cc2[i] for i, c in enumerate(CITIES)}
        lg2 = {}
        for i, c in enumerate(CITIES):
            sdict = copy.deepcopy(g2); sdict['fc_out.bias'] = biases[i]
            lg2[c] = logits_of(sdict, Xte[c], nf)
        rec('FedObserve + local bias', score(te, lg2, None))
        rec('FedObserve + local bias + mask', score(te, lg2, avail2))

        g3, heads, _ = train_fedper(Xtr, ytr, clients, nf, sd)
        lg3 = {}
        for i, c in enumerate(CITIES):
            sdict = copy.deepcopy(g3); sdict.update(heads[i])
            lg3[c] = logits_of(sdict, Xte[c], nf)
        rec('FedPer', score(te, lg3, None))

        print("  seed %3d done (%.0fs)" % (sd, time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "local_bias.csv"), index=False)

    R = pd.DataFrame(rows)
    order = ['FedObserve', 'FedObserve + mask', 'FedObserve + local bias',
             'FedObserve + local bias + mask', 'FedPer']
    print()
    print("  %-32s %8s  %s  %7s"
          % ('method', 'pooled', "  ".join("%6s" % c[:6] for c in CITIES), 'train'))
    for m in order:
        d = R[R.method == m]
        print("  %-32s %8.4f  %s  %7.3f"
              % (m, d.pooled.mean(),
                 "  ".join("%6.3f" % d['city_' + c].mean() for c in CITIES),
                 d.f1_train.mean()), flush=True)
    fp = R[R.method == 'FedPer'].pooled.mean()
    best = R[R.method == 'FedObserve + local bias + mask'].pooled.mean()
    base = R[R.method == 'FedObserve'].pooled.mean()
    if fp > base:
        print("\n  gap to FedPer closed: %.0f%%  (%.4f -> %.4f against %.4f)"
              % (100 * (best - base) / (fp - base), base, best, fp), flush=True)
    return R


if __name__ == "__main__":
    D = load()
    feats = [c for c in D.columns if c not in ('label', 'city', 'session', 'win_id', 'y', 'sid')]
    print("windows %d  features %d  seeds %s" % (len(D), len(feats), list(SEEDS)))
    allR = [run(D, feats, p) for p in ['random window', 'journey-grouped']]
    pd.concat(allR, ignore_index=True).to_csv(os.path.join(OUT, "local_bias.csv"), index=False)
    print("\nsaved local_bias.csv")
