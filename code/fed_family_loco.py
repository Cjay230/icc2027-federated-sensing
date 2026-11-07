# THE DECIDING RUN: does the disclosure result hold for the federated model at family level?
#
# The claim in question: withholding coordinates improves cross-city macro F1 by 0.113 on
# four kinematic families. That was measured with a random forest (ladder_grouped_rf.csv:
# A 0.282 -> B 0.395). The federated version at 8-class level showed nothing
# (ladder_grouped_fed.csv: policy A 0.1803 / 0.1745, policy B 0.1760 / 0.1781 for
# FedAvg / FedObserve). But the random forest effect was also much weaker at 8-class
# (+0.058) than at family level (+0.113), so the federated family case is untested.
#
# This runs exactly that: 4 kinematic families, leave-one-city-out, federated, policy A
# (all features) vs policy B (no coordinates), FedAvg and FedObserve, 3 seeds.
#
# If B > A here, the disclosure claim stands for the federated method. If not, the finding
# is specific to the random forest and must be reported that way.
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
FAM = {'walk': 0, 'jog': 0, 'bike': 1, 'scooter': 1,
       'car': 2, 'bus': 2, 'motorcycle': 2, 'train': 3}
NC = 4; ALLC = list(range(NC))
BS, LR, MOM = 32, 0.01, 0.9
EPOCHS, ROUNDS = 5, 30
SEEDS = (42, 123, 456, 7, 21, 99, 202, 303, 404, 505)
GPS = ['longitude_mean', 'latitude_mean']


def load():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv")); t['city'] = c; d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(FAM)].reset_index(drop=True)
    D['y'] = D.label.map(FAM).astype(int)
    return D


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
    with torch.no_grad():
        return m(torch.tensor(Xte)).argmax(1).numpy()


if __name__ == "__main__":
    D = load()
    feats = [c for c in D.columns if c not in ('label', 'city', 'session', 'win_id', 'y')]
    POL = {'A all features': feats, 'B no position': [c for c in feats if c not in GPS]}
    print("windows %d   target: 4 kinematic families" % len(D))
    print("family sizes:", D.y.value_counts().sort_index().to_dict())
    print("families per city:", {c: sorted(D[D.city == c].y.unique()) for c in CITIES})
    print("=" * 84, flush=True)

    rows = []; t0 = time.time()
    for pname, fs in POL.items():
        for mname, cw, rest, clw in [('FedAvg', False, False, False),
                                     ('FedObserve', True, True, True)]:
            hold = {}
            for ho in CITIES:
                tr = D[D.city != ho]; te = D[D.city == ho]
                others = [c for c in CITIES if c != ho]
                mu = tr[fs].fillna(0).mean(); sg = tr[fs].fillna(0).std().replace(0, 1)
                Xtr = ((tr[fs].fillna(0) - mu) / sg).to_numpy(np.float32)
                Xte = ((te[fs].fillna(0) - mu) / sg).to_numpy(np.float32)
                ytr = tr.y.to_numpy(); yte = te.y.to_numpy()
                clients = [np.where(tr.city.to_numpy() == c)[0] for c in others]
                v = []
                for sd in SEEDS:
                    yp = federate(Xtr, ytr, Xte, clients, len(fs), cw, rest, clw, sd)
                    f = f1_score(yte, yp, average='macro', labels=ALLC, zero_division=0)
                    v.append(f)
                    rows.append(dict(policy=pname, method=mname, holdout=ho, seed=sd, macro_f1=f))
                hold[ho] = float(np.mean(v))
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "fed_family_loco.csv"), index=False)
            print("  %-16s %-11s LOCO mean %.4f  [%s]  (%.0fs)"
                  % (pname, mname, np.mean(list(hold.values())),
                     " ".join("%s %.2f" % (c[:4], hold[c]) for c in CITIES),
                     time.time() - t0), flush=True)

    R = pd.DataFrame(rows)
    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    for mname in ['FedAvg', 'FedObserve']:
        a = R[(R.method == mname) & (R.policy == 'A all features')].macro_f1
        b = R[(R.method == mname) & (R.policy == 'B no position')].macro_f1
        print("  %-11s  A %.4f (sd %.3f) -> B %.4f (sd %.3f)   delta %+.4f"
              % (mname, a.mean(), a.std(), b.mean(), b.std(), b.mean() - a.mean()))
    print()
    print("  random forest, same target, for reference: A 0.282 -> B 0.395  delta +0.113")
    print("\nsaved fed_family_loco.csv")
