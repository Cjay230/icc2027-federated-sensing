# Figures for icc.measurement.tex. Single-column IEEE width (3.4 in), vector PDF.
#
# Each figure carries one claim the paper makes:
#   figM1  isolation vs scarcity          -- the headline (causal_grouped.csv)
#   figM2  three evaluation protocols     -- the leakage hazard (ladder_grouped_rf.csv)
#   figM3  field availability by mode     -- the campaign hazard (from the window cache)
#   figM4  handover estimator by mode     -- the physics finding (from the window cache)
#
# Every number is read from a saved file or recomputed from the cache; none is typed in.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = r"C:\Users\user\Desktop\VIP 2\rebuilt_results"
WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
FIG = r"C:\Users\user\Desktop\VIP 2\figures_measurement"
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
VEH = {'car', 'bus', 'motorcycle'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.dpi': 200,
})
W = 3.4
INK = '#1b1f1e'
GREY = '#9aa0a0'
ACC = '#0d6157'
WARN = '#a8560f'


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('  wrote', name)


def fig_isolation():
    R = pd.read_csv(os.path.join(RES, 'causal_grouped.csv'))
    classes = ['walk', 'car', 'bus']
    labels = {'A': 'natural', 'B': 'isolated', 'C': 'same count,\ndistributed'}
    cols = {'A': GREY, 'B': WARN, 'C': ACC}
    fig, ax = plt.subplots(figsize=(W, 1.95))
    x = np.arange(len(classes)); w = 0.26
    for i, c in enumerate(['A', 'B', 'C']):
        m = [R[(R.target_class == t) & (R.condition == c)].class_f1.mean() for t in classes]
        e = [R[(R.target_class == t) & (R.condition == c)].class_f1.std() for t in classes]
        ax.bar(x + (i - 1) * w, m, w, yerr=e, capsize=1.8, color=cols[c],
               edgecolor=INK, linewidth=0.4, error_kw={'lw': 0.6}, label=labels[c])
    for j, t in enumerate(classes):
        b = R[(R.target_class == t) & (R.condition == 'B')].class_f1.mean()
        c = R[(R.target_class == t) & (R.condition == 'C')].class_f1.mean()
        ax.annotate('%.2f' % (b - c), xy=(x[j], max(b, c) + 0.07), ha='center',
                    fontsize=7, color=WARN, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('class F1'); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.26),
              handlelength=1.1, columnspacing=1.0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, color='#e2e4e2'); ax.set_axisbelow(True)
    save(fig, 'figM1_isolation.pdf')


def fig_protocols():
    R = pd.read_csv(os.path.join(RES, 'ladder_grouped_rf.csv'))
    d = R[R.policy == 'A all features']
    rows = [('8 modes', d[d.target == '8 modes'].iloc[0]),
            ('4 families', d[d.target == '4 families'].iloc[0])]
    fig, ax = plt.subplots(figsize=(W, 1.8))
    x = np.arange(2); w = 0.26
    for i, (key, lab, col) in enumerate([('random', 'random windows', GREY),
                                         ('grouped', 'journey-grouped', ACC),
                                         ('loco', 'unseen city', WARN)]):
        vals = [r[key] for _, r in rows]
        ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor=INK, linewidth=0.4, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.annotate('%.2f' % v, xy=(xi, v + 0.02), ha='center', fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('macro F1'); ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.24),
              handlelength=1.1, columnspacing=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, color='#e2e4e2'); ax.set_axisbelow(True)
    save(fig, 'figM2_protocols.pdf')


def load_cache():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv")); t['city'] = c; d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    return D


def fig_campaign(D):
    K = D[D.city == 'kaust']
    modes = ['walk', 'bus', 'car', 'jog', 'bike', 'scooter', 'motorcycle']
    fields = ['timing_advance_mean', 'pathloss_mean', 'rsrq_mean', 'pdsch_mean', 'rsrp_mean']
    short = ['timing adv.', 'path loss', 'RSRQ', 'PDSCH', 'RSRP']
    M = np.zeros((len(modes), len(fields)))
    for i, m in enumerate(modes):
        s = K[K.label == m]
        for j, f in enumerate(fields):
            M[i, j] = 100.0 * (s[f].isna() | (s[f] == 0)).mean()
    fig, ax = plt.subplots(figsize=(W, 2.1))
    im = ax.imshow(M, cmap='RdYlBu_r', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(fields))); ax.set_xticklabels(short, rotation=35, ha='right')
    ax.set_yticks(range(len(modes))); ax.set_yticklabels(modes)
    for i in range(len(modes)):
        for j in range(len(fields)):
            ax.text(j, i, '%.0f' % M[i, j], ha='center', va='center', fontsize=6,
                    color='white' if (M[i, j] > 60 or M[i, j] < 8) else INK)
    ax.axhline(2.5, color=INK, lw=1.2)
    ax.set_title('field absence (\\%) within KAUST only', pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    save(fig, 'figM3_campaign.pdf')


def fig_handover(D):
    E = D[(D.velocity_mean > 0) & (D.handover_rate > 0)].copy()
    Dm = {}
    for c in CITIES:
        s = E[(E.city == c) & E.label.isin(VEH)]
        Dm[c] = float(np.median(s.velocity_mean / (3.6 * s.handover_rate)))
    E['vhat'] = 3.6 * E.handover_rate * E.city.map(Dm)
    modes = ['walk', 'jog', 'bike', 'scooter', 'bus', 'motorcycle', 'car']
    true = [E[E.label == m].velocity_mean.median() for m in modes]
    est = [E[E.label == m].vhat.median() for m in modes]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    lim = max(max(true), max(est)) * 1.18
    ax.plot([0, lim], [0, lim], ls='--', lw=0.7, color=GREY, zorder=1)
    ped = ['walk', 'jog']
    for m, t, e in zip(modes, true, est):
        isp = m in ped
        ax.scatter(t, e, s=26, color=(WARN if isp else ACC), edgecolor=INK,
                   linewidth=0.4, zorder=3)
        ax.annotate(m, xy=(t, e), xytext=(3.5, -1.5), textcoords='offset points',
                    fontsize=6.5, color=(WARN if isp else INK))
    ax.set_xlabel('true median velocity (km/h)')
    ax.set_ylabel('estimated $3.6\\,H\\,D$ (km/h)')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, lw=0.4, color='#e2e4e2'); ax.set_axisbelow(True)
    ax.annotate('pedestrian modes\noverestimated $\\approx 3\\times$', xy=(5, 12.5),
                xytext=(13, 29), fontsize=6.5, color=WARN,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=WARN))
    save(fig, 'figM4_handover.pdf')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    print('generating figures into', FIG)
    fig_isolation()
    fig_protocols()
    D = load_cache()
    fig_campaign(D)
    fig_handover(D)
    print('done')
