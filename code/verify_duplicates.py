# Reproduces the duplicate-recording finding from Khalil's 28 Oct review.
#
# Khalil identified 17 recordings made by a second or third phone on a ride already in the
# corpus. His pipeline drops them. Mine grouped by source_file, which treats a twin as an
# independent journey, so one phone's log could sit in training while the other sat in test.
#
# This script confirms the pairing two independent ways and measures the damage:
#
#   method 1  window-feature distance, within the same city and class. Two phones a few
#             metres apart for a whole ride produce near-identical window statistics.
#   method 2  overlap of the raw clock-time spans. Two logs recorded simultaneously cover
#             the same wall-clock interval.
#
# Method 2 alone over-detects, because the raw logs carry clock time with no date, so two
# rides on different days at the same hour look simultaneous. Files whose own span exceeds
# six hours are a midnight-wrap artefact and are excluded from the time evidence. A pair is
# reported only when BOTH methods agree, which leaves 15 of the 17.
#
# Finally the paper's own splitter is run over its ten seeds to count how often a confirmed
# pair lands on opposite sides.
#
# Expected output: 15 confirmed pairs, mean 5.5 of 15 straddling per seed, 10 of 10 seeds
# containing at least one leak.
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = "C:/Users/user/Desktop/VIP 2/rebuilt_cache"
RES = os.path.join(HERE, "..", "results")
CITIES = ['jeddah', 'kaust', 'kz', 'mekkah']
LM = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3, 'bike': 4, 'motorcycle': 5, 'jog': 6, 'train': 7}
SEEDS = (42, 123, 456, 7, 21, 99, 202, 303, 404, 505)
TEST_FRAC = 0.2
MAX_SPAN_S = 6 * 3600       # a longer span means the clock wrapped midnight
MIN_TIME_RATIO = 0.8        # overlap as a fraction of the shorter of the two files

# the 17 files Khalil flagged by hand, from study1_cross_region.ipynb cell 3
FLAGGED = [
    "bus_colored_kaust_006", "bus_colored_kaust_007", "bus_jeddah_010", "bus_jeddah_011",
    "bus_mekkah_023", "bus_mekkah_024", "bus_mekkah_025", "bus_mekkah_026",
    "car_jeddah_034", "car_jeddah_036", "car_jeddah_040", "car_kaust_043", "car_kaust_045",
    "walk_jeddah_071", "walk_jeddah_072", "walk_kaust_101", "walk_kaust_102",
]


def load_windows():
    d = []
    for c in CITIES:
        t = pd.read_csv(os.path.join(WIN, c + "_60s_sess.csv"))
        t['city'] = c
        d.append(t)
    D = pd.concat(d, ignore_index=True)
    D['label'] = D['label'].astype(str)
    D = D[D.label.isin(LM)].reset_index(drop=True)
    D['file'] = D.session.astype(str).str.replace('.csv', '', regex=False)
    D['sid'] = D.city + "::" + D.session.astype(str)
    return D


def feature_profiles(D):
    """One standardized mean feature vector per file."""
    feats = [c for c in D.columns
             if c not in ('label', 'city', 'session', 'win_id', 'file', 'sid')]
    Z = D[feats].fillna(0)
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    return pd.concat([D[['city', 'label', 'file']], Z], axis=1) \
             .groupby(['city', 'label', 'file'])[feats].mean()


def confirm(prof, spans):
    """For each flagged file, take its nearest neighbour that also overlaps it in time."""
    bad = set(spans[spans.span > MAX_SPAN_S].file)
    if bad:
        print("excluded from time evidence (clock wrapped midnight): %s\n" % sorted(bad))
    sp = spans.set_index('file')
    confirmed, unconfirmed = [], []
    print("%-22s %-22s %6s %8s" % ('flagged file', 'twin', 'dist', 'overlap'))
    for (city, lab), g in prof.groupby(level=[0, 1]):
        V = g.to_numpy()
        names = [i[2] for i in g.index]
        for a, nm in enumerate(names):
            if nm not in FLAGGED:
                continue
            dist = np.linalg.norm(V - V[a], axis=1)
            dist[a] = np.inf
            hit = None
            for b in np.argsort(dist)[:5]:
                o = names[b]
                if nm in bad or o in bad or nm not in sp.index or o not in sp.index:
                    continue
                ov = min(sp.loc[nm, 't1'], sp.loc[o, 't1']) - max(sp.loc[nm, 't0'], sp.loc[o, 't0'])
                if ov / min(sp.loc[nm, 'span'], sp.loc[o, 'span']) > MIN_TIME_RATIO:
                    hit = (o, dist[b], ov)
                    break
            if hit:
                confirmed.append((nm, hit[0]))
                print("%-22s %-22s %6.2f %8.0f" % (nm, hit[0], hit[1], hit[2]))
            else:
                unconfirmed.append(nm)
                print("%-22s %-22s %6s %8s   not confirmed" % (nm, '-', '-', '-'))
    print("\nconfirmed by both methods: %d of %d flagged files" % (len(confirmed), len(FLAGGED)))
    if unconfirmed:
        print("unconfirmed (no usable time evidence): %s" % unconfirmed)
    return confirmed


def grouped_split(D, seed):
    """Byte-for-byte the splitter used in the paper: groups by file, not by ride."""
    rng = np.random.RandomState(seed)
    test = set()
    for (city, lab), g in D.groupby(['city', 'label']):
        s = sorted(g.sid.unique())
        if len(s) < 2:
            continue
        rng.shuffle(s)
        k = min(max(1, int(round(TEST_FRAC * len(s)))), len(s) - 1)
        test.update(s[:k])
    return test


def measure(D, pairs):
    city_of = dict(zip(D.file, D.city))
    pairs = sorted({tuple(sorted(p)) for p in pairs if p[0] in city_of and p[1] in city_of})
    files = {x for p in pairs for x in p}
    n = len(D[D.file.isin(files)])
    print("\nunique ride pairs: %d, covering %d windows (%.1f%% of %d)"
          % (len(pairs), n, 100.0 * n / len(D), len(D)))
    counts = []
    for sd in SEEDS:
        test = grouped_split(D, sd)
        sep = sum(((city_of[a] + "::" + a + ".csv") in test)
                  != ((city_of[b] + "::" + b + ".csv") in test) for a, b in pairs)
        counts.append(sep)
        print("  seed %3d: %d of %d pairs split across train/test" % (sd, sep, len(pairs)))
    print("\nmean %.1f of %d pairs separated per seed (%.0f%%)"
          % (np.mean(counts), len(pairs), 100.0 * np.mean(counts) / len(pairs)))
    print("at least one leak in %d of %d seeds" % (sum(1 for x in counts if x > 0), len(SEEDS)))
    return pairs


if __name__ == "__main__":
    D = load_windows()
    spans = pd.read_csv(os.path.join(RES, "file_timespans.csv"))
    spans['span'] = spans.t1 - spans.t0
    print("=" * 78)
    print("CONFIRMING THE DUPLICATE RECORDINGS (feature distance AND time overlap)")
    print("=" * 78)
    pairs = confirm(feature_profiles(D), spans)
    print()
    print("=" * 78)
    print("HOW OFTEN A CONFIRMED PAIR STRADDLES THE PAPER'S SPLIT")
    print("=" * 78)
    pairs = measure(D, pairs)

    saved = os.path.join(RES, "confirmed_pairs.json")
    if os.path.exists(saved):
        ref = sorted({tuple(sorted(p)) for p in json.load(open(saved))})
        print("\nmatches results/confirmed_pairs.json:", ref == pairs)
