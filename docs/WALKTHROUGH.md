# Code walkthrough — running order

`README.md` describes the state on 23–24 Oct and is **out of date** — use `CODE_GUIDE.md`
for what each script does, and this file for the order to go through them in.

The bundle holds 13 scripts. Nine superseded ones were removed to keep it focused; they are
all still in `rebuilt_results/` and `CODE_GUIDE.md` Part 2 says what each one was.

Everything below runs from `code/`. Scripts read `rebuilt_cache/*_60s_sess.csv` and write to
`results/`.

---

## 0. The headline: what changed after the review

| | before | after |
|---|---|---|
| split unit | recording file | **ride** (twins merged) |
| isolated class `train` | 0.0071 → 0.1414 | **0.0259 → 0.1289** |
| vs FedRS, macro F1 | lead, p = 0.065 | **tie, p = 0.4229** |
| PDSCH | described as a feature | **named as KAUST-only** |
| GPS | unstated | **named in Sec. III** |

Three of these come directly from Khalil's three points. Say that up front.

---

## 1. The rebuild (unchanged, 5 min)

- **code:** `rebuild_with_session.py`
- **output:** `results/rebuild_session_log.txt`

Re-windows the raw city CSVs, importing `window_features()` from the original
`rebuild_windows_fixed.py` unchanged so features cannot drift. Adds `session` and `win_id`.
Verified byte-identical to the old cache: 7,094 windows, 228 journeys.

**Point to make:** the features are the original author's code, not a reimplementation.

---

## 2. PDSCH is a KAUST indicator (Khalil's point 3)

Not a script — two checks against the raw headers and the window cache:

- `kaust.csv` has `PDSCH scheduled throughput/PRB`; `jeddah.csv`, `kz.csv`, `mekkah.csv`
  do not.
- In the windows, `pdsch_mean` is non-zero in **40.8%** of KAUST and **0.0%** everywhere
  else. All 1,672 such windows are scooter, bike, jog or motorcycle — the four classes
  KAUST exclusively holds.

**Where it landed:** Sec. IV-B, as the worked example of the paper's own step-2 test.

---

## 3. GPS is an input (Khalil's point 2)

Raw logs open `Time, Longitude, Latitude, Velocity`, then the radio columns. Five of the 30
window features come from that block: `longitude_mean`, `latitude_mean`, `velocity_mean`,
`velocity_max`, `velocity_var`.

- **code:** `ladder_grouped.py --stage rf`
- **output:** `results/ladder_grouped_rf.csv`

| policy | n | 8 modes | 4 families |
|---|---|---|---|
| A all features | 30 | 0.636 | 0.748 |
| B no coordinates | 28 | 0.587 | 0.694 |
| C radio only | 25 | 0.479 | 0.684 |
| D radio, no campaign fields | 16 | 0.427 | 0.639 |

**Where it landed:** Sec. III names the positioning block; Sec. IV-B reports 0.639 against
0.748.

---

## 4. The duplicate rides (Khalil's cleaning step)

- **code:** `verify_duplicates.py`
- **output:** `results/confirmed_pairs.json`, `results/file_timespans.csv`

Run it in front of him; it takes under a minute and prints everything.

It takes the 17 files he flagged and confirms the pairing **two independent ways**:
window-feature distance within city+class, and overlap of the raw clock-time spans. The time
method alone over-detects, because the logs carry clock time with no date, so files whose own
span exceeds six hours are excluded as midnight-wrap artifacts. A pair counts only where both
methods agree: **15 of 17** (`car_kaust_045` has no usable time evidence).

Then it runs the paper's own splitter over its ten seeds:

```
unique ride pairs: 15, covering 1728 windows (24.4% of 7094)
mean 5.5 of 15 pairs separated per seed (37%)
at least one leak in 10 of 10 seeds
```

**Check he'll want:** `train`'s two journeys are **not** twins, so the class keeps a test
ride. 228 files collapse to 213 rides. KZ `walk` is still the only single-ride class.

---

## 5. The three ride-grouped re-runs

All three are the original script with `sid` changed from file to ride, nothing else. The
originals are not in this bundle — they are in `rebuilt_results/` — so show the change
directly:

```python
# old: split unit was the recording file
D['sid'] = D.city + "::" + D.session.astype(str)

# new: split unit is the ride, so twin recordings stay together
D['file'] = D.session.astype(str).str.replace('.csv', '', regex=False)
D['ride'] = D.file.map(lambda f: rm.get(f, f))     # rm = confirmed_pairs.json
D['sid']  = D.city + "::" + D.ride
```

| script | was | output |
|---|---|---|
| `causal_ridegrouped.py` | `causal_grouped.py` | `causal_ridegrouped.csv` |
| `fedobserve_ridegrouped.py` | `grouped_10seed.py` | `fedobserve_ridegrouped.csv` |
| `baselines_ridegrouped.py` | `baselines_grouped.py` | `baselines_ridegrouped.csv` |

Seeds, hyperparameters, arms and scoring are untouched, which is why the old and new
results are directly comparable.

### Isolation vs scarcity → Fig. 3

```
class    (A)     (B)     (C)    (C)-(A)   (B)-(C)
walk    0.858   0.707   0.847   -0.011    -0.140
car     0.630   0.289   0.577   -0.052    -0.288
bus     0.664   0.162   0.661   -0.004    -0.499
```

Negative in 9 of 9 treatment runs. The paper reports the gap as at least 0.129 in every
class.

### Main table → Table II

```
FedAvg + mask        0.0259 train   0.6061 macro
Rule + mask          0.1289 train   0.6187 macro
paired:  train 9/10 p=0.0020    macro 9/10 p=0.0322
```

### Baselines → Table III

```
FedProx + mask   0.0254 / 0.6035     vs ours: train p=0.0020, macro p=0.0322
FedRS + mask     0.1089 / 0.6173     vs ours: train p=0.1328, macro p=0.4229
```

---

## 6. The FedRS tie — have the answer ready

This is the one thing that got worse, and he will spot it.

Closing the ride leak helped FedRS more than it helped the rule, so the macro-F1 lead became
a tie. The paper reports it as a tie in both Sec. VI and Limitations, and reads it as support
for the diagnosis: restricting absent-class updates during training recovers most of what the
observation constraint recovers, which is what Sec. V predicts if the failure lies in how
absent classes are updated rather than in any one method.

The paper never claimed the method was novel — the aggregation weight is cwFedAvg and says
so.

---

## 7. Known gaps, say them before he finds them

1. **Mixed splits.** The main table, the baselines and the isolation experiment are
   ride-grouped. The random-forest ladder, the factorial and FedPer are file-grouped. The
   Protocol paragraph discloses this. Each is a comparison at a fixed split, so the contrast
   is unaffected; the absolute values carry the residual inflation.
2. **`train` is pseudo-replicated.** Two rides, so each seed tests one of them. Ten seeds
   vary the partition and initialization, not the test ride. The paper now says this directly
   after the headline numbers.
3. **Campaign fields are retained** in the federated experiments. Sec. IV-B reports what
   removing them costs on the random forest, but Secs. V–VI were not re-run on the 16
   radio-only features. That is the first thing to do for the journal version.
4. **`ablation_final.py:27`** in the main repo pools the scaler across cities while the text
   claims per-silo statistics. Pre-existing, unresolved.

---

## 8. If he asks why the numbers differed from his

They were never comparable. `study1_cross_region.ipynb` runs 3 classes (chance 0.33), 120 s
windows overlapping by half, 3 regions with KZ held out, 33 features with GPS, speed and time
excluded by construction, and drops 20 files. The paper runs 8 classes (chance 0.125), 60 s
non-overlapping windows, 4 clients including KZ.

His 0.532 and the paper's 0.60 are different tasks, not a discrepancy.

Worth saying: his section 8 result — every method falling to 0.38–0.41 on an unseen region —
independently replicates the paper's transfer-collapse finding on a different pipeline.
