# Code and results behind the 23 Oct email

Every claim in the email, mapped to the script that produced it and the saved output.
All runs use the rebuilt windows with journey identifiers (`rebuilt_cache/*_60s_sess.csv`).
Scripts expect the repo layout: they read from `VIP 2/rebuilt_cache/` and write to
`VIP 2/rebuilt_results/`.

---

## 0. The rebuild

> *"I rebuilt the dataset from the raw files, and re-ran everything. The rebuild reproduces
> the original feature values exactly."*

- **code:** `code/rebuild_with_session.py`
- **output:** `results/rebuild_session_log.txt`

Re-windows the raw city CSVs, importing `window_features()` from the original
`rebuild_windows_fixed.py` unchanged so the features cannot drift. Adds two columns the old
cache lacked: `session` (the recording journey) and `win_id`.

```
jeddah   old 1304  new 1304  counts_match=True  features_identical=True   21 journeys
kaust    old 4099  new 4099  counts_match=True  features_identical=True  152 journeys
kz       old  201  new  201  counts_match=True  features_identical=True    9 journeys
mekkah   old 1490  new 1490  counts_match=True  features_identical=True   46 journeys
TOTAL    old 7094  new 7094
```

228 journeys total. `train` is 2 journeys; KZ `walk` is 1.

---

## 1. Withholding GPS improves cross-city transfer

> *"a model given velocity and radio statistics but no coordinates scores 0.113 macro-F1
> higher on mode families... KAUST goes from 0.11 to 0.32 and Mekkah from 0.19 to 0.44"*

- **code:** `code/ladder_grouped.py --stage rf`
- **output:** `results/ladder_grouped_rf.csv`

Four feature policies x three evaluation protocols (random window split, journey-grouped,
leave-one-city-out), on 8-mode and 4-family targets.

4 families, leave-one-city-out:

| policy | macro F1 | per city (jedd / kaus / kz / mekk) |
|---|---|---|
| A all features | 0.282 | 0.42 / 0.11 / 0.41 / 0.19 |
| B no coordinates | **0.395** | 0.43 / 0.32 / 0.38 / 0.44 |

**Read this before quoting it.** The +0.113 is a **random forest**, used deliberately to
separate the effect of the feature policy from the effect of the training algorithm. The
federated replication is smaller:

- **code:** `code/fed_family_loco.py`
- **output:** `results/fed_family_loco.csv`

| policy | FedAvg | FedObserve |
|---|---|---|
| A all features | 0.3247 | 0.3222 |
| B no coordinates | 0.3671 | 0.3685 |
| gain | **+0.042** | **+0.046** |

Both methods gain by similar amounts, which is the useful part: the effect belongs to the
data rather than to one algorithm. But the defensible federated number is ~0.042, not 0.113,
and the gain is concentrated in KZ. A ten-seed version with the per-city breakdown is still
outstanding.

---

## 2. The effect is specific to new sites

> *"Withholding coordinates helps when the test city is unseen and slightly hurts when
> testing within the same cities"*

- **code:** `code/ladder_grouped.py --stage rf`
- **output:** `results/ladder_grouped_rf.csv`

4 families: journey-grouped 0.748 (all features) against 0.694 (no coordinates), so −0.054
within-site; leave-one-city-out +0.113 cross-site. Opposite signs, which is the signature of
a location shortcut.

---

## 3. The handover estimator, and where it fails

> *"estimating a bus at 21.8 km/h against 21.8 true, and is three times too high for
> pedestrians, 10.8 against 3.8... implied cell diameter differs per city, 116 to 180 m"*

- **code:** `code/physics4.py` (sections H and I)
- also: `code/physical_norm.py`, `results/physical_norm.csv`

Fits `v_hat = 3.6 * H * D` with one cell diameter `D` per city, estimated from that city's
own vehicular windows. Jeddah 180 m, KAUST 133, KZ 116, Mekkah 121.

Median estimated against median true velocity: bus 21.8/21.8, motorcycle 23.9/26.0,
walk 10.8/3.8, jog 16.0/8.5. Pearson r with true velocity is 0.330 pooled.

`physical_norm.py` additionally tests whether normalising handover features by `D` improves
cross-site transfer. It does not: gains are small and appear within-city as well as
cross-city, the signature of added scale information rather than a removed site-specific
nuisance.

---

## 4. FedPer scores higher per client

> *"Khalil, on FedPer, I think I understand why it scores higher per client"*

- **code:** `code/masked_inference.py`, `code/local_bias.py`
- **output:** `results/masked_inference.csv`, `results/local_bias.csv`

Journey-grouped, 5 seeds, paired on identical splits and initial weights:

| method | macro F1 | train F1 |
|---|---|---|
| FedAvg | 0.5909 | 0.0061 |
| FedAvg + inference mask | 0.5961 | 0.0065 |
| FedObserve | 0.6039 | 0.0942 |
| FedObserve + inference mask | 0.6091 | 0.1117 |
| + per-client output bias | 0.6138 | 0.1340 |
| **FedPer** | **0.6580** | **0.4559** |

The hypothesis in the email — that FedPer wins because its local head never considers
impossible classes — was **tested and is mostly wrong**. Masking those classes in the shared
model closes only about a quarter of the gap, and FedPer still leads on the classes each
city *does* hold. A per-client output bias closes none of it (2/5 seeds, p = 0.41). The
remaining difference is in the classifier's decision boundaries, which only a local
classifier provides.

Masking is still worth keeping: free, no retraining, and it uses only the availability
indicator already disclosed.

---

## 5. Supporting runs not named in the email

| what | code | output |
|---|---|---|
| isolation vs scarcity, journey-grouped | `code/causal_grouped.py` | `results/causal_grouped.csv` |
| main comparison, 10 seeds, Wilcoxon | `code/grouped_10seed.py` | `results/grouped_10seed.csv` |
| federated disclosure ladder, 8 modes | `code/ladder_grouped.py --stage fed` | `results/ladder_grouped_fed.csv` |
| campaign-availability analysis | `code/physics2.py` (section A) | printed output |
| local-only baseline, each city alone | `code/local_only.py` | `results/local_only.csv` |
| FedPer vs FedObserve, random split | `code/fedper_proper.py` | `results/fedper_proper.csv` |

**Isolation versus scarcity** is the strongest result in the set. Journey-grouped, three
seeds, counts matched exactly between conditions B and C:

| class | A natural | B isolated | C same count, distributed | C−A | B−C |
|---|---|---|---|---|---|
| walk | 0.893 | 0.744 | 0.886 | −0.007 | **−0.142** |
| car | 0.763 | 0.368 | 0.731 | −0.032 | **−0.363** |
| bus | 0.696 | 0.133 | 0.616 | −0.080 | **−0.483** |

Removing samples costs 0.007–0.080. Concentrating the same number at one client costs
0.142–0.483. Under the old random window split the same measurements were −0.117, −0.190
and −0.221, so journey leakage was roughly halving the effect.

**Main comparison**, journey-grouped, 10 seeds, paired:

| method | macro F1 | train F1 |
|---|---|---|
| FedAvg | 0.6010 ± 0.0399 | 0.0065 ± 0.0137 |
| FedObserve | 0.6141 ± 0.0465 | 0.1186 ± 0.0719 |
| FedObserve + mask | 0.6214 ± 0.0479 | 0.1414 ± 0.0873 |

One-sided Wilcoxon against FedAvg: macro F1 p = 0.0244 without the mask, p = 0.0020 with it;
train F1 p = 0.0020 both.

---

## 6. Results completed after the 23 Oct email

All journey-grouped, ten seeds, paired on identical splits and initial weights. The inference
mask is granted to every method, since it uses only the availability indicator and needs no
retraining, so nothing here is masked against unmasked.

**Main comparison** -- `code/baselines_grouped.py`, `code/grouped_10seed.py`,
`code/main_table_10seed.py`; outputs `results/baselines_grouped.csv`,
`results/grouped_10seed.csv`, `results/main_table_10seed.csv`.

| method | `train` F1 | macro F1 |
|---|---|---|
| FedAvg | 0.0071 | 0.6052 |
| FedProx | 0.0103 | 0.6049 |
| FedRS | 0.0764 | 0.6109 |
| **this configuration** | **0.1414** | **0.6214** |

Paired one-sided Wilcoxon: against FedAvg p = 0.0020 (`train`) and 0.0098 (macro); against
FedProx p = 0.0020 and 0.0068; against FedRS p = 0.074 and 0.065, i.e. a lead on both metrics
that does not reach significance at ten seeds.

**Factorial ablation** -- `code/factorial.py`, `results/factorial.csv`. 2^3 over the three
training-time components, 40 matched pairs. Main effects on macro F1: class-weighted loss
+0.016 (p < 0.0001), absent-class restoration +0.001 (p = 0.38), class-wise aggregation
+0.0001 (p = 0.30), inference mask +0.005 (40/40 wins). On `train` all four help, the loss
weighting by +0.077 and class-wise aggregation by +0.023. cwFedAvg-only scores 0.5914, below
FedAvg's 0.5961.

**Disclosure at ten seeds** -- `results/fed_family_loco.csv`. Withholding coordinates gains
+0.047 (FedAvg) and +0.059 (this configuration). The gain is concentrated at KZ, 0.20 to 0.39
and 0.16 to 0.39; the other three cities move by less than 0.01, which is the signature of a
location shortcut rather than a general gain.

**Isolation vs scarcity** -- `results/causal_grouped.csv`. With sample counts matched exactly,
concentration costs 0.142 / 0.363 / 0.483 class F1 for walk / car / bus, against 0.007 /
0.032 / 0.080 for removing the same number of samples. Under the old random split the same
measurements were 0.117 / 0.190 / 0.221, so journey leakage was roughly halving the effect.

**Figures** -- `code/make_figures.py`, output in `figures_measurement/`.

---

## Known gaps

1. **The lead over FedRS is not significant** at ten seeds (p = 0.065 macro, p = 0.074 on
   `train`). The paper reports it as unresolved rather than established.
2. **`ablation_final.py:27`** in the main repo fits one `StandardScaler` across all cities
   pooled, while the paper text claims per-silo statistics. One of the two needs changing.
3. **The disclosure result rests on one of four held-out cities.** The mechanism explains why,
   but the corpus contains one such site.
4. **BalanceFL, SCAFFOLD and the two rarity-weighted rules** from the earlier random-split
   table have not been re-run under journey-grouped splits.
5. **`train` is 112 windows from two journeys**, so every figure involving it rests on a
   single unseen trip.

---

# Addendum, 29 Oct: response to Khalil's code review

Three points were raised on 28 Oct. All three are confirmed. This section gives the
evidence for each and the script that reproduces it.

## A. PDSCH is not a general field, it is a KAUST indicator

> *"there is a field called PDSCH you are mentioning in the paper, which I couldn't find in
> the Cellmob dataset"*

Correct. `PDSCH scheduled throughput/PRB` exists in `kaust.csv` only (column 62) and is
absent from `jeddah.csv`, `kz.csv` and `mekkah.csv`. In the windows:

| city | pdsch_mean nonzero |
|---|---|
| jeddah | 0.0% |
| kaust | 40.8% |
| kz | 0.0% |
| mekkah | 0.0% |

Consequences, measured on the window cache:

- `pdsch_mean > 0` implies the window is KAUST, in 1,672 of 1,672 cases
- those 1,672 windows are scooter (753), bike (537), jog (262), motorcycle (120), and
  nothing else

So a single field hands any model a perfect partition isolating the four KAUST-exclusive
classes. This is an instance of the hazard the paper's Section IV already describes, and
step 2 of the test given there is what detects it. PDSCH is being written up as the worked
example for that section, and the feature description is being corrected.

## B. GPS is an input to the main table

> *"I was also concerned about if you are using GPS as input to your pipeline, which defeats
> the whole premise of the paper"*

Correct. The raw logs begin `Time, Longitude, Latitude, Velocity`, then the radio columns.
Five of the thirty window features come from that GPS block: `longitude_mean`,
`latitude_mean`, `velocity_mean`, `velocity_max`, `velocity_var`.

The cost of removing them was already measured for the disclosure section.
Journey-grouped, random forest, from `results/ladder_grouped_rf.csv`:

| feature policy | n | 8 modes | 4 families |
|---|---|---|---|
| A all features | 30 | 0.636 | 0.748 |
| B no coordinates | 28 | 0.587 | 0.694 |
| C radio only (no GPS, no velocity) | 25 | 0.479 | 0.684 |
| D radio, campaign fields removed too | 16 | **0.427** | **0.639** |

Row D is the honest figure for the paper's stated premise and will be reported alongside
the main table.

## C. The duplicate recordings, and the leak they cause

The cleaning step in `study1_cross_region.ipynb` drops 17 recordings made by a second or
third phone on a ride already in the corpus. All 17 are present in this corpus.

- **code:** `code/verify_duplicates.py`
- **output:** `results/confirmed_pairs.json`, `results/file_timespans.csv`

The pairing is confirmed two independent ways: window-feature distance within city and
class, and overlap of the raw clock-time spans. The time method alone over-detects, because
the logs carry clock time with no date, so files whose own span exceeds six hours are a
midnight-wrap artefact and are excluded from the time evidence. A pair is reported only
where both methods agree, which leaves **15 of the 17** (`car_kaust_045` has no usable time
evidence, its own span wraps midnight).

Running the paper's splitter over its ten seeds:

```
unique ride pairs: 15, covering 1728 windows (24.4% of 7094)
mean 5.5 of 15 pairs separated per seed (37%)
at least one leak in 10 of 10 seeds
```

So grouping by `source_file` is not sufficient, because a twin is treated as an independent
journey. The fix is to group by ride.

- **code:** `code/main_table_ridegrouped.py`
- **output:** `results/main_table_ridegrouped.csv`

Identical to `main_table_10seed.py` in seeds, hyperparameters, arms and scoring; the only
change is that confirmed twins merge into one split unit. The two tables are therefore
directly comparable, and the script prints the difference.

## What this does and does not move

The comparisons in the paper hold the feature set fixed across the conditions being
compared, so they are unaffected by A and B:

- isolation versus scarcity, conditions B and C see identical features
- the journey-leakage result holds on any feature policy: 0.652 to 0.479 on radio only,
  0.558 to 0.427 once campaign fields are removed as well

What changes is the absolute values and the feature description, both of which are being
corrected, plus whatever the ride-grouped re-run costs.

## Reproducing the addendum

```
python code/verify_duplicates.py          # section C, pairs and leak rate
python code/main_table_ridegrouped.py     # the ride-grouped main table
```
