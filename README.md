# Federated Network-Level Sensing — code and results

Analysis code and derived results for *Federated Network-Level Sensing: Challenges and
Lessons from Real-World Cellular Data*, submitted to IEEE ICC 2027.

**Carla Jaffal**, Khalil Alhaj, Kalamkas Zhagyparova, Nour Kouzayha,
Tareq Y. Al-Naffouri, Razane Tajeddine, Hadi Sarieddeen.
American University of Beirut · KAUST · Aalto University.

> Private archive. The paper is under review; the measurement corpus belongs to the
> KAUST group (Zhagyparova et al.) and is **not** redistributed here.

---

## What this repository is

The scripts and the aggregate result files behind every number in the paper. No raw
measurements: the training scripts read a windowed cache built from the KAUST corpus,
which is not included and is not ours to publish. What *is* here is everything needed to
check that the reported figures follow from the experiments that were run.

## Verify the main table in one second

```bash
cd code
python verify_table2.py
```

Reads `results/fedobserve_ridegrouped.csv` and `results/baselines_ridegrouped.csv`,
recomputes all six rows of Table II and all six Wilcoxon signed-rank p-values, asserts ten
seeds per method, and prints `ALL TABLE II VALUES REPRODUCED`. It touches no raw data and
retrains nothing.

| Method | `train` F1 | Macro F1 |
|---|---|---|
| FedAvg, no mask | 0.0160 | 0.5999 |
| FedAvg + mask | 0.0259 | 0.6061 |
| FedProx + mask | 0.0254 | 0.6035 |
| FedRS + mask | 0.1089 | 0.6173 |
| Rule, training only | 0.1112 | 0.6107 |
| **Rule + inference mask** | **0.1289** | **0.6187** |

## My contribution

First author. The work in this repository is mine except where noted below.

**The isolation-versus-scarcity experiment** (`code/causal_ridegrouped.py`). A controlled
intervention separating two effects that are normally confounded: a class confined to one
site is also usually rare. Holding the training-sample count fixed and varying only
concentration, class F1 falls by 0.140–0.499, against 0.004–0.052 for an equivalent
reduction in sample count — a gap of at least 0.129 in every class, negative in all nine
treatment runs, with a naturally single-site class left unchanged as a null control on the
harness.

**Two evaluation hazards and a reusable diagnostic** (`code/ladder_grouped.py`,
`code/physics2.py`, `code/physics4.py`). Window-level splitting inflates macro F1 by 0.211;
field-availability patterns alone predict transport mode at 0.595 accuracy within one site
before any measured value is read. The second is packaged as a four-step test and validated
against the Sussex-Huawei Locomotion dataset, where it correctly separates genuine
environmental signal from a recording artifact.

**The observation-proportional rule and inference-time mask**
(`code/fedobserve_ridegrouped.py`, `code/factorial.py`), with a 2³ factorial isolating what
each component actually contributes — including the finding that the class-wise aggregation
weight emphasized in prior work adds +0.0001 (p = 0.30) here.

**Correcting the evaluation protocol.** The project's earlier results used random
window-level splits. I re-ran the full experimental suite under journey- and ride-grouped
splits, and re-ran every federated baseline under the same protocol so the comparison stayed
like-for-like, rather than reporting the more flattering numbers.

**Reproducibility tooling** (`code/verify_table2.py`, `code/verify_duplicates.py`).

Khalil Alhaj identified the duplicate twin recordings (`results/confirmed_pairs.json`) and
reviewed the code. The measurement corpus was collected by the KAUST group and is described
in Zhagyparova et al., IEEE Int. Conf. Smart Mobility, 2023.

## Layout

```
code/      analysis and training scripts, exactly as run
results/   aggregate outputs: per-seed F1 scores, p-values, ablations
figures/   the figures in the paper
docs/      code guide and walkthrough written for the 29 Oct code review
```

`docs/CODE_GUIDE.md` maps each script to the paper number it produces.

## Running the training scripts

The scripts are preserved byte-identical to the versions that produced `results/`, which
means they still carry absolute paths to a local windowed cache:

```python
WIN = r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
OUT = r"C:\Users\user\Desktop\VIP 2\rebuilt_results"
```

To re-run any of them you need that cache, which is derived from the KAUST corpus and is not
included; point `WIN` at your own copy and `OUT` wherever you want the CSVs.
`verify_table2.py` is the exception — it uses relative paths and runs from a clone as-is.

## Status

Under review at IEEE ICC 2027. Not for redistribution while the paper and the underlying
corpus remain unpublished.
