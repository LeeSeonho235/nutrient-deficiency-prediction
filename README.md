# Multi-label Prediction of Eight Nutrient Deficiencies from Routine Blood Tests (NHANES)

A multi-label deep learning model that screens for eight nutrient deficiencies using only routine health-checkup variables, plus an analysis of how **target leakage** inflates reported performance.

> **Published:** Selected as an excellent paper at the **2026 KIICE Fall Conference** (Korea Institute of Information and Communication Engineering).
> *Multi-label Prediction of Eight Nutrient Deficiencies Using NHANES Data: A Routine Blood Test Approach and Target Leakage Analysis*

## Key results

| Setting | Macro AUROC (DNN) |
|---|---|
| Leaky (label-defining markers given as inputs) | **0.976** |
| Leak-free (routine checkup variables only) | **0.797** |

- A single masked multi-label DNN predicts 8 deficiencies at once from **48 routine variables** (CBC, biochemistry panel, demographics, BMI, exam season) for **15,361 adults** (NHANES 2013-2018).
- Giving the model the markers that define the labels makes every label look almost perfectly predictable. Removing them drops the score by about 0.18, which is the real level of difficulty.
- **Iron deficiency** can be screened from routine tests (AUROC 0.941, sensitivity 0.861, specificity 0.887). Hemoglobin and RDW are the most important features, matching the clinical picture of iron-deficiency anemia.
- **Vitamin B12 (0.598)** and **zinc (0.670)** are hard to screen from routine blood tests alone.

### Leak-free test AUROC by label (5-seed mean)

| Label | DNN | LR | GBM | Test positives (mean) |
|---|---|---|---|---|
| Vitamin D (25(OH)D < 50 nmol/L) | 0.767 | 0.764 | 0.764 | 712 |
| Iron (ferritin < 15 ng/mL) | 0.941 | 0.946 | 0.940 | 83 |
| Folate (RBC < 305 nmol/L) | 0.755 | 0.676 | 0.552 | 3 |
| Calcium (< 8.5 mg/dL) | 0.916 | 0.861 | 0.784 | 15 |
| Vitamin B12 (< 200 pg/mL) | 0.598 | 0.668 | 0.618 | 20 |
| Zinc (< 74 M / < 70 F ug/dL) | 0.670 | 0.720 | 0.706 | 153 |
| Vitamin A (retinol < 20 ug/dL) | 0.925 | 0.830 | 0.861 | 3 |
| Serum iron (< 60 ug/dL) | 0.801 | 0.808 | 0.811 | 571 |

Macro AUROC (leak-free): DNN 0.797, LR 0.788, GBM 0.760 (mean of per-label means; averaging per-seed macro scores gives 0.798 for the DNN).
The DNN is not clearly better than logistic regression label by label. Its advantage is practical: one model handles all 8 labels at once. Folate and vitamin A have only about 3 positive test cases per split, so their scores are unstable.

## Method

- **Data:** CDC NHANES 2013-14, 2015-16, 2017-18. Adults aged 20+ with CBC and biochemistry results (n = 15,361). Not every marker was measured in every cycle, so the three cycles are merged.
- **Labels:** 8 deficiencies from international cutoffs. A person whose marker was **not measured** gets a missing label instead of being treated as normal. Magnesium is excluded because the public data has no serum measurement.
- **Inputs (48):** CBC (15 items), biochemistry (calcium and iron excluded), age, sex, race/ethnicity, income-poverty ratio, BMI, exam season. Only columns available in all cycles are used.
- **Model:** masked multi-label DNN, 48 -> 128 -> 64 -> 32 -> 8 (ReLU, batch norm, dropout 0.4 / 0.3 / 0.2, sigmoid outputs).
  - Masked binary cross-entropy: only measured labels contribute to the loss.
  - Positive-class weight `sqrt((1 - p) / p)`, clipped to 1-20.
- **Baselines:** per-label logistic regression and gradient boosting.
- **Evaluation:** 70/15/15 train/validation/test split, 5 random seeds, AUROC and AUPRC. Decision thresholds are chosen on the validation set (Youden index), not the test set.
- **Leaky condition:** the same pipeline with the 8 label-defining markers added to the inputs, to measure how much leakage inflates performance.

## Run it

Open the notebook in Google Colab (CPU is enough) and choose **Runtime > Run all**. It downloads the NHANES files directly from the CDC site, runs all experiments, and saves tables and figures in `results/` (also downloaded as `results.zip`).

- `NHANES_nutrient_deficiency_v2.ipynb`: final study (this README).
- `v1/`: earlier preliminary experiment (see below).

## About the earlier version (v1)

The first version of this project (NHANES 2017-2018 only, 9 deficiencies) had target leakage: the markers that define the labels were used as inputs, and three markers (magnesium, zinc, vitamin B12) were filled with synthetic values. Its reported validation AUC of about 0.98 is therefore not a realistic estimate. The study was redone as described above, and the leakage analysis became the main contribution of the paper.

| | v1 | v2 (this study) |
|---|---|---|
| Inputs | The markers that define the labels | Routine checkup variables only |
| Labels | 6 measured + 3 synthetic | 8, measured values only |
| Vitamin D column | D2 fraction only | Total 25(OH)D |
| Missing labels | Treated as normal | Masked out of loss and evaluation |
| Threshold | Searched on the test set | Chosen on the validation set |
| Repeats | 1 run | 5 seeds, mean and std |

## Limitations

- No external validation yet (planned: Korean KNHANES).
- Measurement cohorts differ by cycle: for example, ferritin was measured only in some groups, and many of those are women.
- Folate and vitamin A have very few positive cases.
- Serum calcium reflects homeostasis more than intake.

## Tech stack

Python, TensorFlow / Keras, scikit-learn, pandas, matplotlib. Data: CDC NHANES (public XPT files).

## Team

University project at Namseoul University.

- Seonho Lee
- Seonghan Park
- Advisor: Prof. SungAe Kim (corresponding author)

## License

MIT
