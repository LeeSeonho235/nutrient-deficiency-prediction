# Blood Nutrient Deficiency Prediction

Multi-label deep learning on routine blood-test data from CDC NHANES, and a study of how **data leakage** inflates performance.

> **Published:** Selected as an excellent paper at the **2026 KIICE Fall Conference** (Korea Institute of Information and Communication Engineering).
> *Multi-label Prediction of Eight Nutrient Deficiencies Using NHANES Data: A Routine Blood Test Approach and Target Leakage Analysis*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w35uKNSfijqnH-XQPmq4ppEVWhxe2mAb?usp=sharing)

## About this repository

This repository contains the **preliminary (v1) experiment** that came before the paper. While writing the paper we found that v1 had a target leakage problem, so its headline score is not a realistic estimate. The paper redesigns the study and reports the leakage-free results. The final-study code is not included here.

| | This repository (v1, preliminary) | Paper (final study) |
|---|---|---|
| Deficiencies | 9 | 8 (magnesium excluded: no public serum values) |
| Data | NHANES 2017-2018, ~9,000 samples | NHANES 2013-2018 (3 cycles), 15,361 adults |
| Inputs | 9 blood markers + age + sex, including the markers that define the labels | 48 routine-checkup variables, label-defining markers removed |
| Missing labels | Synthetic columns for Mg, Zn, B12 | Treated as missing and masked out of the loss |
| Result | Val AUC about 0.98 (inflated by leakage) | Macro AUROC 0.976 with leakage, **0.797 without leakage** |

## Key findings of the paper

- A masked multi-label DNN predicts 8 nutrient deficiencies at once from 48 routine blood-test and demographic variables (15,361 adults, NHANES 2013-2018).
- **Leakage inflates results:** when the markers that define the labels are given as inputs, macro AUROC is 0.976. After removing them it drops to **0.797**.
- **Iron deficiency** (ferritin < 15 ng/mL) can be screened from routine tests: AUROC 0.941, with hemoglobin and RDW as the most important features.
- Vitamin B12 (0.598) and zinc (0.670) are hard to screen from routine blood tests alone.
- Limitations: no external validation yet; folate and vitamin A have very few positive test cases.

## v1 experiment (this repository)

### Problem

Micronutrient deficiencies are common, but specialized blood tests are often not part of routine checkups. The goal was to screen for deficiencies using standard blood panel values.

### Target nutrients (v1)

| Nutrient | Blood Marker | Deficiency Threshold |
|----------|-------------|---------------------|
| Vitamin D | LBXVD2MS | < 20 ng/mL |
| Iron (Ferritin) | LBXFERR | Male < 24, Female < 11 ng/mL |
| Folate | LBXRBF | < 140 ng/mL |
| Calcium | LBXSCA | < 8.5 mg/dL |
| Magnesium | LBXSMG | < 1.7 mg/dL |
| Zinc | LBXSZN | < 70 µg/dL |
| Vitamin A | LBXVTC | < 30 µg/dL |
| Vitamin B12 | LBDB12SI | < 200 pmol/L |
| Serum Iron | LBXSFE | < 60 µg/dL |

### Model (v1)

Input: 9 blood markers + age + sex. Output: 9 deficiency probabilities.

```
Dense(128, relu) → BatchNorm → Dropout(0.4)
Dense(64, relu)  → BatchNorm → Dropout(0.3)
Dense(32, relu)  → BatchNorm → Dropout(0.2)
Dense(9, sigmoid)  → multi-label output
```

- Loss: binary crossentropy
- EarlyStopping on val_auc, ReduceLROnPlateau on val_loss
- SMOTE for class imbalance, threshold search in the 0.2-0.7 range

### Results (v1)

- Val AUC about 0.98. **This number is inflated:** the inputs contain the markers used to define the labels, so the model can learn the threshold rules directly. The paper quantifies this effect.

<img width="1238" height="500" alt="v1 result 1" src="https://github.com/user-attachments/assets/72eac3dc-29b0-45b7-82da-492e0b4514f3" />
<img width="1194" height="512" alt="v1 result 2" src="https://github.com/user-attachments/assets/5d3f2559-c8cf-4bbb-8b3a-cd69f27b230a" />
<img width="505" height="305" alt="v1 result 3" src="https://github.com/user-attachments/assets/d4eb1471-4c69-47ad-8f05-a154ab10f575" />
<img width="469" height="131" alt="v1 result 4" src="https://github.com/user-attachments/assets/d42154d2-50ce-43fe-b619-f69f80a97343" />

### Data pipeline (v1)

```
CDC NHANES 2017-2018 XPT files (6 files, direct download)
    │
    ├── Merge on SEQN (patient ID)
    ├── Synthetic columns for Mg, Zn, B12 (based on NHANES 2011-2014 statistics)
    ├── Clinical threshold labeling (9 binary targets)
    ├── SMOTE oversampling for rare deficiencies
    ├── StandardScaler normalization
    │
    └── Train / Val / Test split
```

The synthetic Mg, Zn, and B12 columns are placeholders and are not real measurements. The final study does not use them.

### Usage (v1)

```python
predict_deficiency({
    'LBXVD2MS': 15.0,   # Vitamin D
    'LBXFERR':  80.0,   # Ferritin
    'LBXRBF':  350.0,   # Folate
    'LBXSCA':    9.5,   # Calcium
    'LBXSMG':    2.0,   # Magnesium
    'LBXSZN':   85.0,   # Zinc
    'LBXVTC':   55.0,   # Vitamin A
    'LBDB12SI': 500.0,  # Vitamin B12
    'LBXSFE':  100.0,   # Serum Iron
    'RIDAGEYR':  30,    # Age
    'RIAGENDR':   1     # Sex (1=Male, 2=Female)
})
```

Unknown values can be set to `None` and will be auto-imputed.

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Data:** CDC NHANES (public XPT files)
- **Preprocessing:** pandas, scikit-learn (StandardScaler, imputation)
- **Class balancing (v1):** imbalanced-learn (SMOTE)
- **Environment:** Google Colab (T4 GPU)

## Team

University deep learning course project at Namseoul University.

- Seonho Lee
- Seonghan Park
- Advisor: Prof. SungAe Kim (corresponding author)

## License

MIT
