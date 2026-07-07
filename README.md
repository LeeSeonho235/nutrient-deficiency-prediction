# Blood Nutrient Deficiency Prediction

Multi-label DNN that predicts 9 nutrient deficiencies from routine blood test results. Built with CDC NHANES 2017-2018 public health data (~9,000 samples). Achieved **Val AUC 0.9818**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](여기에_Colab_링크_넣기)

## Problem

Micronutrient deficiencies are a leading cause of chronic disease, but specialized testing is expensive and inaccessible for most people. This model screens for 9 deficiencies using only standard blood panel values that are already collected in routine checkups.

## Target Nutrients

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

## Model

Input: 9 blood markers + age + gender → Output: 9 deficiency probabilities

```
Dense(128, relu) → BatchNorm → Dropout(0.4)
Dense(64, relu)  → BatchNorm → Dropout(0.3)
Dense(32, relu)  → BatchNorm → Dropout(0.2)
Dense(9, sigmoid)  → multi-label output
```

- Loss: binary crossentropy
- EarlyStopping on val_auc (patience=20)
- ReduceLROnPlateau on val_loss
- SMOTE for class imbalance
- Optimal threshold search (0.2–0.7 range)

## Results

- **Val AUC: 0.9818**
- Hamming Loss: low (most labels predicted correctly)
- Inference function accepts partial blood values (missing values auto-imputed)

## Data Pipeline

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

## Usage

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
    'RIAGENDR':   1     # Gender (1=Male, 2=Female)
})
```

Unknown values can be set to `None` and will be auto-imputed.

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Data:** CDC NHANES 2017-2018 (public XPT files)
- **Preprocessing:** pandas, scikit-learn (StandardScaler, imputation)
- **Class Balancing:** imbalanced-learn (SMOTE)
- **Environment:** Google Colab (T4 GPU)

## Team

Built as a university deep learning course project at Namseoul University.

- Seonho Lee
- Sung-han Park

## License

MIT
