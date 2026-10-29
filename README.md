# German Credit Risk Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://statistic---german-credit-project-ep15.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A complete machine-learning project for **German Credit Risk Classification**, designed to predict whether a loan applicant should be classified as **Good Credit** or **Bad Credit**.

The project focuses on **cost-sensitive classification** rather than accuracy alone. Missing a genuinely bad-credit applicant is treated as substantially more costly than incorrectly flagging a good-credit applicant.

> **Live application:**  
> https://statistic---german-credit-project-ep15.streamlit.app/

> **Repository:**  
> https://github.com/tnamgia76-debug/Statistic---German-Credit-Project

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Business Objective](#2-business-objective)
- [3. Dataset Structure](#3-dataset-structure)
- [4. Methodology](#4-methodology)
- [5. Preprocessing](#5-preprocessing)
- [6. Model Search Space](#6-model-search-space)
- [7. Threshold Optimization](#7-threshold-optimization)
- [8. Feature Selection Strategy](#8-feature-selection-strategy)
- [9. Nested Cross-Validation](#9-nested-cross-validation)
- [10. Final Model](#10-final-model)
- [11. Benchmark Results](#11-benchmark-results)
- [12. Competition Submission](#12-competition-submission)
- [13. Streamlit Application](#13-streamlit-application)
- [14. Repository Structure](#14-repository-structure)
- [15. Installation](#15-installation)
- [16. Running the Project](#16-running-the-project)
- [17. Deployment](#17-deployment)
- [18. Reproducibility and Leakage Control](#18-reproducibility-and-leakage-control)
- [19. Limitations](#19-limitations)
- [20. Future Improvements](#20-future-improvements)

---

# 1. Project Overview

The goal of this project is to build a reproducible binary classification pipeline for credit-risk prediction.

Each applicant is assigned an estimated probability of belonging to the **Bad Credit** class:

```text
bad_credit = 0  -> Good Credit
bad_credit = 1  -> Bad Credit
```

The final decision is produced by applying a classification threshold to the predicted probability:

```text
P(Bad) >= threshold  -> Bad Credit
P(Bad) < threshold   -> Good Credit
```

The project includes:

- Exploratory Data Analysis
- Reproducible preprocessing pipelines
- Multiple machine-learning algorithms
- Hyperparameter search
- Cost-sensitive threshold optimization
- Nested Cross-Validation
- Leakage-safe correlation / association-based feature selection
- Historical benchmark evaluation
- Competition prediction
- Streamlit deployment
- Single-applicant and batch credit-risk prediction

---

# 2. Business Objective

The primary evaluation objective is the asymmetric business cost:

Cost = 5 * FN + FP

where:

- **FN — False Negative:** an actual Bad applicant predicted as Good
- **FP — False Positive:** an actual Good applicant predicted as Bad

A False Negative is therefore considered **five times more costly** than a False Positive.

This means the project does **not** optimize accuracy alone.

A model may intentionally accept additional False Positives if doing so substantially reduces the number of Bad applicants incorrectly classified as Good.

### Confusion-matrix interpretation

| Actual | Predicted Good | Predicted Bad |
|---|---:|---:|
| Good | TN | FP |
| Bad | FN | TP |

Bad Credit (`1`) is treated as the positive class.

---

# 3. Dataset Structure

The original labeled dataset contains **800 observations**.

A fixed stratified split is used:

| Dataset | Good | Bad | Total | Purpose |
|---|---:|---:|---:|---|
| Development | 480 | 160 | 640 | Model development, Nested CV and final selection |
| Historical Benchmark | 120 | 40 | 160 | Post-selection diagnostic evaluation |
| Competition | Unknown | Unknown | 200 | Unlabeled final predictions |

The class distribution in both labeled subsets is approximately:

```text
Good: 75%
Bad : 25%
```

The 200 competition rows do not contain public target labels.

---

# 4. Methodology

The full modelling workflow is:

```text
800 labeled observations
        |
        +------------------------------+
        |                              |
640 Development                160 Historical Benchmark
        |                              |
        |                              |
Nested Cross-Validation               |
        |                              |
Evaluate selection procedure          |
        |                              |
Final search on 640                   |
        |                              |
Freeze final rule --------------------+
        |
Fit on 640
        |
Evaluate on benchmark
        |
NO further tuning
        |
Refit frozen Top_14 procedure on all 800
        |
Predict 200 competition rows
        |
Export model artifacts
        |
Deploy Streamlit application
```

The key principle is:

> **Inner CV selects. Outer CV evaluates.**

The historical benchmark is not used to choose a model, threshold, feature set or random seed.

---

# 5. Preprocessing

The project uses a `scikit-learn` Pipeline so that preprocessing is fitted only on the relevant training fold.

## Numerical features

The original numerical variables are:

```text
duration
amount
age
```

Pipeline:

```text
Median Imputation
        |
StandardScaler
```

## Categorical features

The remaining input variables are handled as categorical variables even when represented by integer codes.

Pipeline:

```text
Most-Frequent Imputation
        |
OneHotEncoder(
    handle_unknown="ignore"
)
```

This preprocessing design helps prevent data leakage because all learned preprocessing statistics are fitted only on training data.

---

# 6. Model Search Space

The project evaluates **9 model families** and **37 predefined configurations**.

| Model | Hyperparameters | Configurations |
|---|---|---:|
| Dummy | Baseline | 1 |
| Logistic Regression | `C ∈ {0.1, 1, 10}`, `class_weight ∈ {None, balanced}` | 6 |
| Gaussian Naive Bayes | `var_smoothing ∈ {1e-9, 1e-3}` | 2 |
| K-Nearest Neighbors | `n_neighbors ∈ {5, 15, 31}`, `weights ∈ {uniform, distance}` | 6 |
| Decision Tree | `max_depth ∈ {3, 6}`, `min_samples_leaf ∈ {5, 15}` | 4 |
| Random Forest | `max_depth ∈ {5, None}`, `min_samples_leaf ∈ {3, 10}` | 4 |
| Extra Trees | `max_depth ∈ {5, None}`, `min_samples_leaf ∈ {3, 10}` | 4 |
| RBF SVM | `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, 0.01}` | 6 |
| Gradient Boosting | `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {1, 2}` | 4 |

Total:

```text
37 model configurations
```

---

# 7. Threshold Optimization

The project does not automatically use the conventional threshold of `0.50`.

For each candidate model configuration:

1. Generate Out-of-Fold probabilities.
2. Test candidate thresholds.
3. Convert probability into Good/Bad predictions.
4. Compute:

\[
5 \times FN + FP
\]

5. Select the threshold with the lowest cost.

The threshold search used in the final notebook is:

```python
np.arange(0.01, 1.00, 0.01)
```

Therefore, threshold selection is treated as part of model selection.

---

# 8. Feature Selection Strategy

Feature selection is treated as a learned part of the modelling procedure rather than a one-time manual drop based on EDA.

The project uses target-association ranking:

- **Numerical predictors:** absolute Point-Biserial Correlation with `bad_credit`
- **Categorical predictors:** Cramér's V association with `bad_credit`

Because Point-Biserial correlation and Cramér's V are different association statistics, their numerical magnitudes are used here as a pragmatic ranking heuristic rather than interpreted as directly equivalent effect sizes.

Four predefined feature-count strategies are evaluated:

| Strategy | Number of Features |
|---|---:|
| `Top_20` | 20 |
| `Top_18` | 18 |
| `Top_16` | 16 |
| `Top_14` | 14 |

The important leakage-control rule is that the actual feature ranking is recalculated **only from the training portion of each CV fold**. Validation observations are never used to rank or select predictors.

Therefore, `Top_14` means "retain the 14 highest-ranked predictors learned from the current training data"; it does **not** refer to one hard-coded list of 14 variables across all folds.

### Nested-CV comparison

The saved project run produced:

```text
Full-20 baseline Nested-CV total cost : 391
Fold-wise feature-selection cost      : 372
```

Normalized by the 640 Development observations:

```text
Full-20 baseline       : 0.6109 cost / applicant
Feature-selection      : 0.5813 cost / applicant
```

or approximately:

```text
Full-20 baseline       : 61.09 cost / 100 applicants
Feature-selection      : 58.13 cost / 100 applicants
```

In this run, the leakage-safe feature-selection procedure improved the Outer-CV business-cost estimate relative to the Full-20 baseline.

# 9. Nested Cross-Validation

The Development set contains 640 observations.

## Outer CV

A fixed **5-fold Stratified Cross-Validation** structure is used.

Each iteration contains approximately:

```text
Outer Training   = 512 rows
Outer Validation = 128 rows
```

The Outer Validation fold is kept untouched while the complete prediction rule is selected.

## Inner CV

Each 512-row Outer Training set is evaluated using **3-fold Stratified Cross-Validation**.

For every Inner fold:

1. Association scores are calculated using only the Inner Training rows.
2. The requested `Top_k` feature strategy is applied.
3. Preprocessing is fitted only on those Inner Training rows.
4. The candidate model is trained.
5. Probabilities are generated for the Inner Validation rows.

For each candidate strategy/model configuration:

```text
3 Inner folds
      |
Fold-wise feature ranking
      |
OOF probabilities for all 512 rows
      |
Threshold optimization
      |
Inner business cost
```

The candidate with the lowest Inner-CV cost is selected.

The winning rule specifies:

```text
Feature-count strategy
+
Model family
+
Hyperparameters
+
Classification threshold
```

The winning feature-count strategy is then refitted on the complete 512-row Outer Training set to determine the predictor names for that Outer fold. A fresh model is fitted on those 512 observations and evaluated on the untouched 128-row Outer Validation fold.

This process is repeated across all five Outer folds.

The combined Outer predictions estimate the performance of the **complete selection procedure**, rather than the performance of one fixed model.

> **Inner CV selects. Outer CV evaluates.**

# 10. Final Model

After Nested Cross-Validation evaluates the selection procedure, the same 3-fold selection rule is applied to all 640 Development observations.

The final selected rule is:

| Component | Selected Value |
|---|---|
| Feature strategy | `Top_14` |
| Number of features | 14 |
| Model | Logistic Regression |
| `C` | `0.1` |
| `class_weight` | `balanced` |
| Classification threshold | `0.41` |
| Final-search selection cost | `325` |
| Final-search average cost | `0.5078` per applicant |

After `Top_14` was selected, association scores were recalculated on the complete 640-row Development dataset. The 14 Development-stage predictors were:

```text
status
credit_history
duration
purpose
savings
employment_duration
amount
housing
property
personal_status_sex
other_installment_plans
age
job
installment_rate
```

Important:

> The final-search cost of `325` is a **selection criterion**, not an unbiased estimate of generalization performance.

The procedure-level out-of-sample estimate comes from the Outer Nested CV:

```text
Outer-CV total cost            : 372
Average cost per applicant     : 0.58125
Cost per 100 applicants        : 58.125
```

# 11. Benchmark Results

Once the final rule was frozen, it was fitted on the 640 Development observations and evaluated on the 160-row Historical Benchmark.

The benchmark is diagnostic only and is not used to change the feature strategy, model family, hyperparameters, classification threshold, or random seed.

## Business-cost comparison

The updated frozen rule produced:

| Evaluation | N | Total Cost | Average Cost per Applicant | Cost per 100 Applicants |
|---|---:|---:|---:|---:|
| Nested CV Outer | 640 | 372 | **0.58125** | **58.125** |
| Historical Benchmark | 160 | 99 | **0.61875** | **61.875** |

Total cost should not be compared directly between these datasets because the sample sizes differ.

The normalized benchmark cost is therefore compared with the normalized Outer-CV estimate:

```text
Nested CV Outer       : 0.58125 cost / applicant
Historical Benchmark  : 0.61875 cost / applicant
Difference             : 0.03750 cost / applicant
```

Equivalently, the Historical Benchmark is approximately **3.75 cost units per 100 applicants** higher than the Nested-CV estimate.

The benchmark result is reasonably consistent with the Nested-CV estimate in terms of normalized business cost, while still showing variation on the separate 160-row diagnostic set.

## Probability Diagnostics

Probability quality was also examined using calibration analysis and the Brier Score.

| Evaluation | Model Brier | Simple Prior Brier |
|---|---:|---:|
| Nested OOF 640 | **0.1647** | 0.1875 |
| Historical Benchmark 160 | **0.1973** | 0.1875 |

The Nested-CV OOF probabilities outperform the simple prior-probability baseline, while the Historical Benchmark Brier Score is slightly worse than the baseline.

This suggests that the final decision rule remains useful for the project's asymmetric business-cost objective, but probability calibration is less stable on the Historical Benchmark.

Calibration and Brier diagnostics are descriptive only and do not alter the frozen model, feature strategy, or decision threshold.

No further tuning is performed after this benchmark evaluation.

# 12. Competition Submission

After the benchmark diagnostic step, the Development and Historical Benchmark datasets are combined:

```text
640 Development
+
160 Historical Benchmark
=
800 labeled observations
```

The already-selected modelling rule remains frozen:

```text
Feature strategy : Top_14
Model            : Logistic Regression
C                : 0.1
class_weight     : balanced
Threshold        : 0.41
```

The `Top_14` ranking procedure is then refitted on all 800 labeled observations. This may produce a different set of predictor names from the 14 predictors selected on the 640-row Development data, but the number of predictors and the modelling rule are not retuned.

A fresh final pipeline is fitted on all 800 labeled observations and used to generate predictions for the 200 unlabeled Competition rows.

### Final Deployment Feature Set

After refitting the selected `Top_14` association-ranking procedure on all 800 labeled observations, the deployment predictors are:

```text
status
credit_history
duration
purpose
savings
employment_duration
amount
housing
property
personal_status_sex
other_installment_plans
age
foreign_worker
installment_rate
```

The deployment feature set differs slightly from the 640-row Development feature set: `foreign_worker` enters the final Top-14 ranking while `job` is no longer retained.

This is expected because the feature-count strategy is frozen, while the exact predictor ranking is legitimately relearned using all labeled observations during the final deployment refit.


## Label conversion

The internal model target is:

```text
bad_credit = 0 -> Good
bad_credit = 1 -> Bad
```

The competition submission uses the original `kredit` encoding:

```text
kredit = 1 -> Good
kredit = 0 -> Bad
```

Therefore:

```python
competition_kredit = 1 - competition_pred
```

The generated submission contains:

```text
200 rows
2 columns: Id, kredit
0 missing values
73 predicted Good
127 predicted Bad
```

Competition leaderboard results are treated separately from the project's business-cost objective. They are not used to return to model, feature, or threshold tuning.

# 13. Streamlit Application

A Streamlit application is provided to demonstrate the trained model interactively.

## Live App

https://statistic---german-credit-project-ep15.streamlit.app/

## Application Features

### Single Applicant

A user can enter one applicant's information and receive:

- Good / Bad credit-risk classification
- Predicted probability of Bad Credit
- Final decision threshold
- Visual risk indicator

The deployed app applies the same frozen final threshold:

```text
0.41
```

### Batch Prediction

Users can upload a CSV file containing multiple applicants.

The application:

1. Checks required features
2. Generates `P(Bad)` for every row
3. Applies the frozen threshold
4. Returns Good / Bad predictions
5. Allows prediction results to be downloaded

### Model Information

The application displays:

- Final model family
- Number of selected features
- Decision threshold
- Business-cost definition
- Model hyperparameters
- Selected input variables

## Deployment Artifacts

The application loads the already-fitted model rather than retraining it.

```text
artifacts/
├── final_model.joblib
└── model_metadata.joblib
```

`final_model.joblib` contains the complete fitted preprocessing + model pipeline.

`model_metadata.joblib` contains deployment information such as:

```text
model_name
params
feature_strategy
n_features
features
threshold
numeric_features
categorical_features
category_values
numeric_defaults
```

---

# 14. Repository Structure

```text
Statistic---German-Credit-Project/
│
├── main_v2.ipynb
│   └── Main modelling, evaluation and export notebook
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│
├── artifacts/
│   ├── final_model.joblib
│   └── model_metadata.joblib
│
├── models/
│   ├── dummy.py
│   ├── logistic.py
│   ├── gaussian_nb.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── extra_trees.py
│   ├── rbf_svm.py
│   └── gradient_boosting.py
│
└── data_splits_colab/
    └── data_split/
        ├── development_640.csv
        ├── historical_benchmark_160.csv
        └── competition_200.csv
```

The exact repository contents may include additional project-support files.

---

# 15. Installation

## 15.1 Clone the repository

```bash
git clone https://github.com/tnamgia76-debug/Statistic---German-Credit-Project.git
cd Statistic---German-Credit-Project
```

## 15.2 Create a virtual environment

### Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 15.3 Install dependencies

```bash
python -m pip install -r requirements.txt
```

Tested package versions:

```text
Python          3.13
Streamlit       1.64.0
pandas          2.3.3
NumPy           2.2.3
scikit-learn    1.7.2
joblib          1.5.2
```

---

# 16. Running the Project

## Run the Streamlit app locally

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Run the modelling notebook

Open:

```text
main_v2.ipynb
```

Run the notebook from the repository root.

The notebook uses portable project paths:

```python
PROJECT_ROOT = Path.cwd().resolve()
```

This avoids machine-specific absolute paths such as:

```text
D:\...
C:\Users\...
```

and makes the project easier to reproduce on another machine.

---

# 17. Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
GitHub Repository
       |
Streamlit Community Cloud
       |
Install requirements.txt
       |
Run app.py
       |
Load saved model artifacts
       |
Public streamlit.app URL
```

Because the deployed app loads the saved artifacts, the full training notebook does not need to execute every time the application starts.

When the GitHub repository is updated, the Streamlit application can automatically redeploy from the latest repository version.

---

# 18. Reproducibility and Leakage Control

Several rules are used throughout the project to reduce data leakage and optimistic evaluation.

## Data split

The 800 labeled observations are split first into:

```text
640 Development
160 Historical Benchmark
```

using a fixed stratified split.

Feature selection, fitted preprocessing parameters, model selection, and threshold selection are learned without using the Historical Benchmark.

## Preprocessing

Scalers, imputers and encoders are fitted only on training folds.

Validation observations are transformed using rules learned from the corresponding training data.

## Feature selection

Target-aware feature selection is also fitted only on training data.

Within each CV training fold:

```text
Training rows
      |
Point-Biserial / Cramér's V ranking
      |
Top-k feature selection
      |
Preprocessing
      |
Model fitting
```

The corresponding validation rows do not participate in feature ranking.

The actual predictor names can therefore differ between folds even when the same strategy, such as `Top_14`, is evaluated.

## Threshold selection

Thresholds are selected using Inner OOF predictions.

Outer Validation labels are never used to select thresholds.

## Model selection

The Outer Validation data is used only after the complete Inner-CV rule has been selected.

## Historical benchmark

The 160-row Historical Benchmark is used only after the final Development-stage rule has been frozen.

The benchmark result does not trigger:

- feature-strategy changes
- model changes
- parameter changes
- threshold changes
- random-seed changes

## Final 800-row refit

After benchmark evaluation is complete, the 640 Development and 160 Benchmark rows are combined.

The already-selected `Top_14` strategy is refitted on all 800 labeled observations to determine the final deployment predictors.

This is a final refit, not a new model-selection step: `Top_14`, Logistic Regression, `C = 0.1`, `class_weight = balanced`, and threshold `0.41` remain frozen.

## Competition data

Competition labels are not available to the training pipeline.

Competition or leaderboard results are treated as external evaluation information rather than as part of the model-selection loop.

# 19. Limitations

This project is an academic machine-learning demonstration and should not be interpreted as a production banking decision system.

Important limitations include:

- The dataset is relatively small.
- The dataset is historical and may not represent modern lending populations.
- The `5:1` FN-to-FP cost ratio is a benchmark cost assumption rather than a measured modern bank loss function.
- Some variables may be sensitive or act as proxies for sensitive characteristics.
- The model has not been certified for fair-lending compliance.
- Predicted probabilities should not automatically be interpreted as calibrated real-world default probabilities.
- The historical benchmark is diagnostic rather than a fully new blind test.
- The Streamlit app is intended for educational demonstration and risk screening only.

> **The application should not be used to make real lending decisions without substantial additional validation, governance and fairness review.**

---

# 20. Future Improvements

Possible extensions include:

- Probability calibration
- SHAP-based model interpretation
- Applicant-level explanations
- More extensive hyperparameter search
- Additional cost-sensitive algorithms
- Cost-aware ensemble models
- Model calibration monitoring
- Fairness analysis
- Drift detection
- More user-friendly categorical labels in the Streamlit interface
- Authentication and role-based access
- Database integration
- Prediction logging
- REST API deployment
- Docker containerization
- Automated CI/CD testing

---

## Academic Demonstration

This repository was developed as a statistics / machine-learning project focused on understanding:

- classification
- confusion matrices
- asymmetric business cost
- probability thresholds
- preprocessing pipelines
- model comparison
- Nested Cross-Validation
- feature selection
- data leakage
- deployment

The primary goal is not simply to obtain the highest leaderboard score, but to construct a transparent and defensible modelling workflow from raw data to deployed application.
