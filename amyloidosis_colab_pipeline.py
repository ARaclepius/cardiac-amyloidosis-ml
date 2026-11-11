# ======================================================================
# CARDIAC AMYLOIDOSIS ML STUDY — GOOGLE COLAB PIPELINE
#
# Proposal-aligned analysis:
#   - Primary objective: develop echocardiography-based ML models for
#     diagnosis of cardiac amyloidosis.
#   - Secondary outcomes: sensitivity, specificity, accuracy, PPV, NPV,
#     and comparison of model performance.
#
# ANALYSIS PLAN
#   - 3 feature sets: all 26, echo 19, clinical 7
#   - 4 classifiers: Random Forest, XGBoost, RBF-SVM, MLP
#   - 12 model/feature-set combinations
#   - 80:20 stratified hold-out + 5-fold stratified CV within training data
#   - No feature selection, LASSO, PCA, SMOTE, nested CV, Harrell bootstrap
#     or overfitting tests
#   - Hyperparameters tuned exactly like the Jupyter 'BASE' experiment:
#     3-fold inner CV on the 80% development set + one-standard-error rule
#   - Metrics: ROC-AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1
#   - SVM AUC uses decision_function; SVM labels use estimator.predict()
#   - Post-one-hot StandardScaler matches the original Jupyter preprocessing
#   - Demographic/baseline characteristics by amyloidosis status
#   - One combined ROC plot for all 12 final models
#   - Four model-specific metric comparison bar plots
#   - TreeSHAP for Random Forest and XGBoost
#   - PermutationSHAP for SVM and MLP
#
# Google Colab: upload final_data.xlsx when prompted or set DATA_PATH to
# a file in mounted Google Drive.
# ======================================================================

import os
import sys
import json
import re
import math
import warnings
import zipfile
import subprocess
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import joblib

from scipy.stats import (
    shapiro,
    ttest_ind,
    mannwhitneyu,
    chi2_contingency,
    fisher_exact,
)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except Exception:
    colab_files = None
    IN_COLAB = False


# ======================================================================
# 1. USER CONFIGURATION
# ======================================================================

# In Colab, the default is /content/final_data.xlsx. If it does not exist,
# the notebook asks you to upload your Excel file.
# For Google Drive, first mount Drive and set DATA_PATH to the Drive path.
DATA_PATH = "/content/final_data.xlsx" if IN_COLAB else "final_data.xlsx"

TARGET_COLUMN = "amyloidosis"
OUTPUT_DIR = "/content/ML_results" if IN_COLAB else str(Path.cwd() / "ML_results")

# The original pipeline divides the 26 predictors by the Excel column order:
# the first echo variable and all columns after it make up the 19 echo fields.
ECHO_START_PREFIX = "LV End-Diastolic Diameter"
EXPECTED_ALL_FEATURES = 26
EXPECTED_ECHO_FEATURES = 19
EXPECTED_CLINICAL_FEATURES = 7

# Declared categorical variables; Age follows the exact automatic rule from the original Jupyter code.
CATEGORICAL_VARIABLES = [
    "Sex",
    "Age",
    "Hypertension",
    "Diabetes mellitus",
    "Coronary artery disease",
    "Atrial fibrillation",
    "LV dias",
    "LVH",
    "LAViCat",
    "RAViCat",
    "AI",
    "MR",
    "TR",
    "PI",
]

# These settings reproduce the original Jupyter pipeline's Age-type rule.
# None = auto-detect; numeric age in years with >10 unique values is numerical,
# while age bands / low-cardinality codes remain categorical.
AGE_VARIABLE = "Age"
AGE_TREAT_AS_NUMERIC = None
AGE_NUMERIC_UNIQUE_THRESHOLD = 10
AGE_NUMERIC_PARSE_FRACTION = 0.90

# Optional exclusions, normally leave empty.
# If the workbook contains administrative/identifier columns or demographic
# variables that should appear ONLY in the demographic report (e.g. marital
# status, height, weight) but are not part of the original 26 model predictors,
# add those column names here. They remain available to the demographic report.
DROP_COLUMNS = []

# Optional patient/group identifier. None = automatically detect a common ID.
GROUP_COLUMN = None
AUTO_DETECT_GROUP_COLUMN = True
GROUP_COLUMN_CANDIDATES = [
    "patient_id", "patientid", "patient id", "patient_name", "patient name",
    "subject_id", "subjectid", "subject id", "participant_id",
    "case_id", "caseid", "case id", "record_id", "recordid",
    "mrn", "medical_record_number", "patient_code", "patient_no",
    "study_id", "studyid", "patient",
]

TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.50

# Plot options
ROC_GROUP_SHADOW = True  # Toggle silhouette/halo effect for feature-set groups.
SAVE_SVG = True
SHOW_PLOTS_IN_NOTEBOOK = True

# SHAP options
RUN_SHAP = True
SHAP_BACKGROUND_SAMPLES = 40
SHAP_EXPLAIN_SAMPLES = 50
SHAP_PERMUTATION_REPEATS = 8
SHAP_MAX_DISPLAY = 20
SHAP_RANDOM_STATE = RANDOM_STATE + 100

# Automatically create a ZIP of ML_results and download it in Colab.
CREATE_RESULTS_ZIP = True
AUTO_DOWNLOAD_ZIP = True

# ----------------------------------------------------------------------
# HYPERPARAMETER TUNING (this is what the original Jupyter BASE run did)
# ----------------------------------------------------------------------
# In the Jupyter pipeline (RUN_FINAL_DEVELOPMENT_TUNING = True) every model is
# tuned ONCE per feature set on the 80% development data only:
#   - 3-fold stratified inner CV (shuffle, random_state = RANDOM_STATE + 9000)
#   - candidates are ordered simplest -> most complex
#   - choose the simplest candidate whose mean inner AUC is within one
#     standard error of the best candidate (one-SE rule)
# The chosen parameters are then used for BOTH the 5-fold CV estimates and the
# final model that is evaluated on the held-out 20% test set.
# Set TUNE_HYPERPARAMETERS = False to use only the fixed starting values.
TUNE_HYPERPARAMETERS = True
INNER_CV_FOLDS = 3
TUNING_RANDOM_STATE = RANDOM_STATE + 9000
USE_ONE_SE_RULE = True

PARAM_CANDIDATES = {
    "Random Forest": [
        {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 10,
         "min_samples_split": 10, "max_features": "log2"},
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5,
         "min_samples_split": 8, "max_features": "log2"},
        {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 3,
         "min_samples_split": 6, "max_features": "log2"},
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03,
         "min_child_weight": 10, "reg_lambda": 15.0, "gamma": 1.5,
         "subsample": 0.70, "colsample_bytree": 0.70},
        {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.03,
         "min_child_weight": 5, "reg_lambda": 10.0, "gamma": 1.0,
         "subsample": 0.80, "colsample_bytree": 0.70},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
         "min_child_weight": 5, "reg_lambda": 10.0, "gamma": 0.5,
         "subsample": 0.80, "colsample_bytree": 0.80},
    ],
    "SVM": [
        {"C": 0.01, "gamma": 0.01},
        {"C": 0.10, "gamma": 0.01},
        {"C": 0.50, "gamma": "scale"},
    ],
    "MLP": [
        {"hidden_layer_sizes": (8,), "alpha": 1.0},
        {"hidden_layer_sizes": (16,), "alpha": 0.30},
        {"hidden_layer_sizes": (32,), "alpha": 0.10},
    ],
}

# Set automatically in main(): True only if the workbook has repeated patient IDs.
GROUP_AWARE_SPLITTING = False

# Model definitions: fixed starting values (tuned per feature set if enabled above).
MODEL_NAMES = ["Random Forest", "XGBoost", "SVM", "MLP"]
FEATURE_SET_ORDER = ["All 26", "Echo 19", "Clinical 7"]
METRICS = [
    "AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "F1",
]

FEATURE_SET_COLORS = {
    "All 26": "#2878B5",
    "Echo 19": "#E07A24",
    "Clinical 7": "#26966F",
}
MODEL_LINESTYLES = {
    "Random Forest": "-",
    "XGBoost": "--",
    "SVM": "-.",
    "MLP": ":",
}

# The proposal specifically lists age, sex, marital status, height and weight.
# BMI is included when it exists in the workbook. The report also describes
# every study predictor (clinical and echocardiographic) by amyloidosis group.
DEMOGRAPHIC_ALIASES = {
    "Age": ["Age", "Age (years)", "سن"],
    "Sex": ["Sex", "Gender", "Sex (M/F)", "جنسیت"],
    "Marital status": [
        "Marital status", "Marital_Status", "marital_status",
        "MaritalStatus", "Marital status (married/unmarried)",
        "وضعیت تأهل", "وضعیت تاهل",
    ],
    "Height": ["Height", "Height (cm)", "Height_cm", "قد"],
    "Weight": ["Weight", "Weight (kg)", "Weight_kg", "وزن"],
    "BMI": ["BMI", "BMI (kg/m2)", "Body Mass Index", "شاخص توده بدنی"],
}
CONTINUOUS_DEMOGRAPHIC_LABELS = {"age", "height", "weight", "bmi"}


# ======================================================================
# 2. GENERAL HELPERS
# ======================================================================

def normalize_name(value):
    """Normalize column labels for robust matching."""
    text = str(value).strip().casefold()
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", "").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def find_alias_column(columns, aliases):
    lookup = {normalize_name(col): col for col in columns}
    for alias in aliases:
        match = lookup.get(normalize_name(alias))
        if match is not None:
            return match
    return None


def safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    return value.strip("_")


def ensure_output_directories(output_dir):
    output_dir = Path(output_dir)
    folders = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "metric_plots": output_dir / "plots" / "metric_comparisons",
        "shap": output_dir / "SHAP",
        "models": output_dir / "fitted_models",
        "predictions": output_dir / "predictions",
        "tables": output_dir / "tables",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def resolve_dataset_path(path):
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if IN_COLAB:
        print(
            f"Dataset was not found at {candidate}.\n"
            "Please upload the Excel workbook (usually final_data.xlsx)."
        )
        uploaded = colab_files.upload()
        if not uploaded:
            raise FileNotFoundError("No Excel file was uploaded.")
        uploaded_name = next(iter(uploaded.keys()))
        uploaded_path = Path("/content") / uploaded_name
        if not uploaded_path.exists():
            raise FileNotFoundError(
                f"Uploaded file is not available at {uploaded_path}."
            )
        return uploaded_path.resolve()

    raise FileNotFoundError(
        f"Dataset not found: {candidate.resolve()}\n"
        "Set DATA_PATH to the correct Excel file path."
    )


def configure_feature_types(X):
    """Reproduce the original Jupyter pipeline's variable typing exactly."""
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    active_categorical = list(CATEGORICAL_VARIABLES)

    if AGE_TREAT_AS_NUMERIC not in (None, True, False):
        raise ValueError("AGE_TREAT_AS_NUMERIC must be None, True, or False.")

    if AGE_VARIABLE in X.columns:
        age_original = X[AGE_VARIABLE]
        age_numeric = pd.to_numeric(age_original, errors="coerce")
        nonmissing_age = age_original.notna()
        parse_fraction = (
            float(age_numeric[nonmissing_age].notna().mean())
            if nonmissing_age.any() else 0.0
        )
        age_unique_numeric = int(age_numeric.nunique(dropna=True))
        age_looks_continuous = (
            parse_fraction >= AGE_NUMERIC_PARSE_FRACTION
            and age_unique_numeric > AGE_NUMERIC_UNIQUE_THRESHOLD
        )
        treat_age_as_numeric = (
            AGE_TREAT_AS_NUMERIC
            if AGE_TREAT_AS_NUMERIC is not None
            else age_looks_continuous
        )
        if treat_age_as_numeric:
            if parse_fraction < AGE_NUMERIC_PARSE_FRACTION:
                raise ValueError(
                    f"AGE_TREAT_AS_NUMERIC=True, but only {parse_fraction:.1%} "
                    f"of nonmissing '{AGE_VARIABLE}' values can be parsed numerically."
                )
            X[AGE_VARIABLE] = age_numeric
            active_categorical = [c for c in active_categorical if c != AGE_VARIABLE]
            age_decision = "numerical"
        else:
            age_decision = "categorical"
        print(
            f"\nAGE TYPE (original Jupyter rule): {AGE_VARIABLE} treated as {age_decision}; "
            f"unique numeric values={age_unique_numeric}, "
            f"parse fraction={parse_fraction:.1%}, override={AGE_TREAT_AS_NUMERIC!r}."
        )

    categorical = [c for c in active_categorical if c in X.columns]
    missing_categorical = [c for c in active_categorical if c not in X.columns]
    if missing_categorical:
        print("Categorical variables not present in this feature set:", missing_categorical)

    numeric = [c for c in X.columns if c not in categorical]

    for col in numeric:
        original_nonmissing = X[col].notna()
        converted = pd.to_numeric(X[col], errors="coerce")
        invalid = original_nonmissing & converted.isna()
        if invalid.any():
            print(
                f"WARNING: {int(invalid.sum())} non-numeric value(s) in numerical "
                f"feature '{col}' were converted to missing. Examples: "
                f"{X.loc[invalid, col].astype(str).unique()[:5].tolist()}"
            )
        X[col] = converted

    # The original Jupyter script stringified categorical values without trimming.
    for col in categorical:
        X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)

    empty = X.columns[X.isna().all()].tolist()
    if empty:
        print("Removing entirely empty predictors, as in original Jupyter code:", empty)
        X = X.drop(columns=empty)

    numeric = [c for c in X.columns if c not in categorical]
    categorical = [c for c in active_categorical if c in X.columns]
    if X.shape[1] == 0:
        raise ValueError("No usable predictors remain after preprocessing.")

    return X, numeric, categorical


def detect_group_column(df):
    if GROUP_COLUMN is not None:
        if GROUP_COLUMN not in df.columns:
            raise ValueError(
                f"GROUP_COLUMN={GROUP_COLUMN!r} was not found in the workbook."
            )
        return GROUP_COLUMN

    if not AUTO_DETECT_GROUP_COLUMN:
        return None

    lookup = {normalize_name(col): col for col in df.columns}
    for candidate in GROUP_COLUMN_CANDIDATES:
        found = lookup.get(normalize_name(candidate))
        if found is not None and found != TARGET_COLUMN:
            return found
    return None


def _minimum_groups_per_class(y_values, group_values):
    y_values = np.asarray(y_values, dtype=int)
    group_values = np.asarray(group_values)
    counts = [
        len(pd.unique(group_values[y_values == class_value]))
        for class_value in [0, 1]
    ]
    return int(min(counts)) if counts else 0


def make_stratified_cv_splits(X_part, y_part, groups_part=None, requested_splits=5,
                              random_state=RANDOM_STATE, context="cross-validation"):
    """Exact port of the Jupyter CV splitter (keeps each patient's rows together
    only when repeated patient IDs exist)."""
    y_part = np.asarray(y_part, dtype=int)
    if groups_part is None or not GROUP_AWARE_SPLITTING:
        max_splits = min(
            int(requested_splits),
            int(np.bincount(y_part, minlength=2).min()),
        )
        if max_splits < 2:
            raise ValueError(f"Not enough observations per class for {context}.")
        splitter = StratifiedKFold(
            n_splits=max_splits, shuffle=True, random_state=random_state
        )
        return list(splitter.split(X_part, y_part))

    group_array = np.asarray(groups_part)
    max_splits = min(
        int(requested_splits),
        _minimum_groups_per_class(y_part, group_array),
        int(pd.Series(group_array).nunique()),
    )
    last_error = None
    for n_splits in range(max_splits, 1, -1):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            candidate_splits = list(splitter.split(X_part, y_part, groups=group_array))
            valid = all(
                len(np.unique(y_part[tr])) == 2 and len(np.unique(y_part[va])) == 2
                for tr, va in candidate_splits
            )
            if valid:
                return candidate_splits
        except Exception as exc:
            last_error = exc
    message = (
        f"Unable to create class-balanced, patient-disjoint {context}. "
        "Check whether each outcome class is represented across enough patients."
    )
    if last_error is not None:
        message += f" Last error: {last_error}"
    raise ValueError(message)


def make_stratified_holdout_indices(X_part, y_part, test_size, random_state,
                                    groups_part=None, context="holdout"):
    """Exact port of the Jupyter 80:20 hold-out split."""
    y_part = np.asarray(y_part, dtype=int)
    if groups_part is None or not GROUP_AWARE_SPLITTING:
        train_idx, test_idx = train_test_split(
            np.arange(len(y_part)), test_size=test_size,
            stratify=y_part, random_state=random_state,
        )
        return np.asarray(train_idx), np.asarray(test_idx)

    requested_folds = max(2, int(round(1.0 / float(test_size))))
    splits = make_stratified_cv_splits(
        X_part, y_part, np.asarray(groups_part),
        requested_splits=requested_folds, random_state=random_state, context=context,
    )
    valid_splits = [
        (tr, te) for tr, te in splits
        if len(np.unique(y_part[tr])) == 2 and len(np.unique(y_part[te])) == 2
    ]
    if not valid_splits:
        raise ValueError(f"No patient-disjoint {context} split contains both outcome classes.")
    return min(valid_splits, key=lambda pair: abs(len(pair[1]) / len(y_part) - test_size))


def get_holdout_and_cv_splits(X, y, groups=None):
    """
    One 80:20 hold-out split + 5-fold stratified CV inside the development set.
    Identical to the Jupyter pipeline (StratifiedKFold / StratifiedGroupKFold,
    shuffle=True, random_state=42; plain stratified split when no repeated IDs).
    """
    global GROUP_AWARE_SPLITTING
    y = np.asarray(y, dtype=int)
    group_array = np.asarray(groups) if groups is not None else None
    GROUP_AWARE_SPLITTING = bool(
        group_array is not None and pd.Series(group_array).duplicated(keep=False).any()
    )

    train_idx, test_idx = make_stratified_holdout_indices(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        groups_part=group_array if GROUP_AWARE_SPLITTING else None,
        context="main held-out split",
    )
    if GROUP_AWARE_SPLITTING:
        if set(group_array[train_idx]).intersection(set(group_array[test_idx])):
            raise RuntimeError("Patient leakage detected between development and held-out sets.")

    cv_splits = make_stratified_cv_splits(
        X.iloc[train_idx], y[train_idx],
        group_array[train_idx] if GROUP_AWARE_SPLITTING else None,
        requested_splits=CV_FOLDS, random_state=RANDOM_STATE,
        context="standard cross-validation",
    )
    return train_idx, test_idx, cv_splits, GROUP_AWARE_SPLITTING, len(cv_splits)


# ======================================================================
# 3. DEMOGRAPHIC / BASELINE CHARACTERISTICS ANALYSIS
# ======================================================================

def shapiro_normal(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or len(values) > 5000 or np.unique(values).size < 3:
        return False
    try:
        return float(shapiro(values).pvalue) >= 0.05
    except Exception:
        return False


def format_mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "—"
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):.2f} ± {sd:.2f} (n={len(values)})"


def format_median_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "—"
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.2f} [{q1:.2f}, {q3:.2f}] (n={len(values)})"


def numeric_group_summary(values0, values1):
    normal = shapiro_normal(values0) and shapiro_normal(values1)
    formatter = format_mean_sd if normal else format_median_iqr
    summary0 = formatter(values0)
    summary1 = formatter(values1)

    if len(values0) >= 2 and len(values1) >= 2:
        if normal:
            result = ttest_ind(values0, values1, equal_var=False, nan_policy="omit")
            test_name = "Welch independent-samples t-test"
        else:
            result = mannwhitneyu(values0, values1, alternative="two-sided")
            test_name = "Mann–Whitney U test"
        p_value = float(result.pvalue)
    else:
        p_value = np.nan
        test_name = "Not testable (insufficient observations)"

    distribution = "Mean ± SD" if normal else "Median [IQR]"
    return summary0, summary1, p_value, test_name, distribution


def categorical_group_summary(series, y):
    """
    Summarize observed categories among nonmissing records. Missing values are
    reported separately in the baseline table, not treated as a clinical level
    in the inferential chi-square/Fisher comparison.
    """
    y = np.asarray(y, dtype=int)
    observed = series.notna().to_numpy()
    temp = pd.DataFrame({
        "value": series.loc[observed].astype("string").to_numpy(),
        "target": y[observed],
    })
    levels = sorted(temp["value"].unique().tolist(), key=lambda x: str(x))
    group_labels = {0: "Control", 1: "Cardiac amyloidosis"}

    summaries = {}
    level_rows = []
    for label, group_name in group_labels.items():
        subset = temp.loc[temp["target"] == label, "value"]
        denominator = len(subset)
        chunks = []
        for level in levels:
            count = int((subset == level).sum())
            pct = 100 * count / denominator if denominator else np.nan
            if count:
                chunks.append(f"{level}: {count} ({pct:.1f}%)")
            level_rows.append({
                "Category": level,
                "Group": group_name,
                "N": count,
                "Percent": pct,
                "Denominator_nonmissing": denominator,
            })
        summaries[label] = "; ".join(chunks) if chunks else "—"

    table = pd.crosstab(temp["value"], temp["target"])
    if table.shape[0] >= 2 and table.shape[1] == 2:
        try:
            chi2, chi_p, _, expected = chi2_contingency(
                table.to_numpy(), correction=False
            )
            if table.shape == (2, 2) and np.any(expected < 5):
                _, p_value = fisher_exact(table.to_numpy())
                test_name = "Fisher exact test"
            else:
                p_value = chi_p
                test_name = "Pearson chi-square test"
        except Exception:
            p_value = np.nan
            test_name = "Not testable"
    else:
        p_value = np.nan
        test_name = "Not testable (fewer than 2 categories/classes)"

    return summaries[0], summaries[1], float(p_value), test_name, level_rows


def create_demographic_analysis(df, y, predictor_columns, clinical_columns,
                                echo_columns, output_dir):
    """
    Create a proposal-aligned baseline table for demographic variables and
    all study predictors. This is descriptive/inferential reporting only;
    p-values are NOT used for feature selection.
    """
    available_demographics = {}
    for label, aliases in DEMOGRAPHIC_ALIASES.items():
        match = find_alias_column(df.columns, aliases)
        if match is not None and match != TARGET_COLUMN:
            available_demographics[label] = match

    # Include the proposal's demographic fields even when they are not among
    # the 26 ML predictors, then include all 26 clinical/echo predictors.
    variables = []
    for col in available_demographics.values():
        if col not in variables:
            variables.append(col)
    for col in predictor_columns:
        if col not in variables:
            variables.append(col)

    echo_set = set(echo_columns)
    clinical_set = set(clinical_columns)
    demo_set = set(available_demographics.values())
    categorical_norm = {normalize_name(c) for c in CATEGORICAL_VARIABLES}

    rows = []
    level_rows_all = []
    missing_rows = []
    for col in variables:
        values = df[col]
        missing_n = int(values.isna().sum())
        n_nonmissing = int(values.notna().sum())
        variable_type = (
            "Demographic" if col in demo_set
            else "Echocardiographic" if col in echo_set
            else "Clinical"
        )

        numeric_values = pd.to_numeric(values, errors="coerce")
        parse_fraction = (
            float(numeric_values[values.notna()].notna().mean())
            if values.notna().any() else 0.0
        )
        normalized_label = next(
            (label for label, real_col in available_demographics.items()
             if real_col == col),
            "",
        )
        demographic_continuous = (
            normalize_name(normalized_label) in CONTINUOUS_DEMOGRAPHIC_LABELS
        )
        is_categorical_for_ml = normalize_name(col) in categorical_norm
        is_numeric = (
            (demographic_continuous and parse_fraction >= 0.80)
            or (not is_categorical_for_ml and parse_fraction >= 0.80)
        )

        if is_numeric:
            vals = numeric_values.to_numpy(dtype=float)
            vals0 = vals[np.asarray(y) == 0]
            vals1 = vals[np.asarray(y) == 1]
            vals0 = vals0[np.isfinite(vals0)]
            vals1 = vals1[np.isfinite(vals1)]
            all_values = vals[np.isfinite(vals)]

            group0, group1, p_value, test_name, distribution = (
                numeric_group_summary(vals0, vals1)
            )
            overall = (
                format_mean_sd(all_values)
                if shapiro_normal(vals0) and shapiro_normal(vals1)
                else format_median_iqr(all_values)
            )
            rows.append({
                "Variable": col,
                "Variable_Class": variable_type,
                "Analysis_Type": "Continuous",
                "Overall": overall,
                "Controls (0)": group0,
                "Cardiac amyloidosis (1)": group1,
                "P_value": p_value,
                "Statistical_Test": test_name,
                "Summary_Format": distribution,
                "N_nonmissing": n_nonmissing,
                "Missing_N": missing_n,
                "Missing_Control_N": int(values.loc[np.asarray(y) == 0].isna().sum()),
                "Missing_CA_N": int(values.loc[np.asarray(y) == 1].isna().sum()),
                "Missing_Percent": (
                    100 * missing_n / len(df) if len(df) else np.nan
                ),
            })
        else:
            group0, group1, p_value, test_name, level_rows = (
                categorical_group_summary(values, y)
            )
            levels_overall = values.dropna().astype("string")
            overall_chunks = []
            denominator = len(levels_overall)
            for level, count in levels_overall.value_counts(dropna=False).items():
                pct = 100 * count / denominator if denominator else np.nan
                overall_chunks.append(f"{level}: {int(count)} ({pct:.1f}%)")
            rows.append({
                "Variable": col,
                "Variable_Class": variable_type,
                "Analysis_Type": "Categorical",
                "Overall": "; ".join(overall_chunks),
                "Controls (0)": group0,
                "Cardiac amyloidosis (1)": group1,
                "P_value": p_value,
                "Statistical_Test": test_name,
                "Summary_Format": "n (%)",
                "N_nonmissing": n_nonmissing,
                "Missing_N": missing_n,
                "Missing_Control_N": int(values.loc[np.asarray(y) == 0].isna().sum()),
                "Missing_CA_N": int(values.loc[np.asarray(y) == 1].isna().sum()),
                "Missing_Percent": (
                    100 * missing_n / len(df) if len(df) else np.nan
                ),
            })
            for item in level_rows:
                level_rows_all.append({"Variable": col, **item})

        missing_rows.append({
            "Variable": col,
            "N_total": len(df),
            "N_nonmissing": n_nonmissing,
            "N_missing": missing_n,
            "Missing_percent": 100 * missing_n / len(df) if len(df) else np.nan,
        })

    baseline = pd.DataFrame(rows)
    demo_only = baseline[baseline["Variable_Class"] == "Demographic"].copy()
    study_predictors = baseline[
        baseline["Variable_Class"].isin(["Clinical", "Echocardiographic"])
    ].copy()

    cohort_summary = pd.DataFrame([
        {
            "Group": "Total cohort",
            "N": int(len(y)),
            "Percent": 100.0 if len(y) else np.nan,
        },
        {
            "Group": "Controls (0)",
            "N": int(np.sum(np.asarray(y) == 0)),
            "Percent": 100 * np.mean(np.asarray(y) == 0) if len(y) else np.nan,
        },
        {
            "Group": "Cardiac amyloidosis (1)",
            "N": int(np.sum(np.asarray(y) == 1)),
            "Percent": 100 * np.mean(np.asarray(y) == 1) if len(y) else np.nan,
        },
    ])

    missing_demographics = [
        label for label in DEMOGRAPHIC_ALIASES
        if label not in available_demographics
    ]
    analysis_notes = pd.DataFrame({
        "Item": [
            "Study objective",
            "Demographic variables specified in the proposal",
            "Demographic variables found in the workbook",
            "Demographic variables not found in the workbook",
            "Age handling",
            "Continuous variable summaries",
            "Continuous group comparison",
            "Categorical variable summaries",
            "Categorical group comparison",
            "P-value use",
            "Missing data",
        ],
        "Description": [
            "Describe the cohort and compare baseline characteristics between "
            "controls and patients with cardiac amyloidosis.",
            "Age, sex, marital status, height, and weight. BMI is also reported "
            "when present in the workbook.",
            ", ".join(f"{k} = {v}" for k, v in available_demographics.items()) or "None found",
            ", ".join(missing_demographics) or "None",
            "ML preprocessing follows the original Jupyter rule: raw numeric age with >10 unique values and at least 90% parseability is numerical by default; age bands remain categorical. Demographic age in years is summarized continuously.",
            "Mean ± SD when both groups satisfy the Shapiro-Wilk normality check; "
            "otherwise median [IQR].",
            "Welch independent-samples t-test for approximately normal data; "
            "Mann–Whitney U test otherwise.",
            "Number and percentage, with category-level counts reported separately.",
            "Fisher exact test for sparse 2×2 tables; Pearson chi-square otherwise.",
            "Descriptive/inferential reporting only. No p-value is used to remove "
            "or select a predictor in the machine-learning models.",
            "Missingness is reported per variable; model imputers are fitted using "
            "training data only.",
        ],
    })

    output_path = Path(output_dir) / "demographic analysis.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cohort_summary.to_excel(writer, sheet_name="Cohort_Summary", index=False)
        baseline.to_excel(writer, sheet_name="Baseline_Characteristics", index=False)
        demo_only.to_excel(writer, sheet_name="Demographics_Only", index=False)
        study_predictors.to_excel(writer, sheet_name="Clinical_Echo_Variables", index=False)
        pd.DataFrame(level_rows_all).to_excel(
            writer, sheet_name="Categorical_Level_Detail", index=False
        )
        pd.DataFrame(missing_rows).to_excel(
            writer, sheet_name="Missingness", index=False
        )
        analysis_notes.to_excel(writer, sheet_name="Analysis_Notes", index=False)

        # Practical workbook formatting.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_len + 2, 12), 58
                )
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)

    baseline.to_csv(
        Path(output_dir) / "tables" / "demographic_analysis_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cohort_summary.to_csv(
        Path(output_dir) / "tables" / "demographic_analysis_cohort_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nDemographic/baseline analysis saved to: {output_path}")
    print(
        "The proposal's demographic variables found:",
        ", ".join(f"{k} ({v})" for k, v in available_demographics.items())
        or "None",
    )
    if missing_demographics:
        print(
            "Not present as identifiable columns in this workbook:",
            missing_demographics,
        )
    return baseline, output_path


# ======================================================================
# 4. PREPROCESSING AND CLASSIFIERS
# ======================================================================

def make_onehot_encoder(drop=None):
    """OneHotEncoder compatibility helper copied from the Jupyter baseline."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=drop,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            drop=drop,
        )


def build_preprocessing_components(selected_features, numeric_columns, categorical_columns):
    """Exact two-stage impute/ordinal-encode then one-hot pipeline from Jupyter."""
    selected_numeric = [c for c in selected_features if c in numeric_columns]
    selected_categorical = [c for c in selected_features if c in categorical_columns]
    transformers = []

    if selected_numeric:
        try:
            numeric_imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )
        except TypeError:  # Compatibility with older scikit-learn releases.
            numeric_imputer = SimpleImputer(strategy="median")
        numeric_pipeline = SklearnPipeline([
            ("imputer", numeric_imputer),
        ])
        transformers.append(("numeric", numeric_pipeline, selected_numeric))

    if selected_categorical:
        categorical_pipeline = SklearnPipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
            )),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append(("categorical", categorical_pipeline, selected_categorical))

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    n_numeric = len(selected_numeric)
    n_categorical = len(selected_categorical)
    numeric_indices = list(range(n_numeric))
    categorical_indices = list(range(n_numeric, n_numeric + n_categorical))

    post_transformers = []
    if n_numeric:
        post_transformers.append(("numeric", "passthrough", numeric_indices))
    if n_categorical:
        post_transformers.append((
            "categorical",
            make_onehot_encoder(drop=None),
            categorical_indices,
        ))

    postprocessing = ColumnTransformer(
        transformers=post_transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    metadata = {
        "selected_features": list(selected_features),
        "numeric_features": selected_numeric,
        "categorical_features": selected_categorical,
        "numeric_indices": numeric_indices,
        "categorical_indices": categorical_indices,
    }
    return preprocessing, postprocessing, metadata


def create_models():
    """Fixed classifier specifications from the original Jupyter baseline."""
    return OrderedDict([
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_split=8,
                min_samples_leaf=5,
                max_features="log2",
                max_samples=0.80,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=150,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=5,
                subsample=0.80,
                colsample_bytree=0.70,
                reg_lambda=10.0,
                gamma=1.0,
                scale_pos_weight=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        (
            "SVM",
            SVC(
                C=0.10,
                kernel="rbf",
                gamma=0.01,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=0.30,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_model_pipeline(model, model_name, feature_columns, numeric_columns,
                         categorical_columns, y_fit):
    """Recreate original Jupyter pipeline order for baseline (SMOTE passthrough)."""
    preprocessing, postprocessing, metadata = build_preprocessing_components(
        feature_columns,
        numeric_columns,
        categorical_columns,
    )
    classifier = clone(model)

    # Original Jupyter behavior: with SMOTE OFF, calculate XGBoost's class ratio
    # on the current fit partition (each CV fold or complete development set).
    if model_name == "XGBoost":
        n_negative = int(np.sum(np.asarray(y_fit) == 0))
        n_positive = int(np.sum(np.asarray(y_fit) == 1))
        if n_positive > 0:
            classifier.set_params(scale_pos_weight=n_negative / n_positive)

    # Original Jupyter behavior: MLP's internal early-stopping split is row-based,
    # so it is switched off only when repeated-patient (group-aware) splitting is used.
    if model_name == "MLP" and GROUP_AWARE_SPLITTING:
        classifier.set_params(early_stopping=False)

    # Exact original step order; baseline uses no resampling, so this step is passthrough.
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("smote", "passthrough"),
        ("postprocessing", postprocessing),
        ("scaling", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline, metadata


def positive_class_index(estimator):
    """Return the probability-column index corresponding to amyloidosis=1."""
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(
            f"Expected class label 1 (amyloidosis) in estimator.classes_; got {classes!r}."
        )
    return int(matches[0])


def get_model_outputs(fitted_pipeline, X_values, model_name):
    """Return predictions, positive-class probabilities, and ROC ranking scores.

    IMPORTANT: For SVC, the ROC-AUC ranking score is the signed decision
    function (margin), not predict_proba(). SVC's probability=True output is
    Platt-calibrated using an internal CV procedure and may disagree with
    SVC.predict(), especially in a small dataset. The original Jupyter pipeline
    correctly used decision_function() for SVM AUC and predict() for labels.
    Other models use positive-class probability for ROC-AUC.
    """
    classifier = fitted_pipeline.named_steps["classifier"]
    pos_idx = positive_class_index(classifier)

    predicted_labels = np.asarray(fitted_pipeline.predict(X_values), dtype=int)
    predicted_probabilities = np.asarray(
        fitted_pipeline.predict_proba(X_values)[:, pos_idx], dtype=float
    )

    if model_name == "SVM":
        margin = np.asarray(
            fitted_pipeline.decision_function(X_values), dtype=float
        ).reshape(-1)
        classes = np.asarray(classifier.classes_)
        # sklearn's binary decision_function is positive for classes_[1].
        # Explicitly orient it toward the positive amyloidosis class (label 1).
        ranking_scores = margin if classes[1] == 1 else -margin
    else:
        ranking_scores = predicted_probabilities.copy()

    return predicted_labels, predicted_probabilities, ranking_scores


def calculate_metrics(y_true, y_pred, y_auc_score):
    """Use model.predict() for thresholded metrics and a ranking score for AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_auc_score = np.asarray(y_auc_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "AUC": float(roc_auc_score(y_true, y_auc_score)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV": float(npv),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }, y_pred


def metric_mean_sd(values):
    values = np.asarray(values, dtype=float)
    mean_value = float(np.nanmean(values))
    sd_value = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
    return mean_value, sd_value, f"{mean_value:.3f} ± {sd_value:.3f}"


# ======================================================================
# 4B. HYPERPARAMETER TUNING (exact port of the Jupyter inner-CV + one-SE rule)
# ======================================================================

def choose_candidate_by_one_se(candidate_score_lists, use_one_se=True):
    """Choose the simplest candidate within one SE of the best mean inner-CV AUC.

    Candidate lists are ordered from SIMPLEST to MOST COMPLEX, so the first
    eligible candidate is the simplest one.
    """
    means = np.asarray([
        np.mean(scores) if len(scores) else np.nan for scores in candidate_score_lists
    ], dtype=float)
    sds = np.asarray([
        np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        for scores in candidate_score_lists
    ], dtype=float)
    ses = np.asarray([
        sd / np.sqrt(len(scores)) if len(scores) else np.nan
        for sd, scores in zip(sds, candidate_score_lists)
    ], dtype=float)

    if not np.isfinite(means).any():
        raise RuntimeError("All inner-CV candidate AUC values are unavailable.")

    best_index = int(np.nanargmax(means))
    best_mean = float(means[best_index])
    best_se = float(ses[best_index]) if np.isfinite(ses[best_index]) else 0.0
    threshold = best_mean - best_se

    if use_one_se:
        eligible = [
            idx for idx, value in enumerate(means)
            if np.isfinite(value) and value >= threshold
        ]
        selected_index = min(eligible) if eligible else best_index
    else:
        selected_index = best_index

    details = pd.DataFrame({
        "Candidate_Index": np.arange(len(means)),
        "Mean_Inner_AUC": means,
        "SD_Inner_AUC": sds,
        "SE_Inner_AUC": ses,
        "Within_One_SE_Of_Best": np.isfinite(means) & (means >= threshold),
        "Best_Mean_Candidate": np.arange(len(means)) == best_index,
        "Selected_By_One_SE_Rule": np.arange(len(means)) == selected_index,
    })
    return selected_index, best_index, threshold, details


def tune_model_with_inner_cv(X_development, y_development, model_name, base_model,
                             feature_columns, numeric_columns, categorical_columns,
                             groups_development=None):
    """Leakage-safe inner-CV tuning on the development set only."""
    y_development = np.asarray(y_development, dtype=int)
    candidate_params = PARAM_CANDIDATES.get(model_name, [{}])
    inner_splits = make_stratified_cv_splits(
        X_development,
        y_development,
        groups_development if GROUP_AWARE_SPLITTING else None,
        requested_splits=INNER_CV_FOLDS,
        random_state=TUNING_RANDOM_STATE,
        context=f"final development tuning inner CV for {model_name}",
    )

    candidate_score_lists = [[] for _ in candidate_params]
    for candidate_index, params in enumerate(candidate_params):
        for inner_train_idx, inner_val_idx in inner_splits:
            X_inner_train = X_development.iloc[inner_train_idx]
            y_inner_train = y_development[inner_train_idx]
            X_inner_val = X_development.iloc[inner_val_idx]
            y_inner_val = y_development[inner_val_idx]

            tuned_model = clone(base_model).set_params(**params)
            inner_pipeline, _ = build_model_pipeline(
                tuned_model, model_name, feature_columns,
                numeric_columns, categorical_columns, y_inner_train,
            )
            inner_pipeline.fit(X_inner_train[feature_columns], y_inner_train)
            _, _, inner_score = get_model_outputs(
                inner_pipeline, X_inner_val[feature_columns], model_name
            )
            candidate_score_lists[candidate_index].append(
                float(roc_auc_score(y_inner_val, inner_score))
            )

    selected_index, best_index, threshold, details = choose_candidate_by_one_se(
        candidate_score_lists, use_one_se=USE_ONE_SE_RULE
    )
    details.insert(0, "Model", model_name)
    details.insert(1, "Candidate_Parameters", [repr(x) for x in candidate_params])
    details["One_SE_Threshold"] = threshold
    details["Selected_Parameters"] = repr(candidate_params[selected_index])
    details["Best_Mean_Parameters"] = repr(candidate_params[best_index])
    details["Effective_Inner_Folds"] = len(inner_splits)
    return candidate_params[selected_index], details


# ======================================================================
# 5. SHAP HELPERS
# ======================================================================

def extract_positive_class_shap(shap_output):
    """Normalize SHAP outputs from different SHAP versions to (rows, features)."""
    if hasattr(shap_output, "values"):
        values = shap_output.values
    else:
        values = shap_output

    if isinstance(values, list):
        if len(values) > 1:
            values = values[1]
        else:
            values = values[0]

    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[1] == 2:
            values = values[:, 1, :]
        else:
            raise ValueError(f"Unexpected three-dimensional SHAP shape: {values.shape}")
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")
    return values


def transform_for_classifier(pipeline, X_values):
    """Apply original Jupyter preprocessing -> one-hot -> scaler sequence."""
    transformed = pipeline.named_steps["preprocessing"].transform(X_values)
    transformed = pipeline.named_steps["postprocessing"].transform(transformed)
    transformed = pipeline.named_steps["scaling"].transform(transformed)
    return np.asarray(transformed, dtype=float)


def encoded_feature_metadata(pipeline, numeric_columns, categorical_columns):
    """Return output feature names and their original-variable parent mapping."""
    preprocessing = pipeline.named_steps["preprocessing"]
    postprocessing = pipeline.named_steps["postprocessing"]

    encoded_names = list(numeric_columns)
    parent_map = list(numeric_columns)

    if categorical_columns:
        ordinal_pipeline = preprocessing.named_transformers_["categorical"]
        ordinal_encoder = ordinal_pipeline.named_steps["encoder"]
        onehot_encoder = postprocessing.named_transformers_["categorical"]

        # The ordinal encoding preserves the original category ordering. The
        # subsequent one-hot encoder has one output for every ordinal category.
        for column, raw_categories, encoded_categories in zip(
            categorical_columns,
            ordinal_encoder.categories_,
            onehot_encoder.categories_,
        ):
            if len(raw_categories) != len(encoded_categories):
                raise RuntimeError(
                    f"Category-count mismatch when mapping SHAP columns for {column!r}."
                )
            encoded_names.extend([f"{column}={value}" for value in raw_categories])
            parent_map.extend([column] * len(encoded_categories))

    return encoded_names, parent_map


def save_shap_plots(shap_values, plot_data, feature_names, output_folder,
                    plot_title, grouped_importance=None):
    import shap

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        features=plot_data,
        feature_names=feature_names,
        max_display=SHAP_MAX_DISPLAY,
        plot_type="dot",
        show=False,
    )
    plt.title(f"{plot_title} — SHAP beeswarm", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_folder / "SHAP_beeswarm.png", dpi=300, bbox_inches="tight")
    if SAVE_SVG:
        plt.savefig(output_folder / "SHAP_beeswarm.svg", bbox_inches="tight")
    if SHOW_PLOTS_IN_NOTEBOOK and IN_COLAB:
        plt.show()
    plt.close()

    if grouped_importance is None:
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Absolute_SHAP": np.mean(np.abs(shap_values), axis=0),
        })
    else:
        importance = grouped_importance.copy()

    importance = importance.sort_values(
        "Mean_Absolute_SHAP", ascending=False
    ).reset_index(drop=True)
    importance.to_csv(
        output_folder / "SHAP_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_imp = importance.head(SHAP_MAX_DISPLAY).sort_values(
        "Mean_Absolute_SHAP", ascending=True
    )
    fig_height = max(5.5, 0.32 * len(plot_imp))
    plt.figure(figsize=(10, fig_height))
    plt.barh(
        plot_imp["Feature"],
        plot_imp["Mean_Absolute_SHAP"],
        color="#4776A8",
        alpha=0.90,
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title(f"{plot_title} — Global SHAP feature importance", fontsize=13)
    plt.grid(axis="x", alpha=0.22)
    plt.tight_layout()
    plt.savefig(
        output_folder / "SHAP_feature_importance_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SAVE_SVG:
        plt.savefig(
            output_folder / "SHAP_feature_importance_bar.svg",
            bbox_inches="tight",
        )
    if SHOW_PLOTS_IN_NOTEBOOK and IN_COLAB:
        plt.show()
    plt.close()
    return importance




def run_tree_shap(pipeline, X_train, X_test, feature_columns,
                  numeric_columns, categorical_columns, output_folder,
                  title):
    import shap

    classifier = pipeline.named_steps["classifier"]

    rng = np.random.default_rng(SHAP_RANDOM_STATE)
    bg_n = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
    exp_n = min(SHAP_EXPLAIN_SAMPLES, len(X_test))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    exp_idx = rng.choice(len(X_test), size=exp_n, replace=False)

    X_bg = X_train.iloc[bg_idx][feature_columns]
    X_exp = X_test.iloc[exp_idx][feature_columns]

    bg_transformed = transform_for_classifier(pipeline, X_bg)
    exp_transformed = transform_for_classifier(pipeline, X_exp)

    encoded_names, parent_map = encoded_feature_metadata(
        pipeline, numeric_columns, categorical_columns
    )
    if len(parent_map) != len(encoded_names):
        raise RuntimeError(
            "Could not map transformed one-hot columns back to original variables."
        )

    explainer = shap.TreeExplainer(
        classifier,
        data=bg_transformed,
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_output = explainer(exp_transformed, check_additivity=False)
    values = extract_positive_class_shap(shap_output)

    # For the beeswarm, retain encoded categories (e.g. LVH_yes) so category
    # contributions remain visible. For the global bar, aggregate all levels
    # back to their original predictor.
    grouped_rows = []
    for feature in feature_columns:
        indices = [i for i, parent in enumerate(parent_map) if parent == feature]
        if not indices:
            continue
        combined_absolute = np.sum(np.abs(values[:, indices]), axis=1)
        grouped_rows.append({
            "Feature": feature,
            "Mean_Absolute_SHAP": float(np.mean(combined_absolute)),
        })
    grouped_importance = pd.DataFrame(grouped_rows)

    importance = save_shap_plots(
        values,
        exp_transformed,
        encoded_names,
        output_folder,
        title,
        grouped_importance=grouped_importance,
    )
    pd.DataFrame(values, columns=encoded_names).to_csv(
        Path(output_folder) / "SHAP_values_encoded_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return importance


def run_permutation_shap(pipeline, X_train, X_test, feature_columns,
                         numeric_columns, categorical_columns,
                         output_folder, title):
    """
    Model-agnostic PermutationSHAP for SVM/MLP.

    The explainer receives the numeric matrix produced by the fitted training
    preprocessor. This avoids arithmetic on raw string categories inside the
    SHAP masker. One-hot columns are retained in the beeswarm, and SHAP
    contributions are aggregated back to their original input variables for
    the global feature-importance bar plot.
    """
    import shap

    classifier = pipeline.named_steps["classifier"]

    rng = np.random.default_rng(SHAP_RANDOM_STATE)
    bg_n = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
    exp_n = min(SHAP_EXPLAIN_SAMPLES, len(X_test))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    exp_idx = rng.choice(len(X_test), size=exp_n, replace=False)

    X_bg = X_train.iloc[bg_idx][feature_columns]
    X_exp = X_test.iloc[exp_idx][feature_columns]

    # Preprocessing has already been fitted on the development data only.
    # Transforming the background and held-out observations does not refit it.
    background_transformed = transform_for_classifier(pipeline, X_bg)
    explain_transformed = transform_for_classifier(pipeline, X_exp)
    encoded_names, parent_map = encoded_feature_metadata(
        pipeline, numeric_columns, categorical_columns
    )

    if len(parent_map) != len(encoded_names):
        raise RuntimeError(
            "Could not map transformed one-hot columns back to original variables."
        )

    def predict_classifier_output(transformed_values):
        values = np.asarray(transformed_values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if isinstance(classifier, SVC):
            margin = np.asarray(classifier.decision_function(values), dtype=float).reshape(-1)
            classes = np.asarray(classifier.classes_)
            return margin if classes[1] == 1 else -margin
        return classifier.predict_proba(values)[:, positive_class_index(classifier)]

    masker = shap.maskers.Independent(background_transformed)
    explainer = shap.PermutationExplainer(
        predict_classifier_output,
        masker,
        feature_names=encoded_names,
    )
    max_evals = (
        2 * len(encoded_names) + 1
    ) * SHAP_PERMUTATION_REPEATS
    shap_output = explainer(
        explain_transformed,
        max_evals=max_evals,
    )
    values = extract_positive_class_shap(shap_output)

    grouped_rows = []
    for feature in feature_columns:
        indices = [i for i, parent in enumerate(parent_map) if parent == feature]
        if not indices:
            continue
        combined_absolute = np.sum(np.abs(values[:, indices]), axis=1)
        grouped_rows.append({
            "Feature": feature,
            "Mean_Absolute_SHAP": float(np.mean(combined_absolute)),
        })
    grouped_importance = pd.DataFrame(grouped_rows)

    importance = save_shap_plots(
        values,
        explain_transformed,
        encoded_names,
        output_folder,
        title,
        grouped_importance=grouped_importance,
    )
    pd.DataFrame(values, columns=encoded_names).to_csv(
        Path(output_folder) / "SHAP_values_encoded_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return importance


# ======================================================================
# 6. VISUALIZATION HELPERS
# ======================================================================

def plot_all_roc_curves(roc_records, output_dir):
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    handles = []
    labels = []

    for feature_set in FEATURE_SET_ORDER:
        for model_name in MODEL_NAMES:
            key = (feature_set, model_name)
            if key not in roc_records:
                continue
            fpr, tpr, auc_value = roc_records[key]
            color = FEATURE_SET_COLORS[feature_set]
            line, = ax.plot(
                fpr,
                tpr,
                color=color,
                linestyle=MODEL_LINESTYLES[model_name],
                linewidth=2.0,
                label=f"{feature_set} — {model_name} (AUC={auc_value:.3f})",
                alpha=0.95,
            )
            if ROC_GROUP_SHADOW:
                line.set_path_effects([
                    path_effects.Stroke(
                        linewidth=6.0,
                        foreground=color,
                        alpha=0.18,
                    ),
                    path_effects.Normal(),
                ])
            handles.append(line)
            labels.append(line.get_label())

    ax.plot(
        [0, 1], [0, 1],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Chance",
    )
    ax.set(
        xlabel="1 − Specificity (False Positive Rate)",
        ylabel="Sensitivity (True Positive Rate)",
        title="ROC Curves — All 12 Model / Feature-Set Combinations",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.22)
    ax.legend(
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / "ROC_AUC_all_12_models.png",
        dpi=320,
        bbox_inches="tight",
    )
    if SAVE_SVG:
        fig.savefig(
            Path(output_dir) / "ROC_AUC_all_12_models.svg",
            bbox_inches="tight",
        )
    if SHOW_PLOTS_IN_NOTEBOOK and IN_COLAB:
        plt.show()
    plt.close(fig)


def plot_model_metric_comparisons(summary_numeric, output_dir):
    """One plot per model: seven held-out metrics compared across feature sets."""
    for model_name in MODEL_NAMES:
        model_df = summary_numeric[
            summary_numeric["Model"] == model_name
        ].set_index("Feature_Set").reindex(FEATURE_SET_ORDER)

        x = np.arange(len(METRICS))
        width = 0.24
        fig, ax = plt.subplots(figsize=(13, 6.5))

        for i, feature_set in enumerate(FEATURE_SET_ORDER):
            values = [
                model_df.loc[feature_set, f"Test_{metric}"]
                if feature_set in model_df.index else np.nan
                for metric in METRICS
            ]
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=feature_set,
                color=FEATURE_SET_COLORS[feature_set],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(METRICS)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Held-out test performance")
        ax.set_title(
            f"{model_name}: Performance Across the Three Feature Sets",
            fontsize=13,
            pad=12,
        )
        ax.grid(axis="y", alpha=0.22)
        ax.legend(title="Feature set", frameon=True)
        fig.tight_layout()

        filename = safe_filename(model_name) + "_feature_set_metric_comparison"
        fig.savefig(
            Path(output_dir) / f"{filename}.png",
            dpi=320,
            bbox_inches="tight",
        )
        if SAVE_SVG:
            fig.savefig(
                Path(output_dir) / f"{filename}.svg",
                bbox_inches="tight",
            )
        if SHOW_PLOTS_IN_NOTEBOOK and IN_COLAB:
            plt.show()
        plt.close(fig)


# ======================================================================
# 7. MAIN ANALYSIS
# ======================================================================

def main():
    folders = ensure_output_directories(OUTPUT_DIR)
    data_file = resolve_dataset_path(DATA_PATH)

    print("=" * 88)
    print("CARDIAC AMYLOIDOSIS — BASELINE ML ANALYSIS")
    print("=" * 88)
    print("Dataset:", data_file)
    print("Output folder:", folders["root"])
    print("Feature sets: All 26 / Echo 19 / Clinical 7")
    print("Classifiers: Random Forest / XGBoost / SVM / MLP")
    print("Total model-feature-set combinations: 12")
    print("Validation: 80:20 hold-out + 5-fold CV in development data")
    print("Feature selection: NONE (original Jupyter baseline experiment)")
    print("LASSO: OFF | Stepwise: OFF | SMOTE: OFF | PCA: OFF")
    print("Original Jupyter preprocessing: median imputation; categorical constant imputation; OrdinalEncoder; OneHotEncoder; StandardScaler")
    print("Original Jupyter model hyperparameters, class weighting, fit-time XGBoost class ratio, and CV/holdout seeds are restored.")
    print("Hyperparameter tuning (Jupyter BASE behaviour):", TUNE_HYPERPARAMETERS,
          "| inner folds:", INNER_CV_FOLDS, "| one-SE rule:", USE_ONE_SE_RULE)
    print("Overfitting tests / Harrell bootstrap: REMOVED")
    print("SHAP:", RUN_SHAP)
    print("ROC group shadow:", ROC_GROUP_SHADOW)

    df = pd.read_excel(data_file, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names detected: {duplicates}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. Columns: {df.columns.tolist()}"
        )

    target_numeric = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    unexpected = set(target_numeric.dropna().unique()) - {0, 1}
    if unexpected:
        raise ValueError(
            f"Unexpected target values {unexpected}. Expected 0=control and 1=CA."
        )
    valid_rows = target_numeric.notna()
    df = df.loc[valid_rows].copy()
    y = target_numeric.loc[valid_rows].astype(int).to_numpy()

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Both target classes are required: 0=controls, 1=cardiac amyloidosis."
        )

    group_column = detect_group_column(df)
    groups = None
    group_aware = False
    if group_column is not None:
        group_series = df[group_column].astype("string")
        missing_group = group_series.isna()
        if missing_group.any():
            # Each missing ID is assigned a unique row ID rather than grouping
            # all missing IDs into one artificial patient.
            row_missing = np.flatnonzero(missing_group.to_numpy())
            group_series.iloc[row_missing] = [
                f"__MISSING_ID_ROW_{i}" for i in row_missing
            ]
        groups = group_series.astype(str).to_numpy()
        group_label_counts = pd.DataFrame({
            "group": groups,
            "target": y,
        }).groupby("group")["target"].nunique()
        if (group_label_counts > 1).any():
            raise ValueError(
                "A patient/group identifier has conflicting target labels. "
                "Resolve these repeated-patient records before model evaluation."
            )
        group_aware = bool(pd.Series(groups).duplicated(keep=False).any())
        print(
            f"Detected identifier: {group_column}; "
            f"repeated groups: {group_aware}; unique groups: {pd.Series(groups).nunique()}"
        )
    else:
        print("No patient identifier detected; row-level stratified splitting will be used.")

    exclusions = {TARGET_COLUMN}
    exclusions.update([c for c in DROP_COLUMNS if c in df.columns])
    if group_column is not None:
        exclusions.add(group_column)
    predictor_columns = [c for c in df.columns if c not in exclusions]

    # Preserve the original pipeline's column-order feature-set boundary.
    matching_echo_columns = [
        c for c in df.columns
        if c.casefold().startswith(ECHO_START_PREFIX.casefold())
    ]
    if len(matching_echo_columns) != 1:
        raise ValueError(
            "Could not uniquely identify the echo/clinical boundary.\n"
            f"Prefix: {ECHO_START_PREFIX!r}\n"
            f"Matches: {matching_echo_columns}\n"
            f"Workbook columns: {df.columns.tolist()}"
        )

    echo_start_column = matching_echo_columns[0]
    echo_start_position = list(df.columns).index(echo_start_column)

    clinical_columns = [
        c for c in predictor_columns
        if list(df.columns).index(c) < echo_start_position
    ]
    echo_columns = [
        c for c in predictor_columns
        if list(df.columns).index(c) >= echo_start_position
    ]
    all_columns = predictor_columns.copy()

    feature_sets = OrderedDict([
        ("All 26", all_columns),
        ("Echo 19", echo_columns),
        ("Clinical 7", clinical_columns),
    ])

    counts = {
        "All 26": len(all_columns),
        "Echo 19": len(echo_columns),
        "Clinical 7": len(clinical_columns),
    }
    expected = {
        "All 26": EXPECTED_ALL_FEATURES,
        "Echo 19": EXPECTED_ECHO_FEATURES,
        "Clinical 7": EXPECTED_CLINICAL_FEATURES,
    }
    print("\nFEATURE SET INVENTORY")
    for name in FEATURE_SET_ORDER:
        print(f"{name}: {counts[name]} variables")
    if counts != expected:
        raise ValueError(
            "The workbook's feature counts do not match the original study split.\n"
            f"Expected: {expected}\n"
            f"Observed: {counts}\n"
            "Check the echo boundary, DROP_COLUMNS, and identifier detection."
        )

    X_all, numeric_all, categorical_all = configure_feature_types(
        df[all_columns]
    )
    print("\nCategorical predictors found:", categorical_all)
    print("Numerical predictors found:", numeric_all)

    # Demographic/baseline characteristics are based on the proposal's
    # demographic fields and the study's clinical/echocardiographic predictors.
    demographic_df = df.copy()
    baseline_table, demographic_file = create_demographic_analysis(
        demographic_df,
        y,
        all_columns,
        clinical_columns,
        echo_columns,
        folders["root"],
    )
    print("\nBASELINE CHARACTERISTICS PREVIEW")
    display(baseline_table)

    # One common hold-out split and one common CV partition are reused across
    # all three feature sets to make comparisons fair.
    train_idx, test_idx, cv_splits, is_grouped, actual_cv_folds = (
        get_holdout_and_cv_splits(
            X_all,
            y,
            groups=groups,
        )
    )
    X_train_all = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_all = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx] if groups is not None else None

    print("\nDATA PARTITION")
    print("Total:", len(y))
    print("Training:", len(y_train))
    print("Held-out test:", len(y_test))
    print("CV folds:", actual_cv_folds)
    print("Group-aware:", is_grouped)
    print("Training class counts:", pd.Series(y_train).value_counts().sort_index().to_dict())
    print("Test class counts:", pd.Series(y_test).value_counts().sort_index().to_dict())

    summary_rows = []
    summary_numeric_rows = []
    cv_fold_rows = []
    roc_records = {}
    fitted_models = {}
    prediction_frame = pd.DataFrame({
        "Original_Row_Index": df.index[test_idx],
        "True_Label": y_test,
    })
    confusion_rows = []
    svm_diagnostic_rows = []
    shap_summary_rows = []
    tuning_tables = []
    selected_param_rows = []

    for feature_set_name in FEATURE_SET_ORDER:
        feature_columns = [c for c in feature_sets[feature_set_name] if c in X_all.columns]
        numeric_cols = [c for c in numeric_all if c in feature_columns]
        categorical_cols = [c for c in categorical_all if c in feature_columns]

        X_train = X_train_all[feature_columns].copy()
        X_test = X_test_all[feature_columns].copy()

        print("\n" + "=" * 88)
        print(f"FEATURE SET: {feature_set_name} ({len(feature_columns)} variables)")
        print("=" * 88)

        base_models = create_models()

        for model_name, base_model in base_models.items():
            print(f"\nTraining {model_name} — {feature_set_name}")

            # Jupyter BASE behaviour: tune once on the development set (inner CV
            # + one-SE rule), then reuse the chosen parameters for CV and test.
            if TUNE_HYPERPARAMETERS:
                tuned_params, tuning_details = tune_model_with_inner_cv(
                    X_train, y_train, model_name, base_model,
                    feature_columns, numeric_cols, categorical_cols,
                    groups_development=groups_train,
                )
                tuning_details.insert(0, "Feature_Set", feature_set_name)
                tuning_tables.append(tuning_details)
                print(f"  Selected hyperparameters: {tuned_params}")
            else:
                tuned_params = {}
            selected_param_rows.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Tuned": bool(TUNE_HYPERPARAMETERS),
                "Selected_Parameters": repr(tuned_params) if tuned_params else "fixed starting values",
            })

            fold_metric_values = {metric: [] for metric in METRICS}
            for fold_no, (fold_train_idx, fold_val_idx) in enumerate(cv_splits, start=1):
                X_fold_train = X_train.iloc[fold_train_idx]
                X_fold_val = X_train.iloc[fold_val_idx]
                y_fold_train = y_train[fold_train_idx]
                y_fold_val = y_train[fold_val_idx]

                # Model-specific XGBoost imbalance weight is recalculated
                # from each fold's training sample.
                fold_models = create_models()
                fold_model = clone(fold_models[model_name]).set_params(**tuned_params)
                fold_pipeline, _ = build_model_pipeline(
                    fold_model,
                    model_name,
                    feature_columns,
                    numeric_cols,
                    categorical_cols,
                    y_fold_train,
                )
                fold_pipeline.fit(X_fold_train, y_fold_train)
                fold_prediction, fold_probability, fold_auc_score = get_model_outputs(
                    fold_pipeline, X_fold_val, model_name
                )
                fold_metrics, _ = calculate_metrics(
                    y_fold_val, fold_prediction, fold_auc_score
                )

                for metric in METRICS:
                    fold_metric_values[metric].append(fold_metrics[metric])

                cv_fold_rows.append({
                    "Feature_Set": feature_set_name,
                    "Model": model_name,
                    "Fold": fold_no,
                    **fold_metrics,
                })
                print(
                    f"  Fold {fold_no}/{actual_cv_folds}: "
                    f"AUC={fold_metrics['AUC']:.3f}, "
                    f"Accuracy={fold_metrics['Accuracy']:.3f}"
                )

            # Final model: fit once on all 80% development observations.
            final_pipeline, _ = build_model_pipeline(
                clone(base_model).set_params(**tuned_params),
                model_name,
                feature_columns,
                numeric_cols,
                categorical_cols,
                y_train,
            )
            final_pipeline.fit(X_train, y_train)
            test_prediction, test_probability, test_auc_score = get_model_outputs(
                final_pipeline, X_test, model_name
            )
            test_metrics, _ = calculate_metrics(
                y_test, test_prediction, test_auc_score
            )

            key = (feature_set_name, model_name)
            fitted_models[key] = final_pipeline
            prediction_prefix = safe_filename(feature_set_name) + "_" + safe_filename(model_name)
            prediction_frame[f"{prediction_prefix}_Probability"] = test_probability
            prediction_frame[f"{prediction_prefix}_Prediction"] = test_prediction
            prediction_frame[f"{prediction_prefix}_AUC_Score"] = test_auc_score

            fpr, tpr, _ = roc_curve(y_test, test_auc_score)
            auc_value = float(roc_auc_score(y_test, test_auc_score))
            roc_records[key] = (fpr, tpr, auc_value)

            tn, fp, fn, tp = confusion_matrix(
                y_test, test_prediction, labels=[0, 1]
            ).ravel()
            confusion_rows.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            })

            if model_name == "SVM":
                probability_auc = float(roc_auc_score(y_test, test_probability))
                decision_auc = float(roc_auc_score(y_test, test_auc_score))
                svm_diagnostic_rows.append({
                    "Feature_Set": feature_set_name,
                    "Model": model_name,
                    "SVC_Classes": repr(final_pipeline.named_steps["classifier"].classes_.tolist()),
                    "N_Predicted_Control": int(np.sum(test_prediction == 0)),
                    "N_Predicted_Amyloidosis": int(np.sum(test_prediction == 1)),
                    "N_Probability_Greater_Or_Equal_0_5": int(np.sum(test_probability >= 0.5)),
                    "N_Probability_Less_Than_0_5": int(np.sum(test_probability < 0.5)),
                    "AUC_From_Predict_Probability": probability_auc,
                    "AUC_From_Oriented_Decision_Function": decision_auc,
                    "Decision_Function_Min": float(np.min(test_auc_score)),
                    "Decision_Function_Max": float(np.max(test_auc_score)),
                    "Probability_Min": float(np.min(test_probability)),
                    "Probability_Max": float(np.max(test_probability)),
                })
                print(
                    "  SVM scoring check: "
                    f"decision-function AUC={decision_auc:.3f}; "
                    f"predict_proba AUC={probability_auc:.3f}; "
                    f"predict() positive={int(np.sum(test_prediction == 1))}/{len(test_prediction)}; "
                    f"probability>=0.5={int(np.sum(test_probability >= 0.5))}/{len(test_probability)}"
                )

            row = {
                "Feature_Set": feature_set_name,
                "N_Features": len(feature_columns),
                "Model": model_name,
            }
            numeric_row = {
                "Feature_Set": feature_set_name,
                "N_Features": len(feature_columns),
                "Model": model_name,
            }
            for metric in METRICS:
                mean_value, sd_value, formatted = metric_mean_sd(
                    fold_metric_values[metric]
                )
                row[f"CV {metric} (mean ± SD)"] = formatted
                row[f"Test {metric}"] = round(test_metrics[metric], 4)
                numeric_row[f"CV_{metric}_Mean"] = mean_value
                numeric_row[f"CV_{metric}_SD"] = sd_value
                numeric_row[f"Test_{metric}"] = test_metrics[metric]

            summary_rows.append(row)
            summary_numeric_rows.append(numeric_row)

            model_path = folders["models"] / (
                safe_filename(feature_set_name) + "__" +
                safe_filename(model_name) + ".joblib"
            )
            joblib.dump({
                "pipeline": final_pipeline,
                "feature_set": feature_set_name,
                "model": model_name,
                "tuned_hyperparameters": tuned_params,
                "feature_columns": feature_columns,
                "categorical_columns": categorical_cols,
                "numeric_columns": numeric_cols,
                "target_column": TARGET_COLUMN,
                "positive_class": 1,
                "classification_threshold_for_probability_models": 0.50,
                "classification_rule": "pipeline.predict(X)",
                "auc_score_method": (
                    "positive-class-oriented decision_function" if model_name == "SVM"
                    else "positive-class predict_proba"
                ),
            }, model_path)

            print(
                f"  Held-out test: AUC={test_metrics['AUC']:.3f}, "
                f"Accuracy={test_metrics['Accuracy']:.3f}, "
                f"Sensitivity={test_metrics['Sensitivity']:.3f}, "
                f"Specificity={test_metrics['Specificity']:.3f}, "
                f"PPV={test_metrics['PPV']:.3f}, "
                f"NPV={test_metrics['NPV']:.3f}, "
                f"F1={test_metrics['F1']:.3f}"
            )

            # SHAP is run on the finalized model and real held-out patients only.
            if RUN_SHAP:
                shap_folder = folders["shap"] / safe_filename(feature_set_name) / safe_filename(model_name)
                shap_folder.mkdir(parents=True, exist_ok=True)
                shap_title = f"{feature_set_name} — {model_name}"
                try:
                    if model_name in ["Random Forest", "XGBoost"]:
                        shap_importance = run_tree_shap(
                            final_pipeline,
                            X_train,
                            X_test,
                            feature_columns,
                            numeric_cols,
                            categorical_cols,
                            shap_folder,
                            shap_title,
                        )
                        explainer_name = "TreeSHAP"
                    else:
                        shap_importance = run_permutation_shap(
                            final_pipeline,
                            X_train,
                            X_test,
                            feature_columns,
                            numeric_cols,
                            categorical_cols,
                            shap_folder,
                            shap_title,
                        )
                        explainer_name = "PermutationSHAP"

                    for _, imp_row in shap_importance.iterrows():
                        shap_summary_rows.append({
                            "Feature_Set": feature_set_name,
                            "Model": model_name,
                            "SHAP_Explainer": explainer_name,
                            "Feature": imp_row["Feature"],
                            "Mean_Absolute_SHAP": imp_row["Mean_Absolute_SHAP"],
                        })
                    print(f"  SHAP completed using {explainer_name}.")
                except Exception as shap_error:
                    error_text = repr(shap_error)
                    (shap_folder / "SHAP_ERROR.txt").write_text(
                        error_text,
                        encoding="utf-8",
                    )
                    print(
                        f"  WARNING: SHAP failed for {model_name} / "
                        f"{feature_set_name}: {error_text}"
                    )

        # Print a compact per-feature-set performance table.
        current_summary = pd.DataFrame([
            row for row in summary_rows if row["Feature_Set"] == feature_set_name
        ])
        print(f"\n{feature_set_name} — MODEL PERFORMANCE")
        display(current_summary)

    # ==================================================================
    # 8. SAVE PERFORMANCE TABLES AND PREDICTIONS
    # ==================================================================

    summary_df = pd.DataFrame(summary_rows)
    summary_numeric_df = pd.DataFrame(summary_numeric_rows)
    cv_fold_df = pd.DataFrame(cv_fold_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    summary_df.to_csv(
        folders["tables"] / "model_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_numeric_df.to_csv(
        folders["tables"] / "model_performance_numeric.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_fold_df.to_csv(
        folders["tables"] / "cross_validation_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confusion_df.to_csv(
        folders["tables"] / "heldout_confusion_matrix_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    svm_diagnostics_df = pd.DataFrame(svm_diagnostic_rows)
    if not svm_diagnostics_df.empty:
        svm_diagnostics_df.to_csv(
            folders["tables"] / "svm_scoring_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nSVM SCORING DIAGNOSTICS")
        display(svm_diagnostics_df.round(4))
    selected_params_df = pd.DataFrame(selected_param_rows)
    selected_params_df.to_csv(
        folders["tables"] / "selected_hyperparameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tuning_candidates_df = (
        pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame()
    )
    if not tuning_candidates_df.empty:
        tuning_candidates_df.to_csv(
            folders["tables"] / "hyperparameter_tuning_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nSELECTED HYPERPARAMETERS (inner-CV + one-SE rule)")
        display(selected_params_df)
    prediction_frame.to_csv(
        folders["predictions"] / "heldout_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    performance_xlsx = folders["root"] / "model_performance_summary.xlsx"
    with pd.ExcelWriter(performance_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="All_12_Models", index=False)
        for feature_set_name in FEATURE_SET_ORDER:
            sheet_name = {
                "All 26": "All_26",
                "Echo 19": "Echo_19",
                "Clinical 7": "Clinical_7",
            }[feature_set_name]
            summary_df[
                summary_df["Feature_Set"] == feature_set_name
            ].to_excel(writer, sheet_name=sheet_name, index=False)
        cv_fold_df.to_excel(writer, sheet_name="CV_Fold_Metrics", index=False)
        confusion_df.to_excel(writer, sheet_name="Test_Confusion_Matrices", index=False)
        if not svm_diagnostics_df.empty:
            svm_diagnostics_df.to_excel(writer, sheet_name="SVM_Scoring_Diagnostics", index=False)
        selected_params_df.to_excel(writer, sheet_name="Selected_Hyperparameters", index=False)
        if not tuning_candidates_df.empty:
            tuning_candidates_df.to_excel(writer, sheet_name="Tuning_Candidates", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_len + 2, 12), 42
                )
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)

    print("\nOVERALL 12-MODEL COMPARISON")
    display(summary_df)

    # ==================================================================
    # 9. ALL-12-MODEL ROC CURVES AND MODEL-SPECIFIC METRIC COMPARISONS
    # ==================================================================

    plot_all_roc_curves(roc_records, folders["plots"])
    plot_model_metric_comparisons(
        summary_numeric_df,
        folders["metric_plots"],
    )

    print("\nSaved combined ROC plot:")
    print(folders["plots"] / "ROC_AUC_all_12_models.png")
    print("Saved four model-specific metric comparison plots:")
    print(folders["metric_plots"])

    # Save aggregate SHAP importance, if any explanation completed.
    shap_summary_df = pd.DataFrame(shap_summary_rows)
    if not shap_summary_df.empty:
        shap_summary_df.to_csv(
            folders["shap"] / "SHAP_global_importance_all_models.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # A concise run manifest clarifies exactly what was and was not performed.
    manifest = {
        "Dataset": str(data_file),
        "Target": TARGET_COLUMN,
        "Positive_Class": "Cardiac amyloidosis (1)",
        "Negative_Class": "Control (0)",
        "Feature_Sets": {
            "All 26": all_columns,
            "Echo 19": echo_columns,
            "Clinical 7": clinical_columns,
        },
        "Models": MODEL_NAMES,
        "Total_Model_Feature_Set_Configurations": 12,
        "Declared_Categorical_Variables": CATEGORICAL_VARIABLES,
        "Categorical_Variables_Actually_Used": categorical_all,
        "Numerical_Variables_Actually_Used": numeric_all,
        "Age_Treat_As_Numeric_Override": AGE_TREAT_AS_NUMERIC,
        "Age_Numeric_Unique_Threshold": AGE_NUMERIC_UNIQUE_THRESHOLD,
        "Age_Numeric_Parse_Fraction": AGE_NUMERIC_PARSE_FRACTION,
        "Classifier_Starting_Hyperparameters": {
            "Random Forest": {"n_estimators": 300, "max_depth": 4, "min_samples_split": 8, "min_samples_leaf": 5, "max_features": "log2", "max_samples": 0.80, "class_weight": "balanced_subsample"},
            "XGBoost": {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 10.0, "gamma": 1.0, "scale_pos_weight": "training-fold negatives / positives"},
            "SVM": {"C": 0.10, "kernel": "rbf", "gamma": 0.01, "probability": True, "class_weight": "balanced"},
            "MLP": {"hidden_layer_sizes": [16], "activation": "relu", "solver": "adam", "alpha": 0.30, "learning_rate_init": 0.001, "max_iter": 1000, "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 20, "tol": 0.0001},
        },
        "Preprocessing_Order": [
            "numeric: median imputation",
            "categorical: constant imputation with __MISSING__",
            "categorical: OrdinalEncoder(handle_unknown=use_encoded_value, unknown_value=-1)",
            "categorical: OneHotEncoder(handle_unknown=ignore, drop=None)",
            "StandardScaler after one-hot encoding",
        ],
        "Echo_Boundary_Column": echo_start_column,
        "Holdout_Test_Size": TEST_SIZE,
        "CV_Folds_Used": actual_cv_folds,
        "Group_Aware_Splitting": bool(is_grouped),
        "Feature_Selection": False,
        "LASSO": False,
        "Stepwise_Selection": False,
        "SMOTE": False,
        "PCA": False,
        "Hyperparameter_Tuning": bool(TUNE_HYPERPARAMETERS),
        "Hyperparameter_Tuning_Method": (
            f"{INNER_CV_FOLDS}-fold stratified inner CV on the development set "
            f"(random_state={TUNING_RANDOM_STATE}); one-SE rule={USE_ONE_SE_RULE}"
            if TUNE_HYPERPARAMETERS else "none"
        ),
        "Param_Candidates": {k: [repr(c) for c in v] for k, v in PARAM_CANDIDATES.items()},
        "Selected_Hyperparameters": selected_param_rows,
        "Nested_CV": False,
        "Harrell_Bootstrap": False,
        "Overfitting_Analyses": False,
        "Metrics": METRICS,
        "Classification_Decision_Rule": "Estimator.predict(); for SVM this uses the SVC decision margin.",
        "AUC_Ranking_Score": "SVM: positive-class-oriented decision_function; RF/XGBoost/MLP: positive-class probability.",
        "SVM_Probability_Use": "Stored for reference only; original Jupyter classification metrics use SVC.predict() and ROC-AUC uses SVC.decision_function().",
        "ROC_Group_Shadow": bool(ROC_GROUP_SHADOW),
        "SHAP_Enabled": bool(RUN_SHAP),
        "SHAP_Methods": {
            "Random Forest": "TreeSHAP",
            "XGBoost": "TreeSHAP",
            "SVM": "PermutationSHAP",
            "MLP": "PermutationSHAP",
        },
        "SHAP_Background_Size_Max": SHAP_BACKGROUND_SAMPLES,
        "SHAP_Explained_Test_Samples_Max": SHAP_EXPLAIN_SAMPLES,
        "SHAP_Permutation_Repeats": SHAP_PERMUTATION_REPEATS,
    }
    (folders["root"] / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""CARDIAC AMYLOIDOSIS ML RESULTS

Dataset: {data_file}
Outcome: {TARGET_COLUMN} (1=cardiac amyloidosis; 0=control)

Design:
- Three feature sets: All 26, Echo 19, Clinical 7
- Four classifiers: Random Forest, XGBoost, RBF-SVM, MLP
- Twelve model/feature-set configurations
- Stratified 80:20 hold-out; {actual_cv_folds}-fold CV in the development set
- Metrics: ROC-AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1
- No feature selection, LASSO, stepwise selection, PCA, SMOTE, nested CV,
  Harrell bootstrap, or separate overfitting tests.
- Hyperparameters are tuned per model and feature set on the development set
  only ({INNER_CV_FOLDS}-fold inner CV, one-SE rule) exactly as in the Jupyter
  BASE experiment; see tables/selected_hyperparameters.csv. Tuning enabled: {TUNE_HYPERPARAMETERS}.

SHAP:
- Random Forest and XGBoost: TreeSHAP
- SVM and MLP: PermutationSHAP
- Original Jupyter baseline preprocessing is retained: numeric median imputation; categorical constant imputation + OrdinalEncoder; separate OneHotEncoder; StandardScaler after one-hot encoding.
- Age type follows original Jupyter auto-detection (numeric raw ages with >10 unique values are continuous unless AGE_TREAT_AS_NUMERIC overrides this).
- Explanations are based on real observations from the held-out test partition.
- Both explainers use the numeric feature matrix produced by the fitted preprocessor
  AND post-one-hot StandardScaler.
- SVM PermutationSHAP explains the amyloidosis-oriented decision margin; MLP
  PermutationSHAP explains positive-class probability.
- For SVM, AUC/ROC and SHAP use decision_function; classification metrics use
  SVC.predict(). Platt-calibrated SVC predict_proba is retained for reference only.
- Global SHAP importance bars aggregate one-hot contributions to the original
  clinical/echocardiographic variables.

Demographic analysis:
- See 'demographic analysis.xlsx'.
- Includes proposal-specified demographic fields when present plus all study
  clinical/echocardiographic predictors, summarized by outcome group.
- P-values are descriptive comparisons only and are not used for feature selection.

ROC grouped shadow/halo setting: {ROC_GROUP_SHADOW}
"""
    (folders["root"] / "README_results.txt").write_text(readme, encoding="utf-8")

    # Zip the complete output folder for convenient Colab download.
    zip_path = Path(str(folders["root"]) + ".zip")
    if CREATE_RESULTS_ZIP:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for file_path in folders["root"].rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(folders["root"].parent))
        print("\nResults ZIP created:", zip_path)
        if IN_COLAB and AUTO_DOWNLOAD_ZIP:
            colab_files.download(str(zip_path))

    print("\n" + "=" * 88)
    print("ANALYSIS COMPLETED")
    print("=" * 88)
    print("Twelve model-feature-set combinations completed.")
    print("Demographic analysis:", demographic_file)
    print("Performance workbook:", performance_xlsx)
    print("Plots:", folders["plots"])
    print("Metric comparisons:", folders["metric_plots"])
    print("SHAP:", folders["shap"] if RUN_SHAP else "Disabled")
    print("All results:", folders["root"])

    return {
        "summary": summary_df,
        "summary_numeric": summary_numeric_df,
        "cv_fold_metrics": cv_fold_df,
        "demographic_file": demographic_file,
        "performance_file": performance_xlsx,
        "output_dir": folders["root"],
        "roc_records": roc_records,
    }


# ======================================================================
# 10. RUN
# ======================================================================

if __name__ == "__main__":
    RESULTS = main()
