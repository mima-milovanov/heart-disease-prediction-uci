import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import mutual_info_classif
import os


# ============================================================
# Global Settings
# ============================================================

SEED = 42
REPORTS_DIR = "reports"
METRICS_CSV = os.path.join(REPORTS_DIR, "metrics_summary.csv")
ROC_PNG = os.path.join(REPORTS_DIR, "roc_curve.png")
MI_CSV = os.path.join(REPORTS_DIR, "mi_interactions.csv")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# Load & Basic cleaning
# ============================================================

df = pd.read_csv("data/heart_disease_uci.csv")

# Drop irrelevant columns
df = df.drop(["id", "dataset"], axis=1)

# Create binary target: 0 = no disease, 1 = disease
df["target"] = (df["num"] > 0).astype(int)
df = df.drop("num", axis=1)

df_clean = df.copy()

# Replace 0 values with median (for trestbps and chol)
for col in ["trestbps", "chol"]:
    median_value = df_clean[col].median()
    df_clean[col] = df_clean[col].replace(0, median_value)

# ============================================================
# Basic EDA
# ============================================================

sns.countplot(x="target", data=df)
plt.title("Target Distribution (Presence of Heart Disease)")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# ============================================================
# Feature Engineering & Mutual Information Screening
# ============================================================
#
if "cp" in df_clean.columns and df_clean["cp"].dtype == "object":
    df_clean["cp"] = df_clean["cp"].astype("category").cat.codes

# Make sure that key columns are numeric
for col in ["oldpeak", "trestbps", "thalch", "chol", "exang", "age"]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

df_clean["age_chol_interaction"] = df_clean["age"] * df_clean["chol"]
df_clean["age_trestbps_interaction"] = df_clean["age"] * df_clean["trestbps"]
df_clean["oldpeak_thalch_interaction"] = df_clean["oldpeak"] * df_clean["thalch"]
df_clean["ca_thalch_interaction"] = df_clean["ca"] * df_clean["thalch"]
df_clean["ca_oldpeak_interaction"] = df_clean["ca"] * df_clean["oldpeak"]
df_clean["ca_exang_interaction"] = df_clean["ca"] * df_clean["exang"]
df_clean["trestbps_thalch_interaction"] = df_clean["trestbps"] * df_clean["thalch"]
df_clean["exang_thalch_interaction"] = df_clean["exang"] * df_clean["thalch"]
df_clean["oldpeak_exang_interaction"] = df_clean["oldpeak"] * df_clean["exang"]

# Build cp_oldpeak_interaction safely
if "cp_oldpeak_interaction" in df_clean.columns:
    df_clean.drop(columns=["cp_oldpeak_interaction"], inplace=True)
df_clean["cp_oldpeak_interaction"] = df_clean["cp"] * df_clean["oldpeak"]


# Collect interaction columns
interaction_cols = [c for c in df_clean.columns if "_interaction" in c]
print(f"Found {len(interaction_cols)} interaction features:\n{interaction_cols}")

X_inter = df_clean[interaction_cols].copy()
# Drop interactions that are entirely NaN
all_nan_cols = [c for c in interaction_cols if X_inter[c].isna().all()]
if all_nan_cols:
    print(f"Dropping all-NaN interactions (no observed values): {all_nan_cols}")
    X_inter.drop(columns=all_nan_cols, inplace=True)

# If nothing left, stop gracefully
if X_inter.shape[1] == 0:
    print("No valid interaction features left after filtering.")
# Else: Impute and compute MI
imputer = SimpleImputer(strategy="median")
X_inter_imputed = imputer.fit_transform(X_inter)

# Build MI for each interaction with the target
mi_values = mutual_info_classif(X_inter_imputed, df_clean["target"], random_state=SEED)

mi_df = pd.DataFrame({
    "Feature": X_inter.columns,
    "Mutual_Information": mi_values
}).sort_values(by="Mutual_Information", ascending=False).reset_index(drop=True)

print("\n=== Mutual Information Ranking (interactions vs target) ===")
print(mi_df.round(4).to_string(index=False))

# Define threshold + always keep 'trestbps_thalch_interaction'
MI_THRESHOLD = 0.05
keep_interactions = mi_df[mi_df["Mutual_Information"] >= MI_THRESHOLD]["Feature"].tolist()

if "trestbps_thalch_interaction" not in keep_interactions and \
        "trestbps_thalch_interaction" in df_clean.columns:
    keep_interactions.append("trestbps_thalch_interaction")

drop_interactions = [c for c in interaction_cols if c not in keep_interactions]
print(f"\n Keep interactions with MI ≥ {MI_THRESHOLD}: {keep_interactions}")
if drop_interactions:
    print(f"Dropping weak interactions: {drop_interactions}")
    df_clean.drop(columns=drop_interactions, inplace=True)

# ============================================================
# Define X/y & Column Types (explicit lists)
# ============================================================
# Map sex to 0/1 and keep as category
if "sex" in df_clean.columns and df_clean["sex"].dtype == "object":
    df_clean["sex"] = df_clean["sex"].map({"Female": 0, "Male": 1})

# Mark known categoricals so they go to One-Hot
CAT_COLS = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"] if c in df_clean.columns]

for c in CAT_COLS:
    df_clean[c] = df_clean[c].astype("object")

X = df_clean.drop("target", axis=1)
y = df_clean["target"]

categorical_features = CAT_COLS[:]
numeric_features = [c for c in X.columns if c not in categorical_features]

# Sanity check: ensure numeric_features are truly numeric (no object/category)
bad_numeric = X[numeric_features].select_dtypes(include=["object", "category"]).columns.tolist()
if bad_numeric:
    raise ValueError(f"These ended up in numeric_features but are non-numeric: {bad_numeric}")

# ============================================================
# Quick EDA: Check new interaction distributions
# ============================================================

df_clean[keep_interactions].hist(figsize=(15, 10), bins=20)
plt.suptitle("Distribution of New Interaction Features", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation map with new columns
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean[keep_interactions + ["target"]].corr(), annot=True, cmap="RdBu", center=0)
plt.title("Correlation of New Interactions with Target")
plt.show()

# ============================================================
# Preprocessor (Median Impute + Scale + OneHot)
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define robust Cross-Validation strategy
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

# ============================================================
# Model Tuning
# ============================================================

models_and_params = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=SEED), {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs']
    }),

    "Decision Tree": (DecisionTreeClassifier(random_state=SEED), {
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }),

    "Random Forest": (RandomForestClassifier(random_state=SEED), {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2, 4]
    }),

    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED), {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),

    "CatBoost": (CatBoostClassifier(random_seed=42, verbose=0, allow_writing_files=False), {
        'iterations': [100, 300, 500],
        'learning_rate': [0.05, 0.1, 0.2],
        'depth': [4, 6, 8]
    })
}

# ============================================================
# Model - Stacking Ensemble Setup
# ============================================================

base_models = [
    ('lr', LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=5000, random_state=SEED)),
    ('rf', RandomForestClassifier(max_depth=5, n_estimators=100, random_state=SEED)),
    ('xgb',
     XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=100, eval_metric="logloss",
                   random_state=SEED)),
    ('cat', CatBoostClassifier(depth=4, iterations=100, learning_rate=0.05, random_seed=SEED, verbose=0, allow_writing_files=False))
]

# Final meta-classifier
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=5000),
    cv=5,
    n_jobs=-1
)

# Add Stacking Ensemble to the models dictionary for training/reporting
final_models_and_params = models_and_params.copy()
final_models_and_params["Stacking Ensemble"] = (stack_model, {})

# ============================================================
# Evaluation
# ============================================================

results = {}
best_models_for_plot = {}

for name, (model, params_grid) in final_models_and_params.items():
    print(f"\n=== {name} ===")

    if params_grid:
        print(f"Tuning with GridSearchCV...")
        clf = GridSearchCV(model, params_grid, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
        clf.fit(X_train_processed, y_train)
        best_model = clf.best_estimator_
        print(f"Best params: {clf.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_processed, y_train)

    y_pred = best_model.predict(X_test_processed)
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        y_proba = best_model.decision_function(X_test_processed)

    auc = roc_auc_score(y_test, y_proba)
    rep = classification_report(y_test, y_pred, output_dict=True)
    rep_df = pd.DataFrame(rep).T

    print("\nClassification Report:")
    print(rep_df.fillna("").to_string())

    # Save metrics for summary
    results[name] = {
        "ROC-AUC": auc,
        "F1-score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    # Store fitted model for ROC plotting
    best_models_for_plot[name] = best_model

# Summary table → CSV
results_df = pd.DataFrame(results).T[["ROC-AUC", "F1-score", "Accuracy", "Precision", "Recall"]]
print("\n=== Model Comparison Summary ===")
print(results_df.round(3).to_string())
results_df.to_csv(METRICS_CSV, index=True)
print(f"\nSaved metrics to: {METRICS_CSV}")

# ROC Curve Comparison → PNG
plt.figure(figsize=(10, 8))
for name, best_model in best_models_for_plot.items():
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test_processed)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(ROC_PNG, dpi=200, bbox_inches="tight")
print(f"Saved ROC curves to: {ROC_PNG}")
plt.show()
