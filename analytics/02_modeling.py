from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "titanic.csv")

# Use a compact feature set that matches the brief's required categorical/numeric examples.
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

X = df[features].copy()
y = df[target].copy()

print("Class balance:")
print(y.value_counts())
print(y.value_counts(normalize=True).mul(100).round(2))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []
fitted = {}

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
        "confusion_matrix": cm.tolist()
    })

    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

# Decision tree visualization with labeled features/classes.
tree_pipe = fitted["Decision Tree"]
tree_model = tree_pipe.named_steps["model"]
tree_pre = tree_pipe.named_steps["preprocessor"]
feature_names = tree_pre.get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not survived", "Survived"],
    filled=False,
    max_depth=3
)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png")
plt.close()

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png")
plt.close()

classification_df = pd.DataFrame(results)
classification_df.to_csv(OUT / "model_comparison.csv", index=False)

# Confusion matrices.
for row in results:
    cm = np.array(row["confusion_matrix"])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(f"Confusion Matrix — {row['model']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe = row["model"].lower().replace(" ", "_")
    plt.savefig(OUT / f"confusion_{safe}.png")
    plt.close()

# Imbalance comparison using Random Forest.
imbalance_results = []

variants = {
    "baseline": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "class_weight_balanced": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
}

for label, estimator in variants.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    imbalance_results.append({
        "variant": label,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    })

smote_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])
smote_pipe.fit(X_train, y_train)
smote_pred = smote_pipe.predict(X_test)
imbalance_results.append({
    "variant": "SMOTE_training_fold_only",
    "precision": precision_score(y_test, smote_pred, zero_division=0),
    "recall": recall_score(y_test, smote_pred, zero_division=0),
    "f1": f1_score(y_test, smote_pred, zero_division=0)
})

imbalance_df = pd.DataFrame(imbalance_results)
imbalance_df.to_csv(OUT / "imbalance_comparison.csv", index=False)

# Hyperparameter tuning with OOB-enabled final estimator.
rf_base = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    bootstrap=True
)

rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_base)
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    rf_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
best_rf_model = best_rf.named_steps["model"]
best_params = grid.best_params_
oob_score = best_rf_model.oob_score_

# Evaluate tuned model.
tuned_pred = best_rf.predict(X_test)
tuned_prob = best_rf.predict_proba(X_test)[:, 1]
tuned_metrics = {
    "accuracy": accuracy_score(y_test, tuned_pred),
    "precision": precision_score(y_test, tuned_pred, zero_division=0),
    "recall": recall_score(y_test, tuned_pred, zero_division=0),
    "f1": f1_score(y_test, tuned_pred, zero_division=0),
    "auc": roc_auc_score(y_test, tuned_prob)
}

# Regression side-task: predict fare.
reg_features = ["pclass", "age", "sibsp", "parch"]
X_reg = df[reg_features]
y_reg = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
reg_pipe.fit(Xr_train, yr_train)
reg_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, reg_pred)
rmse = np.sqrt(mean_squared_error(yr_test, reg_pred))
r2 = r2_score(yr_test, reg_pred)
n = len(yr_test)
p = len(reg_features)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

residuals = yr_test - reg_pred
plt.figure(figsize=(7, 5))
sns.scatterplot(x=reg_pred, y=residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "regression_residuals.png")
plt.close()

# Save the complete best pipeline.
joblib.dump(best_rf, MODELS / "best_pipeline.joblib")

# Reload and verify prediction on raw input.
reloaded = joblib.load(MODELS / "best_pipeline.joblib")
reload_prediction = reloaded.predict(X_test.head(3))

report = []
report.append("# Modeling Report\n")
report.append("## Class balance\n")
report.append(y.value_counts().to_string())
report.append("\n\n")
report.append(
    "A stratified split was used because the target contains two classes and "
    "stratification preserves their observed proportions in train and test.\n"
)

report.append("\n## Classification comparison\n")
report.append(classification_df.to_string(index=False))

report.append("\n## Imbalance comparison\n")
report.append(imbalance_df.to_string(index=False))
report.append(
    "\nSMOTE was applied inside an imbalanced-learn pipeline after the train/test "
    "split, so oversampling is limited to the training fold.\n"
)

report.append("\n## GridSearchCV\n")
report.append(f"Best parameters: {best_params}\n")
report.append(f"OOB score: {oob_score:.4f}\n")
report.append(f"Tuned test metrics: {tuned_metrics}\n")

report.append("\n## Regression\n")
report.append(f"MAE: {mae:.4f}\n")
report.append(f"RMSE: {rmse:.4f}\n")
report.append(f"R2: {r2:.4f}\n")
report.append(f"Adjusted R2: {adjusted_r2:.4f}\n")
report.append(
    "\nHeteroscedasticity conclusion: inspect the residual plot. "
    "A widening or narrowing residual spread across fitted values indicates "
    "heteroscedasticity; a roughly constant random spread does not.\n"
)

report.append("\n## Reload check\n")
report.append(f"Reloaded pipeline predictions on raw rows: {reload_prediction.tolist()}\n")

# Keep the required metric groups separate.
report.append("\n## Metric-group structure\n")
report.append(
    "Classification metrics (accuracy, precision, recall, F1, AUC) are kept "
    "separate from regression metrics (MAE, RMSE, R2, Adjusted R2) because the "
    "two model types use different scales and objectives.\n"
)

(OUT / "modeling_report.md").write_text("\n".join(report), encoding="utf-8")
print((OUT / "modeling_report.md").read_text())
