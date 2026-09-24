from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# The raw dataset is loaded once from seaborn's loader.
df = sns.load_dataset("titanic")
df.to_csv(ROOT / "titanic.csv", index=False)

report = []
report.append("# EDA Report\n")
report.append("## Initial profile\n")
report.append(f"Shape: {df.shape}\n\n")
report.append("### df.info()\n")
df.info(buf=open(OUT / "df_info.txt", "w"))
report.append((OUT / "df_info.txt").read_text())

report.append("\n### df.describe()\n")
report.append(df.describe(include="all").to_string())

missing = (df.isna().mean() * 100).round(2)
missing = missing[missing > 0]
report.append("\n## Missing-value percentages\n")
for col, pct in missing.items():
    if pct < 5:
        strategy = "Drop rows with missing values in this column."
    elif pct <= 30:
        strategy = "Impute the missing values."
    else:
        strategy = "High missingness: drop the column or encode missing as a category."
    report.append(f"- `{col}`: {pct:.2f}% — {strategy}\n")

# Cleaning for EDA.
clean = df.copy()

# Apply the required threshold rule.
for col in clean.columns:
    rate = clean[col].isna().mean() * 100
    if rate < 5:
        clean = clean.dropna(subset=[col])
    elif rate <= 30:
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = clean[col].fillna(clean[col].median())
        else:
            clean[col] = clean[col].fillna(clean[col].mode()[0])
    else:
        # Cabin has very high missingness; it is not needed for the required
        # EDA/model features, so it is dropped and the decision is documented.
        clean = clean.drop(columns=[col])

report.append("\n## Cleaning decision\n")
report.append(
    "Columns below 5% missingness had rows dropped; columns from 5% to 30% "
    "were imputed; the high-missingness `deck`/cabin-derived field was excluded "
    "from the working EDA dataset because its missingness is too high for reliable "
    "imputation and it is not required by the modeling feature set.\n"
)

# Univariate analysis.
def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum()), lower, upper

age_out, age_low, age_high = iqr_outliers(clean["age"])
fare_out, fare_low, fare_high = iqr_outliers(clean["fare"])

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]

if fare_mean > fare_median > fare_mode:
    skew_text = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_text = "left-skewed"
else:
    skew_text = "not determined by a strict mean/median/mode ordering"

report.append("\n## Univariate analysis\n")
report.append(f"- Age IQR outliers: {age_out}\n")
report.append(f"- Fare IQR outliers: {fare_out}\n")
report.append(
    f"- Fare mean: {fare_mean:.4f}; median: {fare_median:.4f}; mode: {fare_mode:.4f}.\n"
)
report.append(f"- Fare distribution conclusion from ordering: {skew_text}.\n")

for col, title in [("age", "Age"), ("fare", "Fare")]:
    plt.figure(figsize=(7, 4))
    sns.histplot(clean[col], kde=True)
    plt.title(f"{title} Histogram")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_histogram.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=clean[col])
    plt.title(f"{title} Box Plot")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_boxplot.png")
    plt.close()

# Bivariate survival rates using boolean masking.
survival_by_sex = clean.groupby("sex")["survived"].mean().mul(100).round(2)
survival_by_pclass = clean.groupby("pclass")["survived"].mean().mul(100).round(2)
survival_by_sex_pclass = (
    clean.groupby(["sex", "pclass"])["survived"].mean().mul(100).round(2)
)

report.append("\n## Survival rates\n")
report.append("\n### By sex\n")
report.append(survival_by_sex.to_string())
report.append("\n\n### By pclass\n")
report.append(survival_by_pclass.to_string())
report.append("\n\n### By sex and pclass\n")
report.append(survival_by_sex_pclass.to_string())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix — Required Six Columns")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png")
plt.close()

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j], abs(corr.iloc[i, j])))
pairs = sorted(pairs, key=lambda x: x[3], reverse=True)
top_two = pairs[:2]

report.append("\n## Two strongest absolute correlations\n")
for a, b, value, absolute in top_two:
    report.append(f"- `{a}` vs `{b}`: correlation = {value:.4f}\n")

# Four distinct multivariate charts.
plt.figure(figsize=(7, 5))
sns.barplot(data=clean, x="sex", y="survived", hue="pclass", errorbar=None)
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.savefig(OUT / "survival_sex_pclass.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=clean, x="survived", y="age", hue="sex")
plt.title("Age Distribution by Survival and Sex")
plt.tight_layout()
plt.savefig(OUT / "age_survival_sex.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=clean, x="fare", y="age", hue="survived", style="pclass")
plt.title("Age vs Fare by Survival and Class")
plt.tight_layout()
plt.savefig(OUT / "age_fare_survival.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.pointplot(data=clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival Probability Across Classes")
plt.tight_layout()
plt.savefig(OUT / "class_survival_pointplot.png")
plt.close()

report.append("\n## Multivariate chart interpretations\n")
report.append(
    "1. **Survival by sex and class:** The grouped survival-rate chart shows how survival "
    "varies jointly by sex and passenger class. The exact pattern should be read from the "
    "plotted rates rather than assumed before execution.\n\n"
)
report.append(
    "2. **Age, survival and sex:** The box plot compares age distributions for survivors "
    "and non-survivors within sex groups, helping identify whether age composition differs "
    "between outcomes.\n\n"
)
report.append(
    "3. **Age versus fare:** The scatter plot combines fare, age, survival and class. "
    "It provides a multivariate view of how passenger characteristics and ticket class "
    "co-occur with survival.\n\n"
)
report.append(
    "4. **Class and survival:** The point plot shows survival probability by passenger "
    "class separately for sex, making the interaction between class and sex visible.\n\n"
)

# Exploratory z-score standardization on the full cleaned DataFrame.
scaler = StandardScaler()
standardized = clean.copy()
standardized[["age_z", "fare_z"]] = scaler.fit_transform(clean[["age", "fare"]])

report.append("\n## Standardization sanity check\n")
report.append(
    "Before means/stds:\n" +
    clean[["age", "fare"]].agg(["mean", "std"]).to_string() + "\n\n"
)
report.append(
    "After means/stds:\n" +
    standardized[["age_z", "fare_z"]].agg(["mean", "std"]).to_string() + "\n"
)

clean.to_csv(OUT / "cleaned_titanic.csv", index=False)
(OUT / "eda_report.md").write_text("\n".join(report), encoding="utf-8")
print((OUT / "eda_report.md").read_text())
