# Module 2 — Analytics Pipeline

Run these files in order:

```powershell
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` loads `sns.load_dataset("titanic")` exactly once, immediately saves `titanic.csv`, cleans the data, and writes the EDA results and charts.

`02_modeling.py` reads only `titanic.csv`, then performs the modeling pipeline with train-only preprocessing.

Generated artifacts include:
- `outputs/eda_report.md`
- EDA charts
- `outputs/model_comparison.csv`
- `outputs/imbalance_comparison.csv`
- `outputs/modeling_report.md`
- `models/best_pipeline.joblib`
- `outputs/regression_residuals.png`
