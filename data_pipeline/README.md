# Module 1 — Data Pipeline

This module scrapes the first five pages of Books to Scrape, cleans the fields, converts GBP to INR at the fixed project rate of 105.50, stores the data in a normalized SQLite database, and demonstrates SQL plus pandas analysis.

Run:

```powershell
python run_pipeline.py
```

Generated files:
- `data/books_clean.csv`
- `database/zepto_books.db`
- `sql_outputs.txt`
