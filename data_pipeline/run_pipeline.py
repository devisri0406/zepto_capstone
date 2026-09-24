from pathlib import Path
import sqlite3
import statistics
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def scrape_pages(page_count=5):
    rows = []
    for page in range(1, page_count + 1):
        url = BASE_URL if page == 1 else f"{BASE_URL}catalogue/page-{page}.html"
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 Zepto-Capstone"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article.product_pod"):
            title = article.h3.a.get("title", "").strip()
            price_text = article.select_one(".price_color").get_text(strip=True)
            rating_class = article.select_one("p.star-rating").get("class", [])
            star_rating = next((x for x in rating_class if x in RATING_MAP), "")
            availability = article.select_one(".availability").get_text(" ", strip=True)

            # The category is taken from the page breadcrumb for each listing page.
            # The All Products catalogue does not expose a category per card, so
            # we visit the book detail page to obtain the exact category.
            book_href = article.h3.a.get("href")
            detail_url = requests.compat.urljoin(url, book_href)
            detail = requests.get(
                detail_url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 Zepto-Capstone"}
            )
            detail.raise_for_status()
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            breadcrumb = detail_soup.select("ul.breadcrumb li")
            category = breadcrumb[-2].get_text(strip=True) if len(breadcrumb) >= 2 else "Unknown"

            rows.append({
                "title": title,
                "price": price_text,
                "star_rating": star_rating,
                "availability": availability,
                "category": category,
            })
    return pd.DataFrame(rows)


def clean_data(df):
    df = df.copy()

    def parse_price(value):
        try:
            return float(str(value).replace("£", "").replace("Â", "").strip())
        except (TypeError, ValueError):
            return None

    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability"].str.contains(
        "In stock", case=False, na=False
    )

    # Numeric parsing failures are median-imputed as required.
    for col in ["price_gbp", "rating"]:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    return df[[
        "title", "price_gbp", "price_inr", "rating", "in_stock", "category"
    ]]


def create_database(df):
    db_path = DB_DIR / "zepto_books.db"
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            )
        """)

        for category in sorted(df["category"].unique()):
            conn.execute(
                "INSERT INTO categories(category_name) VALUES (?)",
                (category,)
            )

        category_map = dict(
            conn.execute("SELECT category_name, category_id FROM categories")
        )

        for row in df.itertuples(index=False):
            conn.execute("""
                INSERT INTO books
                (title, price_gbp, price_inr, rating, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row.title, row.price_gbp, row.price_inr, row.rating,
                int(row.in_stock), category_map[row.category]
            ))

    return db_path


def run_queries(db_path):
    queries = {
        "Q1 SELECT/WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
            ORDER BY rating DESC, price_gbp DESC;
        """,
        "Q2 ORDER BY": """
            SELECT title, price_gbp, price_inr
            FROM books
            ORDER BY price_inr DESC;
        """,
        "Q3 LIMIT": """
            SELECT title, rating, price_gbp
            FROM books
            ORDER BY rating DESC, price_gbp DESC
            LIMIT 10;
        """,
        "Q4 DISTINCT": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name;
        """,
        "Q5 BETWEEN": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp;
        """,
        "Q6 JOIN": """
            SELECT c.category_name, b.title, b.rating, b.price_gbp, b.price_inr
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
            LIMIT 10;
        """
    }

    output = []
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        for name, query in queries.items():
            df = pd.read_sql_query(query, conn)
            output.append(f"\n{name}\n{'=' * len(name)}\n{df.to_string(index=False)}\n")
            if name == "Q6 JOIN":
                join_sql_df = df.copy()

        # Read at least two query results into pandas using pd.read_sql.
        q1_df = pd.read_sql_query(queries["Q1 SELECT/WHERE"], conn)
        q6_df = pd.read_sql_query(queries["Q6 JOIN"], conn)

        # Reproduce the JOIN directly with in-memory DataFrames.
        books_df = pd.read_sql_query("""
            SELECT book_id, title, rating, price_gbp, price_inr, category_id
            FROM books
        """, conn)
        categories_df = pd.read_sql_query("""
            SELECT category_id, category_name
            FROM categories
        """, conn)

        merged_df = pd.merge(
            books_df,
            categories_df,
            on="category_id",
            how="inner"
        )
        merged_df = merged_df[
            ["category_name", "title", "rating", "price_gbp", "price_inr"]
        ].sort_values(
            ["category_name", "rating", "price_gbp"],
            ascending=[True, False, False]
        ).head(10).reset_index(drop=True)

        q6_compare = q6_df.reset_index(drop=True)

    output.append(
        "\nJOIN equivalence check (pd.read_sql vs pd.merge)\n"
        "=================================================\n"
        f"pd.read_sql result:\n{q6_compare.to_string(index=False)}\n\n"
        f"pd.merge result:\n{merged_df.to_string(index=False)}\n\n"
        f"Equivalent: {q6_compare.equals(merged_df)}\n"
    )

    (ROOT / "sql_outputs.txt").write_text("\n".join(output), encoding="utf-8")


def main():
    print("Scraping books.toscrape.com...")
    df = scrape_pages(page_count=5)

    if len(df) < 60:
        raise RuntimeError(f"Only {len(df)} books scraped; at least 60 are required.")

    clean = clean_data(df)
    clean.to_csv(DATA_DIR / "books_clean.csv", index=False)

    db_path = create_database(clean)
    run_queries(db_path)

    print(f"Rows: {len(clean)}")
    print(f"Categories: {clean['category'].nunique()}")
    print(f"Fixed conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR")
    print(f"Database: {db_path}")
    print(f"SQL output: {ROOT / 'sql_outputs.txt'}")


if __name__ == "__main__":
    main()
