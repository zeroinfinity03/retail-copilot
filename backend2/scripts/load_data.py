"""
Load H&M CSVs into a DuckDB database.

Expects 3 files in backend/raw_data/:
  - articles.csv
  - customers.csv
  - transactions_train.csv

Creates: backend/data/db/hm.duckdb with 3 tables (articles, customers, transactions).
"""

from pathlib import Path
import duckdb

BACKEND_DIR = Path(__file__).parent.parent      # backend/
CSV_DIR = BACKEND_DIR / "raw_data"
DB_PATH = BACKEND_DIR / "data" / "db" / "hm.duckdb"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = {
    "articles": CSV_DIR / "articles.csv",
    "customers": CSV_DIR / "customers.csv",
    "transactions": CSV_DIR / "transactions_train.csv",
}


def check_files() -> None:
    missing = [name for name, path in EXPECTED_FILES.items() if not path.exists()]
    if missing:
        print("❌ Missing files in backend/raw_data/:")
        for name in missing:
            print(f"   - {EXPECTED_FILES[name].name}")
        print("\nPlease download from Kaggle and place them here:")
        print(f"   {CSV_DIR}")
        raise SystemExit(1)

    print("✅ All 3 CSV files found:")
    for name, path in EXPECTED_FILES.items():
        size_mb = path.stat().st_size / 1e6
        print(f"   {path.name:30s} {size_mb:>10.1f} MB")


def load_csv_to_table(con: duckdb.DuckDBPyConnection, table: str, csv_path: Path) -> None:
    """Drop and recreate the table from CSV using DuckDB's native CSV reader."""
    print(f"\n📥 Loading {csv_path.name} → table '{table}' ...")
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
        [str(csv_path)],
    )
    row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"   {row_count:,} rows loaded")


def print_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    print(f"\n📋 Schema for '{table}':")
    columns = con.execute(f"DESCRIBE {table}").fetchall()
    for col_name, col_type, *_ in columns:
        print(f"   {col_name:35s} {col_type}")


def print_samples(con: duckdb.DuckDBPyConnection, table: str, n: int = 3) -> None:
    print(f"\n🔍 Sample rows from '{table}':")
    rows = con.execute(f"SELECT * FROM {table} LIMIT {n}").fetchdf()
    print(rows.to_string(index=False, max_colwidth=40))


def quick_sanity_checks(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 60)
    print("Sanity checks")
    print("=" * 60)

    txn_range = con.execute(
        "SELECT MIN(t_dat) AS min_date, MAX(t_dat) AS max_date FROM transactions"
    ).fetchone()
    print(f"📅 Transaction date range: {txn_range[0]}  →  {txn_range[1]}")

    counts = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM customers)    AS customer_count,
            (SELECT COUNT(*) FROM articles)     AS article_count,
            (SELECT COUNT(*) FROM transactions) AS transaction_count
        """
    ).fetchone()
    print(f"👥 Customers:    {counts[0]:>12,}")
    print(f"🏷  Articles:     {counts[1]:>12,}")
    print(f"💳 Transactions: {counts[2]:>12,}")

    channel_mix = con.execute(
        """
        SELECT sales_channel_id, COUNT(*) AS txns
        FROM transactions
        GROUP BY sales_channel_id
        ORDER BY sales_channel_id
        """
    ).fetchall()
    print("\n🛒 Channel mix (1 = in-store, 2 = online):")
    for channel, txns in channel_mix:
        print(f"   channel {channel}: {txns:>12,} transactions")

    top_groups = con.execute(
        """
        SELECT a.index_name, COUNT(*) AS txns
        FROM transactions t
        JOIN articles a USING (article_id)
        GROUP BY a.index_name
        ORDER BY txns DESC
        """
    ).fetchall()
    print("\n👗 Transactions by index_name (top categories):")
    for index_name, txns in top_groups:
        print(f"   {str(index_name):30s} {txns:>12,}")


def main() -> None:
    print(f"📂 CSVs from:  {CSV_DIR}")
    print(f"💾 DB path:    {DB_PATH}\n")

    check_files()

    con = duckdb.connect(str(DB_PATH))

    for table, csv_path in EXPECTED_FILES.items():
        load_csv_to_table(con, table, csv_path)

    for table in EXPECTED_FILES:
        print_schema(con, table)
        print_samples(con, table)

    quick_sanity_checks(con)

    con.close()
    print(f"\n✅ Done. DuckDB written to: {DB_PATH}")
    print(f"   Size: {DB_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
