import pandas as pd


def fix_corrupted_csv(filepath="Nifty500_10Yr_Historical.csv"):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)

    original_len = len(df)

    # The artificial flatline started on April 9, 2026.
    # We keep only the rows BEFORE this date.
    clean_df = df[df.index < '2026-04-09']

    new_len = len(clean_df)
    dropped_rows = original_len - new_len

    clean_df.to_csv(filepath)
    print(f"Success! Dropped {dropped_rows} corrupted rows.")
    print(f"CSV is restored to its real end date: {clean_df.index[-1].date()}")


if __name__ == "__main__":
    fix_corrupted_csv()