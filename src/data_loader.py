"""
data_loader.py
===============
STEP 1 of the pipeline: Import and Load the Data.

Why this step matters:
    Every downstream step (EDA, cleaning, modelling) depends on the data
    being loaded consistently and correctly typed. A dedicated loader
    function means the notebook, the training script, and the Streamlit app
    all read the data the exact same way -- so there is no risk of the app
    silently using a differently-cleaned copy of the file than the one the
    model was trained on.
"""

import pandas as pd

from src.config import RAW_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Dry Bean dataset from the Excel file.

    Why an Excel reader and not read_csv:
        The source file supplied for this project is an .xlsx workbook, so
        we use pandas' Excel engine (openpyxl) directly. Keeping the reading
        logic in one function means if the data source ever moves to CSV or
        a database, only this function needs to change.
    """
    df = pd.read_excel(path)
    return df


def basic_overview(df: pd.DataFrame) -> None:
    """
    Print the standard first-look trio: head, info, describe.

    Why this matters:
        Before any cleaning or modelling, a data scientist must confirm:
        (1) the data loaded with the right shape and columns (.head()),
        (2) the data types and null counts are as expected (.info()),
        (3) the numeric ranges look sane, e.g. no negative areas (.describe()).
        Skipping this step is how silent data-quality bugs slip into a model.
    """
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 80)

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- INFO ---")
    df.info()

    print("\n--- DESCRIBE (numeric columns) ---")
    print(df.describe().T)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Confirm the expected feature columns and target column are present.

    Why this matters:
        If a future export of this dataset renames or drops a column, this
        check fails loudly and immediately instead of letting a KeyError
        surface confusingly deep inside a scikit-learn pipeline later.
    """
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    print("Schema check passed: all expected feature and target columns are present.")


if __name__ == "__main__":
    data = load_raw_data()
    validate_schema(data)
    basic_overview(data)
