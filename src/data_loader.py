"""Data loading and column renaming utilities for the credit risk dataset."""

import pandas as pd


RENAME_MAP = {
    "LIMIT_BAL": "limite_credito",
    "SEX": "sexo",
    "EDUCATION": "escolaridade",
    "MARRIAGE": "estado_civil",
    "AGE": "idade",
    "PAY_0": "status_pagamento_1",
    "PAY_2": "status_pagamento_2",
    "PAY_3": "status_pagamento_3",
    "PAY_4": "status_pagamento_4",
    "PAY_5": "status_pagamento_5",
    "PAY_6": "status_pagamento_6",
    "BILL_AMT1": "valor_fatura_1",
    "BILL_AMT2": "valor_fatura_2",
    "BILL_AMT3": "valor_fatura_3",
    "BILL_AMT4": "valor_fatura_4",
    "BILL_AMT5": "valor_fatura_5",
    "BILL_AMT6": "valor_fatura_6",
    "PAY_AMT1": "valor_pagamento_1",
    "PAY_AMT2": "valor_pagamento_2",
    "PAY_AMT3": "valor_pagamento_3",
    "PAY_AMT4": "valor_pagamento_4",
    "PAY_AMT5": "valor_pagamento_5",
    "PAY_AMT6": "valor_pagamento_6",
    "default payment next month": "inadimplente_proximo_mes",
}

EXPECTED_COLUMNS = [
    "limite_credito",
    "sexo",
    "escolaridade",
    "estado_civil",
    "idade",
    "status_pagamento_1",
    "status_pagamento_2",
    "status_pagamento_3",
    "status_pagamento_4",
    "status_pagamento_5",
    "status_pagamento_6",
    "valor_fatura_1",
    "valor_fatura_2",
    "valor_fatura_3",
    "valor_fatura_4",
    "valor_fatura_5",
    "valor_fatura_6",
    "valor_pagamento_1",
    "valor_pagamento_2",
    "valor_pagamento_3",
    "valor_pagamento_4",
    "valor_pagamento_5",
    "valor_pagamento_6",
    "inadimplente_proximo_mes",
]


def load_dataset(source, header=1):
    """Load the credit card default dataset from a file path or URL.

    Parameters
    ----------
    source : str
        Path or URL to the XLS/XLSX/CSV file.
    header : int, default 1
        Row number to use as the column names (0-indexed).

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns and ID column dropped.
    """
    if str(source).endswith(".csv"):
        df = pd.read_csv(source, header=header)
    else:
        df = pd.read_excel(source, header=header)

    df = rename_columns(df)

    if "ID" in df.columns:
        df = df.drop(columns="ID")

    return df


def rename_columns(df):
    """Rename dataset columns from English to Portuguese standard names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed per RENAME_MAP.
    """
    return df.rename(columns=RENAME_MAP)


def validate_columns(df):
    """Check that the DataFrame has all expected columns after renaming.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of missing columns)
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing
