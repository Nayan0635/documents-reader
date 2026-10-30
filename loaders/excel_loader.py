import pandas as pd

def load(path: str) -> str:
    sheets = pd.read_excel(path, sheet_name=None)   # all sheets
    parts = []
    for name, df in sheets.items():
        parts.append(f"### Sheet: {name}\n{df.to_csv(index=False)}")
    return "\n\n".join(parts)