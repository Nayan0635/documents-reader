import pandas as pd

def load(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_csv(index=False)   # keep as text; or use df.to_markdown()