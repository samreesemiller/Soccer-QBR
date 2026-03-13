import pandas as pd

def load_data(filepath: str):
  passes = pd.read_excel(filepath, sheet_name='Data Entry')
  combinations = pd.read_excel(filepath, sheet_name='Combinations')
  return passes, combinations

def clean_passes(df: pd.DataFrame) -> pd.DataFrame:
  df = df.dropna(how="all").drop_duplicates()
  df['success'] = df['Completed'].map({1: 1, 0: 0, 'Y': 1, 'N': 0})
  return df.reset_index(drop=True)

def load_and_prepare(filepath: str):
  passes, combinations = load_data(filepath)
  passes = clean_passes(passes)
  return passes, combinations

