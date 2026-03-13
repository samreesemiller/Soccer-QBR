import pandas as pd

ENCODINGS = {
  'Direction': {'F': 2, 'N': 1, 'B': 0},
  'Area': {'A': 2, 'M': 1, 'D': 0},
  'Body Pt': {'F': 0, 'H': 1, 'B': 0},
  'Foot': {'S': 1, 'W': 0},
  'Thru': {'Y': 1,'N': 0},
  'Air': {'Y': 1,'N': 0},
  'Cross': {'Y': 1,'N': 0}
}

FEATURE_COLS = ['Distance', 'Direction', 'Area', 'Min. Def.', 'Body Pt', 'Foot', 'Touches', 'Thru', 'Air', 'Cross']

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  for col, mapping in ENCODINGS.items():
    if col in df.columns:
      df[col] = df[col].map(mapping)
  return df

def prepare_features(df: pd.DataFrame, target_col: str='success'):
  df = encode_features(df)
  available = [col for col in FEATURE_COLS if col in df.columns]
  X = df[available]
  y = df[target_col] if target_col in df.columns else None
  return X, y

def prepare_combinations(df: pd.DataFrame):
  """Encode the combinations sheet for prediction"""
  df = encode_features(df)
  available = [col for col in FEATURE_COLS if col in df.columns]
  return df[available]