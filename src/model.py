from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def train_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
  model = XGBClassifier(n_estimator=100, max_depth=4,learning_rate=0.1, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]
  print(classification_report(y_test, y_pred))
  print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
  return model

def predict_combinations(model, X_combos):
  # Returns predicted pass success probability for all 25,920 combos
  probs = model.predict_proba(X_combos)[:, 1]
  return probs