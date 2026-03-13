from src.data_loader import load_and_prepare
from src.features import prepare_features, prepare_combinations
from src.model import train_model, predict_combinations
from src.visualize import plot_all
import pandas as pd

filepath = 'data/PassModel.xlsx'

passes, combinations = load_and_prepare(filepath)
X, y = prepare_features(passes)
model = train_model(X, y)

X_combo = prepare_combinations(combinations)
combinations['predicted_success_prob'] = predict_combinations(model, X_combo)

combinations.to_excel("data/PassModel_predictions.xlsx", index=False)
print("Predictions saved")

