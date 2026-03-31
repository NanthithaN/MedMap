import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

MODEL_PATH = "xgboost_model.pkl"
DATASET_PATH = "../dataset/medmap_ds.csv"

FEATURES = [
    'population',
    'nearby_hospital_count',
    'avg_distance_to_hospital_km',
    'day_1', 'day_2', 'day_3',
    'week_1', 'week_2', 'week_3',
    'month_1'
]

TARGET = 'month_2'

class DemandPredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.is_trained = False
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.is_trained = True
                print("Loaded pre-trained model.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def train(self):
        """Trains the model on the entire dataset."""
        print(f"Loading dataset from {DATASET_PATH}...")
        try:
            df = pd.read_csv(DATASET_PATH)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load dataset: {e}"}

        # Check if dataset has necessary columns
        missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
        if missing_cols:
            return {"status": "error", "message": f"Missing columns in dataset: {missing_cols}"}

        X = df[FEATURES]
        y = df[TARGET]

        print("Training model...")
        self.model.fit(X, y)
        self.is_trained = True

        # Save model
        joblib.dump(self.model, MODEL_PATH)

        # Calculate a basic training error for logging
        preds = self.model.predict(X)
        mse = mean_squared_error(y, preds)
        
        return {"status": "success", "message": "Model trained globally", "mse": mse}

    def predict(self, selected_areas: list[str]):
        """Predicts demand ONLY for the selected areas."""
        if not self.is_trained:
            return {"status": "error", "message": "Model is not trained yet."}

        try:
            df = pd.read_csv(DATASET_PATH)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load dataset during prediction: {e}"}

        filtered_df = df[df['area_name'].isin(selected_areas)].copy()
        
        if filtered_df.empty:
            return {"status": "error", "message": "None of the selected areas were found in the dataset."}

        X_predict = filtered_df[FEATURES]
        
        # Predict
        predictions = self.model.predict(X_predict)
        
        # Ensure non-negative demand and round to integer
        predictions = [max(0, int(round(p))) for p in predictions]
        
        # Return mapping of area_name -> predicted_demand
        results = dict(zip(filtered_df['area_name'], predictions))
        
        # Add basic info mapping
        area_info = {}
        for _, row in filtered_df.iterrows():
            area_info[row['area_name']] = {
                "population": row['population'],
                "demand": results[row['area_name']]
            }

        return {"status": "success", "predictions": area_info}

# Singleton instance
predictor = DemandPredictor()
