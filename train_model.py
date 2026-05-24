import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
def train_and_save_model():
    print("Generating synthetic historical business data...")
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'competitor_count': np.random.randint(0, 10, n_samples),
        'foot_traffic_index': np.random.uniform(10, 100, n_samples),
        'residential_density': np.random.uniform(1000, 50000, n_samples),
        'transit_stops': np.random.randint(0, 5, n_samples),
        'business_type_encoded': np.random.randint(0, 4, n_samples) 
    })
    success_prob = (
        (data['foot_traffic_index'] / 100 * 0.4) + 
        (data['residential_density'] / 50000 * 0.4) - 
        (data['competitor_count'] / 10 * 0.2) +
        (data['transit_stops'] / 5 * 0.1)
    )
    data['success'] = (success_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    X = data.drop('success', axis=1)
    y = data['success']
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    with open('business_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Success! Model trained and saved as 'business_model.pkl'.")
if __name__ == "__main__":
    train_and_save_model()