# server/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import os

# --- Configuration ---
# Assuming parcels_10000.csv is in the same directory as this script
CSV_FILE = 'parcels_10000.csv'
MODEL_FILENAME = 'routing_model.pkl'

# Your CSV uses 0, 1 for parcel_type and 0, 1, 2 for route_direction, so no mapping needed here
# We only select the required numerical feature columns.
FEATURE_COLS = ['source_city_id', 'destination_city_id', 'parcel_type']
TARGET_COL = 'route_direction'

def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepares the data for training assuming pre-encoded numerical values."""
    required_cols = FEATURE_COLS + [TARGET_COL]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV is missing one of the required columns: {required_cols}")

    # Ensure data types are appropriate and handle missing values if any (simple drop for prototype)
    df = df.dropna(subset=required_cols)
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    
    if len(X) == 0:
        raise ValueError("Dataset is empty after dropping missing values.")

    return X, y

def train_and_save_model(csv_path: str, model_path: str):
    """Loads data, trains the model, and saves it using joblib."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        X, y = prepare_data(df)
    except ValueError as e:
        print(f"Error during data preparation: {e}")
        return

    print(f"Dataset size: {len(X)} samples.")
    
    # Split data (use all data for training if not enough to split, simpler for retraining)
    if len(X) < 10:
        print("Warning: Dataset too small for splitting. Using all data for training.")
        X_train, y_train = X, y
        X_test, y_test = X, y # Use training data for a basic test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a simple Multi-layer Perceptron (NN) Classifier
    model = MLPClassifier(hidden_layer_sizes=(50,), 
                          max_iter=500, 
                          activation='relu', 
                          solver='adam', 
                          random_state=42,
                          # Suppress warnings about convergence for prototype brevity
                          # You should enable this for tuning: verbose=True)
                          ) 

    print("--- Training model... ---")
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"--- Training complete. Test Accuracy: {accuracy:.4f} ---")

    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, CSV_FILE)
    model_file_path = os.path.join(current_dir, MODEL_FILENAME)
    
    train_and_save_model(csv_file_path, model_file_path)