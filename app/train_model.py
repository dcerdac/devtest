# train_model.py
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Configuration
NUM_FLOORS = 10      # Number of floors in the building
N_SAMPLES = 10000    # Number of synthetic samples to generate

def generate_synthetic_data(n_samples, num_floors):
    """
    Generate a synthetic dataset of elevator calls.

    Each sample has:
      - hour: Hour of the day (0-23)
      - current_floor: The floor where the elevator is idle before a call
      - next_call_floor: The floor from which the next call is made

    The likelihood of a call from a given floor varies with the hour.
    """
    data = []
    for _ in range(n_samples):
        # Randomly choose an hour of the day and the elevator's current floor
        hour = np.random.randint(0, 24)
        current_floor = np.random.randint(1, num_floors + 1)
        
        # Bias the call probability based on the hour
        if 7 <= hour <= 9:
            # Morning: lower floors (1-4) are more likely
            probs = np.array([0.15 if floor <= 4 else 0.05 for floor in range(1, num_floors + 1)])
        elif 10 <= hour <= 16:
            # Midday: mid-level floors (4-7) are more likely
            probs = np.array([0.05 if floor < 4 or floor > 7 else 0.15 for floor in range(1, num_floors + 1)])
        elif 17 <= hour <= 21:
            # Evening: higher floors (7-10) are more likely
            probs = np.array([0.05 if floor < 7 else 0.15 for floor in range(1, num_floors + 1)])
        else:
            # Off-peak hours: uniform probability
            probs = np.ones(num_floors)
        
        # Normalize probabilities to sum to 1
        probs = probs / probs.sum()
        next_call_floor = np.random.choice(np.arange(1, num_floors + 1), p=probs)
        
        data.append({
            "hour": hour,
            "current_floor": current_floor,
            "next_call_floor": next_call_floor
        })
    return pd.DataFrame(data)

def main():
    # Generate synthetic dataset
    df = generate_synthetic_data(N_SAMPLES, NUM_FLOORS)
    print("Sample of synthetic data:")
    print(df.head())

    # Prepare features and target
    X = df[["hour", "current_floor"]]
    y = df["next_call_floor"]

    # Split into training and testing sets (for evaluation purposes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Optionally, you can evaluate the model here with X_test and y_test

    # Save the trained model to disk
    model_filename = "elevator_model.pkl"
    joblib.dump(clf, model_filename)
    print(f"Model trained and saved to '{model_filename}'")

if __name__ == "__main__":
    main()
