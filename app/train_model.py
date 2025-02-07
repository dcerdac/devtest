import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

API_BASE_URL = "http://127.0.0.1:5000"

def fetch_historical_data():
    url = f"{API_BASE_URL}/historical"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def prepare_dataframe(data):
    df = pd.DataFrame(data)
    
    if "vacant" not in df:
        df["vacant"] = 0
    if "is_resting" not in df:
        df["is_resting"] = 0
    if "hour" not in df:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    
    df["vacant"] = df["vacant"].astype(int)
    df["is_resting"] = df["is_resting"].astype(int)
    
    df = df.dropna(subset=["floor"])
    
    X = df[["hour", "floor", "vacant", "is_resting"]]
    y = df["floor"].shift(-1).fillna(df["floor"]).astype(int)
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    model_filename = "elevator_model2.pkl"
    joblib.dump(clf, model_filename)
    
    print(f"Model saved to {model_filename}")
    print(f"Training Accuracy: {clf.score(X_train, y_train):.2f}")
    print(f"Test Accuracy: {clf.score(X_test, y_test):.2f}")

def main():
    data = fetch_historical_data()
    if data:
        X, y = prepare_dataframe(data)
        train_model(X, y)
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()