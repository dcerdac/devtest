# app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__)
# Configure your database URI; for this example, we're using a SQLite database.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///elevator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------------------
# Database Models
# ---------------------------
class ElevatorDemand(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    floor = db.Column(db.Integer, nullable=False)
    hour = db.Column(db.Integer)  # Hour of day for time-based analysis

    def __init__(self, floor):
        self.floor = floor
        self.hour = datetime.utcnow().hour

class ElevatorState(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    floor = db.Column(db.Integer, nullable=False)
    vacant = db.Column(db.Boolean, nullable=False)
    is_resting = db.Column(db.Boolean, default=False)

# Uncomment the following lines on the first run to create the database tables.
# with app.app_context():
#     db.create_all()

# ---------------------------
# Load the Trained ML Model
# ---------------------------
MODEL_FILENAME = "elevator_model.pkl"
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"Model loaded from '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_next_call(hour, current_floor):
    """
    Given the current hour and current floor, predict the probability
    for each floor being the next call floor using the trained model.
    """
    # Reshape input to a 2D array as expected by scikit-learn
    features = np.array([[hour, current_floor]])
    probabilities = model.predict_proba(features)[0]
    floor_labels = model.classes_
    # Map each floor (label) to its predicted probability
    prediction = dict(zip(floor_labels, probabilities))
    return prediction

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict_endpoint():
    """
    Prediction Endpoint.
    
    You can supply either query parameters (GET) or a JSON payload (POST) with:
      - hour: current hour (0-23)
      - current_floor: the floor where the elevator is idle
      
    If these parameters are missing, the endpoint will try to use the latest vacant
    ElevatorState record.
    """
    if request.method == "POST":
        data = request.get_json()
        hour = data.get("hour")
        current_floor = data.get("current_floor")
    else:
        hour = request.args.get("hour", type=int)
        current_floor = request.args.get("current_floor", type=int)

    # Fallback: If parameters are missing, use the latest vacant ElevatorState
    if hour is None or current_floor is None:
        latest_state = ElevatorState.query.filter_by(vacant=True).order_by(ElevatorState.timestamp.desc()).first()
        if latest_state:
            current_floor = latest_state.floor
            hour = latest_state.timestamp.hour
        else:
            return jsonify({"error": "Missing parameters and no idle ElevatorState found."}), 400

    # Generate predictions based on the provided or retrieved values
    predictions = predict_next_call(hour, current_floor)
    optimal_floor = max(predictions, key=predictions.get)
    
    result = {
        "hour": hour,
        "current_floor": current_floor,
        "predicted_probabilities": predictions,
        "optimal_floor": int(optimal_floor)
    }
    return jsonify(result)

# A simple endpoint to add a new ElevatorState (for testing/updating state)
@app.route("/state", methods=["POST"])
def add_state():
    """
    Add a new ElevatorState record.
    
    Expected JSON payload:
    {
        "floor": <int>,
        "vacant": <bool>,
        "is_resting": <bool>  // Optional
    }
    """
    data = request.get_json()
    floor = data.get("floor")
    vacant = data.get("vacant")
    is_resting = data.get("is_resting", False)
    
    if floor is None or vacant is None:
        return jsonify({"error": "Missing required parameters: 'floor' and 'vacant'."}), 400

    new_state = ElevatorState(floor=floor, vacant=vacant, is_resting=is_resting)
    db.session.add(new_state)
    db.session.commit()
    
    return jsonify({"message": "ElevatorState record added."}), 201

if __name__ == "__main__":
    app.run(debug=True)
