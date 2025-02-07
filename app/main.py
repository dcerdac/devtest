from flask import Flask, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from io import StringIO
import csv
import random
from datetime import datetime, timezone  # Import timezone
import joblib
import numpy as np
from flasgger import Swagger

app = Flask(__name__)

# ---------------------------------------
#  Database Configuration
# ---------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///elevator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
swagger = Swagger(app)  # Enable Swagger UI

# ---------------------------------------
#  Models
# ---------------------------------------
class ElevatorDemand(db.Model):
    __tablename__ = 'elevator_demands'
    
    id = db.Column(db.Integer, primary_key=True)
    floor = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  # Use timezone-aware datetime
    hour = db.Column(db.Integer)

class ElevatorState(db.Model):
    __tablename__ = 'elevator_states'

    id = db.Column(db.Integer, primary_key=True)
    floor = db.Column(db.Integer, nullable=False)
    vacant = db.Column(db.Boolean, nullable=False)
    is_resting = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  # Use timezone-aware datetime

# Ensure database tables exist
with app.app_context():
    db.create_all()

# ---------------------------------------
#  HELPER: Synthetic Data Generator
# ---------------------------------------
def generate_synthetic_records(num_samples=10, max_floor=10):
    """
    Generates a list of synthetic records for demonstration/training.
    Each record might mimic a 'demand' or 'state'.
    """
    synthetic_data = []
    for i in range(num_samples):
        # Randomly decide if this record is a 'demand' or a 'state'
        record_type = random.choice(["demand", "state"])

        # Assign random floor info
        floor = random.randint(1, max_floor)
        timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour
        
        if record_type == "demand":
            synthetic_data.append({
                "type": "demand",
                "id": f"sync_{i}",  # some dummy ID
                "floor": floor,
                "timestamp": timestamp.isoformat(),
                "hour": hour
            })
        else:
            # For 'state', also pick random vacant/resting flags
            vacant = bool(random.getrandbits(1))
            is_resting = bool(random.getrandbits(1))
            synthetic_data.append({
                "type": "state",
                "id": f"sync_{i}",
                "floor": floor,
                "vacant": vacant,
                "is_resting": is_resting,
                "timestamp": timestamp.isoformat()
            })
    return synthetic_data
# ---------------------------------------
#  Load Pre-Trained Model
# ---------------------------------------
MODEL_FILENAME = "elevator_model.pkl"
try:
    model = joblib.load(MODEL_FILENAME)
    print(f" Model loaded from '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"Warning: Could not load model from '{MODEL_FILENAME}'. Error: {e}")
    model = None
# ---------------------------------------
#  Prediction Function
# ---------------------------------------
def predict_next_floor(hour, current_floor):
    """
    Predicts the most likely floor that will call the elevator next.
    Uses the trained model.
    """
    if model is None:
        return {"error": "Model not loaded"}

    # Prepare the input as an array for the model
    input_data = np.array([[hour, current_floor]])  # Format: [[hour, current_floor]]
    
    # Predict probabilities for each floor
    probabilities = model.predict_proba(input_data)[0]  # Get probabilities from classifier

    # Map probabilities to floor numbers (1 to 10)
    floor_predictions = {floor: prob for floor, prob in enumerate(probabilities, start=1)}

    return floor_predictions
# ---------------------------------------
#  Prediction Endpoint
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the next elevator call floor based on current state.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            hour:
              type: integer
              example: 15
            current_floor:
              type: integer
              example: 5
    responses:
      200:
        description: Predicted next floor
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415  # Ensure request is JSON

    data = request.get_json()
    hour = data.get("hour")
    current_floor = data.get("current_floor")

    if hour is None or current_floor is None:
        return jsonify({"error": "Missing parameters"}), 400

    # Call prediction model
    prediction_map = predict_next_floor(hour, current_floor)
    if "error" in prediction_map:
        return jsonify({"error": prediction_map["error"]}), 500

    optimal_floor = max(prediction_map, key=prediction_map.get)

    return jsonify({
        "hour": hour,
        "current_floor": current_floor,
        "predicted_probabilities": prediction_map,
        "optimal_floor": int(optimal_floor)
    }), 200


# ---------------------------------------
#  Historical Data Endpoint
# ---------------------------------------
@app.route("/historical", methods=["GET"])
def get_historical_data():
    """
    Retrieve all historical elevator data (real + synthetic).
    ---
    tags:
      - Historical Data
    responses:
      200:
        description: Returns all historical elevator records
    """
    demands = ElevatorDemand.query.all()
    demand_records = [
        {"type": "demand", "id": d.id, "floor": d.floor, "timestamp": d.timestamp.isoformat(), "hour": d.hour}
        for d in demands
    ]
    
    states = ElevatorState.query.all()
    state_records = [
        {"type": "state", "id": s.id, "floor": s.floor, "vacant": s.vacant, "is_resting": s.is_resting, "timestamp": s.timestamp.isoformat()}
        for s in states
    ]

    synthetic = generate_synthetic_records(num_samples=10, max_floor=10)
    all_data = demand_records + state_records + synthetic

    return jsonify(all_data), 200


# ---------------------------------------
#  Elevator Demand Logging
# ---------------------------------------
@app.route("/demands", methods=["POST"])
def create_demand():
    """
    Create an elevator demand (call).
    ---
    tags:
      - Demands
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            floor:
              type: integer
              example: 3
    responses:
      201:
        description: Demand created successfully
    """
    data = request.get_json()
    if not data or "floor" not in data:
        return jsonify({"error": "Missing 'floor' in request body."}), 400

    new_demand = ElevatorDemand(floor=data["floor"])
    db.session.add(new_demand)
    db.session.commit()

    return jsonify({"message": "Elevator demand recorded.", "id": new_demand.id}), 201


# ---------------------------------------
#  Fetch Demands Endpoint
# ---------------------------------------
@app.route("/demands", methods=["GET"])
def get_demands():
    """
    Retrieve all elevator demands.
    ---
    tags:
      - Demands
    responses:
      200:
        description: Returns a list of elevator demands
    """
    demands = ElevatorDemand.query.all()
    demand_records = [
        {"id": d.id, "floor": d.floor, "timestamp": d.timestamp.isoformat(), "hour": d.hour}
        for d in demands
    ]
    return jsonify(demand_records), 200

# ---------------------------------------
#  Elevator State Logging
# ---------------------------------------
@app.route("/states", methods=["POST"])
def create_state():
    """
    Record the elevator's current state.
    ---
    tags:
      - States
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            floor:
              type: integer
              example: 5
            vacant:
              type: boolean
              example: true
            is_resting:
              type: boolean
              example: false
    responses:
      201:
        description: State recorded successfully
    """
    data = request.get_json()
    if not data or "floor" not in data or "vacant" not in data:
        return jsonify({"error": "Missing required fields (floor, vacant)."}), 400

    new_state = ElevatorState(
        floor=data["floor"],
        vacant=data["vacant"],
        is_resting=data.get("is_resting", False)
    )
    db.session.add(new_state)
    db.session.commit()

    return jsonify({"message": "Elevator state recorded.", "id": new_state.id}), 201


# ---------------------------------------
#  Fetch States Endpoint
# ---------------------------------------
@app.route("/states", methods=["GET"])
def get_states():
    """
    Retrieve all elevator states.
    ---
    tags:
      - States
    responses:
      200:
        description: Returns a list of elevator states
    """
    states = ElevatorState.query.all()
    state_records = [
        {"id": s.id, "floor": s.floor, "vacant": s.vacant, "is_resting": s.is_resting, "timestamp": s.timestamp.isoformat()}
        for s in states
    ]
    return jsonify(state_records), 200

# ---------------------------------------
#  Export CSV Data
# ---------------------------------------
@app.route("/export/csv", methods=["GET"])
def export_csv():
    """
    Export historical + synthetic data as a CSV file.
    ---
    tags:
      - Export
    responses:
      200:
        description: CSV file with elevator data
        content:
          text/csv:
            schema:
              type: string
              format: binary
    """
    demands = ElevatorDemand.query.all()
    demand_records = [
        {"type": "demand", "id": d.id, "floor": d.floor, "timestamp": d.timestamp.isoformat(), "hour": d.hour}
        for d in demands
    ]
    
    states = ElevatorState.query.all()
    state_records = [
        {"type": "state", "id": s.id, "floor": s.floor, "vacant": s.vacant, "is_resting": s.is_resting, "timestamp": s.timestamp.isoformat()}
        for s in states
    ]

    synthetic = generate_synthetic_records(num_samples=10, max_floor=10)
    all_data = demand_records + state_records + synthetic

    def parse_iso(record):
        return datetime.fromisoformat(record["timestamp"])
    all_data.sort(key=parse_iso)

    output = StringIO()
    writer = csv.writer(output)
    
    fieldnames = [
        "type", "id", "floor", "timestamp", "hour",
        "vacant", "is_resting"
    ]
    writer.writerow(fieldnames)
    
    for row in all_data:
        writer.writerow([
            row.get("type", ""),
            row.get("id", ""),
            row.get("floor", ""),
            row.get("timestamp", ""),
            row.get("hour", ""),
            row.get("vacant", ""),
            row.get("is_resting", "")
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=elevator_export.csv"}
    )


#  Run the App
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
