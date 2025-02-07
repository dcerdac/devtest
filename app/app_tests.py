import pytest
import json
import csv
from io import StringIO
from main import app, db  

@pytest.fixture
def client():
    """Set up a test client with a test database."""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    app.config['TESTING'] = True

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client
        with app.app_context():
            db.drop_all()

def test_create_demand_success(client):
    """Test POST /demands with valid input."""
    response = client.post('/demands', json={'floor': 3})
    assert response.status_code == 201
    data = response.get_json()
    assert 'id' in data
    assert data['message'] == "Elevator demand recorded."

def test_create_state_success(client):
    """Test POST /states with valid input."""
    response = client.post('/states', json={'floor': 5, 'vacant': True, 'is_resting': True})
    assert response.status_code == 201
    data = response.get_json()
    assert 'id' in data
    assert data['message'] == "Elevator state recorded."

def test_get_demands(client):
    """Test GET /demands returns the created demands."""
    client.post('/demands', json={'floor': 3})
    client.post('/demands', json={'floor': 5})

    response = client.get('/demands')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 2

def test_get_states(client):
    """Test GET /states returns the created states."""
    client.post('/states', json={'floor': 2, 'vacant': True, 'is_resting': False})
    client.post('/states', json={'floor': 4, 'vacant': False, 'is_resting': True})

    response = client.get('/states')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 2

def test_export_csv(client):
    """Test GET /export/csv returns a valid CSV file."""
    client.post('/demands', json={'floor': 3})
    client.post('/states', json={'floor': 5, 'vacant': True, 'is_resting': True})

    response = client.get('/export/csv')
    assert response.status_code == 200
    assert response.content_type == "text/csv"

    csv_data = response.data.decode('utf-8').splitlines()
    reader = csv.reader(csv_data)
    rows = list(reader)

    assert len(rows) > 1
    assert rows[0] == ["type", "id", "floor", "timestamp", "hour", "vacant", "is_resting"]
    assert any(row[0] == 'demand' for row in rows[1:])
    assert any(row[0] == 'state' for row in rows[1:])
