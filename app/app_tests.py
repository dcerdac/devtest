import pytest
from main import app, db
import json

@pytest.fixture
def client():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
        db.drop_all()

def test_create_demand_success(client):
    response = client.post('/api/demand', 
                         json={'floor': 3},
                         content_type='application/json')
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'id' in data
    assert data['floor'] == 3

def test_create_demand_invalid_floor(client):
    response = client.post('/api/demand', 
                         json={'floor': 25},
                         content_type='application/json')
    assert response.status_code == 400

def test_create_state_success(client):
    response = client.post('/api/state', 
                         json={'floor': 5, 'vacant': True, 'is_resting': True},
                         content_type='application/json')
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'id' in data
    assert data['floor'] == 5

def test_get_demand_analytics(client):
    # Create some test demands
    client.post('/api/demand', json={'floor': 3})
    client.post('/api/demand', json={'floor': 3})
    client.post('/api/demand', json={'floor': 5})
    
    response = client.get('/api/analytics/demand')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
