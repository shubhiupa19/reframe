import pytest
import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


os.environ["API_KEY"] = "test-key"  # set before app loads                                                                                           
                                                                                                                                                       
from app import app                                                                                                                                  
                  
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:                                                                                                                
        yield client
                                                                                                                                                       
def test_predict_returns_results(client):                                                                                                            
    response = client.post(
        "/predict",                                                                                                                                  
        json={"text": "I always fail at everything I do."}
      )
    
    assert response.status_code == 200
    result = response.get_json()
    assert "results" in result                                                                                                                                                
      # YOUR CODE HERE: assert status code is 200
      # YOUR CODE HERE: assert "results" key is in response JSON                                                                                       
      # Hint: response.get_json() gives you the parsed JSON body
                                                                                                                                                       
def test_predict_missing_text_returns_400(client):
    response = client.post("/predict", json={"text": ""}) 
                                                                                           
    assert response.status_code == 400    

def test_predict_text_too_long_returns_400(client):
    response = client.post("/predict", json={"text": "a" * 5001})   
    result = response.get_json()

    assert response.status_code == 400
    assert result["error"] == "Text is too long, please enter a shorter message"                                                                                                                           
   
def test_feedback_without_api_key_returns_401(client):                                                                                               
    response = client.post(
          "/feedback",                                                                                                                                 
          json={"text": "I always fail", "predicted_distortion": "Overgeneralization", "is_accepted": True, "confidence": 0.8}
      )                                                                                                                                                

    assert response.status_code == 401
                                                                                                                                                       
def test_feedback_with_api_key_returns_feedback_id(client):                                                                                          
    response = client.post(
        "/feedback",                                                                                                                                 
        json={"text": "I always fail", "predicted_distortion": "Overgeneralization", "is_accepted": True, "confidence": 0.8},
        headers={"X-API-Key": "test-key"}                                                                                                            
    )

    result = response.get_json()
    assert isinstance(result["feedback_id"], int)   

def test_feedbacK_text_too_long(client):
    response = client.post("/feedback",  json={"text": "a"*501, "predicted_distortion": "Overgeneralization", "is_accepted": True, "confidence": 0.8},  headers={"X-API-Key": "test-key"}) 
    result = response.get_json()
    assert response.status_code == 400
    assert result["error"] == "Sentence is too long for feedback"

def test_predict_model_error(client):
    with patch('app.classifier') as mock_classifier:
        mock_classifier.side_effect = Exception("Model Crashed")
        response = client.post("/predict", json={"text": "sample sentence"})
        assert response.status_code == 500

def test_feedback_model_error(client):
    with patch('app.save_feedback') as mock_save_feedback:
        mock_save_feedback.side_effect = Exception("Model Crashed")
        response = client.post("/feedback",   json={"text": "I always fail", "predicted_distortion": "Overgeneralization", "is_accepted": True, "confidence": 0.8},
        headers={"X-API-Key": "test-key"}  )
        assert response.status_code == 500