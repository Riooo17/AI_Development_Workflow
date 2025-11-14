# Example stub for deployment
# In a real scenario, this could be a REST API using Flask or FastAPI

def predict_dropout(student_features, trained_model):
    """
    Returns dropout prediction for a single student input
    """
    prediction = trained_model.predict([student_features])
    probability = trained_model.predict_proba([student_features])[0,1]
    return prediction[0], probability
