import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocessing import preprocess_newdata,preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config
import joblib
from fastapi import HTTPException
import logging



# Function to train the model
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model_path = os.path.join(config.MODELS_DIR, f"{name.replace(" ","").strip().lower()}_model.pkl")
        trained_model = train_and_save_model(model, X_train, y_train, model_path)
        results[name] = evaluate_model(trained_model, X_test, y_test)
    return results

def train_and_save_model(model, X_train, y_train, model_path):
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save the model to dir-models
    joblib.dump(model, model_path)
    return model

# Function to find the accuracy of the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results={
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    return results

# Function to predict new-input-data
def predict_new(model_name, input_data):
    # Fetching the selected model path from central-configuration module
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    # Load the model from joblib
    model = joblib.load(model_path)
    # Predict with selected module for input-data
    result = model.predict(input_data)
    return {"value" : int(result[0])} # Return the prediction results (format like {'value':0/1})



     
     