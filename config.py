import os


# Base directory for model storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Paths for storing models, encoders, scalers, etc.
MODELS_DIR = os.path.join(BASE_DIR, "models")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
OPERATORS_PATH = os.path.join(MODELS_DIR, "operator_codes.joblib")

# Model paths
LOGISTIC_MODEL_PATH = os.path.join(MODELS_DIR, "logisticregression_model.pkl")
DECISION_TREE_MODEL_PATH = os.path.join(MODELS_DIR, "decisiontree_model.pkl")
RANDOM_FOREST_MODEL_PATH = os.path.join(MODELS_DIR, "randomforest_model.pkl")

# Data path
DATA_PATH = os.path.join(BASE_DIR,"personal_data.csv")





