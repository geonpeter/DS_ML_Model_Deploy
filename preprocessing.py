import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ENCODER_PATH, SCALER_PATH, OPERATORS_PATH
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# Create instances of LabelEncoder and StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

# Function of pre-processing the data (Employer Data)
def preprocess_data(path):

    
    # Load data
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}!!")

    # Extract operator codes for validation
    operator_codes = df['OperatorCode'].unique().tolist()
    joblib.dump(operator_codes, OPERATORS_PATH)

    
    # Process date and extract features
    df['DeclarationDate'] = pd.to_datetime(df['DeclarationDate'])
    df['hour'] = df['DeclarationDate'].dt.hour
    df['day'] = df['DeclarationDate'].dt.day
    df['weekday'] = df['DeclarationDate'].dt.weekday
    df['month'] = df['DeclarationDate'].dt.month
    df['operator_code'] = encoder.fit_transform(df['OperatorCode'])
    df['presence_state'] = df['IdPresenceState'].apply(lambda x: 1 if x == 1 else 0)

    # Drop unused columns
    df_cleaned = df.drop(columns=['IdPresenceDeclaration', 'OperatorCode', 'DeclarationDate', 'IdPresenceState'])

    # Split data
    X = df_cleaned[['hour', 'day', 'weekday', 'month', 'operator_code']]
    y = df_cleaned['presence_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Save encoder and scaler
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
 
    
    # Return the transformed data back to train the model
    return X_train, X_test, y_train, y_test


# Function to pre-process the input-data
def preprocess_newdata(op_code, date, time):
    

    
    # Extract features from input-data    
    df_new = {
        'hour': [time.hour],
        'day': [date.day],
        'weekday': [date.weekday()],
        'month': [date.month],
        'operator_code': [op_code]
    }
    
    # Encode the operator_code
    encoder = joblib.load(ENCODER_PATH)
    df_new['operator_code'] = encoder.transform(df_new['operator_code'])
    
 
    # Standardize the new data
    df_new = pd.DataFrame(df_new)
    scaler = joblib.load(SCALER_PATH)
    df_new = scaler.transform(df_new)
    
 
    
    # Return the tranformed data (numpy array) to predict 
    return df_new
