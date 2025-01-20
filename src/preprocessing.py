import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select relevant features (modify as needed)
    features = [
        'Age', 'Gender', 'Weight (kg)', 'Height (m)', 
        'Max_BPM', 'Avg_BPM', 'Resting_BPM', 
        'Session_Duration (hours)', 'Workout_Type', 
        'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
        'Experience_Level','BMI'
    ]

    # Separate features and target
    X = df[features]
    y = df['Calories_Burned']
    
    # Handle categorical columns
    categorical_cols = [
       'Gender',
        'Workout_Type',
    ]
    
    # label encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

def save_preprocessing_artifacts(scaler, feature_names, scaler_path, feature_names_path):
    import joblib
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    # Save feature names
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))