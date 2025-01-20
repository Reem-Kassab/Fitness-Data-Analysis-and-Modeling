import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data, save_preprocessing_artifacts

def train_GYM_model(dataset_path, model_path, scaler_path, feature_names_path):
    # Load and preprocess data
    X_scaled, y, scaler, feature_names = load_and_preprocess_data(dataset_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(rf_model, model_path)
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(
        scaler, feature_names, 
        scaler_path, feature_names_path
    )
    
    # Generate feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances for Calories burne Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Return performance metrics
    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }

# If script is run directly, train the model
if __name__ == '__main__':
    metrics = train_GYM_model(
        dataset_path='D:/Data Science GDG/GYM project/Dataset/gym_members_exercise_tracking.csv',
        model_path='trained_model.pkl',
        scaler_path='scaler.pkl',
        feature_names_path='feature_names.txt'
    )
    print("Model Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")