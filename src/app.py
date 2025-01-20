import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="Calories Burned Predictor", 
    page_icon="💪", 
    layout="wide"
)

# Load preprocessing artifacts
@st.cache_resource
def load_model_and_artifacts():
    """Load saved model, scaler, and feature names"""
    try:
        model = joblib.load('D:/Data Science GDG/Calorie Burn Predictor/src/trained_model.pkl')
        scaler = joblib.load('D:/Data Science GDG/Calorie Burn Predictor/src/scaler.pkl')
        with open('D:/Data Science GDG/Calorie Burn Predictor/src/feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Main app
def main():
    st.title("🚀 Calories Burned Prediction 🔥")
    
    # Load model
    model, scaler, feature_names = load_model_and_artifacts()
    if model is None or scaler is None or feature_names is None:
        st.error("Could not load the model or required files. Please check your model files.")
        return
    
    # Sidebar for input features
    st.sidebar.header("🔥 Track Your Fitness Journey")
    st.write(
        "💪 Ready to track your workout progress? Add your activity details to start predicting the calories you’ve burned and stay on top of your fitness goals! 🔥"
    )
    
    # Collect numerical inputs with error handling for input ranges
    try:
        Session_Duration = st.sidebar.number_input(
            "Session Duration (hours)", 
            min_value=1, 
            max_value=10, 
            value=5,
            step=1
        )
        st.write(f'Your Session Duration is: {Session_Duration} hours')

        Avg_BPM = st.sidebar.slider(
            "Select your Average BPM (Heart Rate):",
            min_value=0,
            max_value=250,
            value=70,
            step=1
        )
        st.write(f'Your Average BPM: {Avg_BPM}')

        Age = st.sidebar.number_input(
            "Enter your Age:", 
            min_value=18,    
            max_value=100,   
            value=25,
            step=1         
        )
        st.write(f'Your Age: {Age}')

        Fat_Percentage = st.sidebar.slider(
            "Select your Fat Percentage (%):", 
            min_value=0.0,     
            max_value=100.0,   
            value=20.0,        
            step=0.1           
        )
        st.write(f'Your Fat Percentage: {Fat_Percentage}%')

        Height = st.sidebar.number_input(
            "Enter your Height (m):", 
            min_value=0.5,  
            max_value=2.5,  
            value=1.75,     
            step=0.01
        )
        st.write(f'Your Height: {Height} m')

        Weight = st.sidebar.number_input(
            "Enter your Weight (kg):", 
            min_value=30.0,  
            max_value=200.0, 
            value=70.0,      
            step=0.1
        )
        st.write(f'Your Weight: {Weight} kg')

        # Categorical inputs
        gender = st.sidebar.selectbox(
            "Select your Gender:", 
            options=["Male", "Female"]
        )
        st.write(f'Your Gender: {gender}')

        workout_type = st.sidebar.selectbox(
            "Select your Workout Type:", 
            options=["Yoga", "HIIT", "Cardio", "Strength"]
        )
        st.write(f'Your Workout Type: {workout_type}')

    except Exception as e:
        st.error(f"Error in input fields: {e}")
        return
    
    # Prediction button
    if st.sidebar.button("Predict calories burned"):
        try:
            # Prepare input data
            input_data = {
                'Session_Duration (hours)': Session_Duration,
                'Avg_BPM': Avg_BPM,
                'Age': Age,
                'Fat_Percentage': Fat_Percentage,
                'Height (m)': Height,  
                'Weight (kg)': Weight,
                'Gender': gender,
                'Workout_Type': workout_type
            }
            
            # Convert to dataframe
            input_df = pd.DataFrame([input_data])

            # Encode categorical columns
            categorical_cols = ['Gender', 'Workout_Type']
            input_df_encoded = input_df.copy()
            label_encoder = LabelEncoder()
            for column in categorical_cols:
                input_df_encoded[column] = label_encoder.fit_transform(input_df[column])

            # Ensure all original features are present
            for col in feature_names:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0
            
            # Reorder columns to match training data
            input_df_encoded = input_df_encoded[feature_names]
            
            # Scale input
            input_scaled = scaler.transform(input_df_encoded)
            
            # Predict
            predicted_price = model.predict(input_scaled)[0]
            
            # Display results
            st.subheader("🔥 Your Workout Results Are In! See How Many Calories You Burned 💪")
            st.metric(
                label="Estimated Calories Burned",  
                value=f"{predicted_price:.2f} kcal"  
            )
            
            # Feature importance visualization
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)

            # Radar chart
            categories = feature_importance['feature']
            values = feature_importance['importance']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

            fig, ax = plt.subplots(figsize=(8, 8), dpi=80, subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='skyblue', alpha=0.25)
            ax.plot(angles, values, color='blue', linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=12, color='#34495E')
            plt.title("Feature Importance Radar Chart", fontsize=16, fontweight='bold', color='#2C3E50')
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        # About section
        st.sidebar.markdown("## About")
        st.sidebar.info(
            "This app predicts the calories burned based on various factors like heart rate, session duration, age, and more. "
            "Enter your details in the sidebar and click 'Predict' to estimate the calories burned."
        )

if __name__ == '__main__':
    main()
