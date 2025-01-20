# Project Overview

This project involves analyzing a fitness dataset to gain insights into users' workout behaviors and health metrics. The dataset includes attributes such as age, gender, BMI, session duration, workout type, calories burned, and maximum heart rate (Max_BPM). The primary objectives are to perform exploratory data analysis (EDA), preprocess the data, remove outliers, and build predictive models to understand relationships and predict outcomes effectively.

# Steps Taken

## 1. Exploratory Data Analysis (EDA)

Visualized the relationships between key variables such as:

- Age vs. Calories Burned

- Workout Type vs. Gender

- Session Duration vs. Max_BPM

- Created pair plots, heatmaps, box plots, and line charts to identify trends, correlations, and anomalies.

Key Insights:

- Older individuals tend to burn fewer calories.

- Men favor strength training, while women prefer yoga and cardio.

- Overweight users often perform high-intensity workouts less frequently.

## 2. Data Preprocessing

Handled missing values by:

- Filling numerical columns with the mean/median.

- Filling categorical columns with the mode.

- Encoded categorical variables:

- Used Label Encoding and One-Hot Encoding for Gender and Workout_Type.

Standardized the dataset:

- Applied MinMaxScaler to BMI, Calories_Burned, and Weight (kg).

## 3. Outlier Removal

Identified outliers using the IQR method for numerical columns:

- BMI

- Calories_Burned

- Weight (kg)

Removed rows with values outside the acceptable range (1.5 × IQR).

## 4. Modeling

Built and trained models to predict key outcomes:

- Regression model for Calories_Burned prediction.

Evaluated model performance using:

- Mean Squared Error (MSE): 0.0059

- R-squared (R²): 99.44%

For classification, accuracy was lower (~28%), suggesting the need for feature engineering and hyperparameter tuning.

# Tools and Libraries

## Python Libraries:

- Pandas, NumPy for data manipulation.

- Matplotlib, Seaborn for visualization.

- Scikit-learn for machine learning.

- Streamlit 

## Challenges Faced

- Handling class imbalance in Workout_Type.

- Low classification accuracy for Workout_Type due to limited features.

- Need for deeper feature engineering to improve model performance.

## Future Steps

### Feature Engineering:

- Derive new features such as workout intensity scores or average heart rate recovery time.

- Use domain knowledge to create meaningful predictors.

### Hyperparameter Tuning:

- Optimize model parameters using GridSearchCV or RandomizedSearchCV.

### Explore Advanced Models:

- Use ensemble models like Random Forest or XGBoost for better classification performance.

### Deploy the Model:

- Build a web app using Streamlit or Flask for users to input their data and get predictions.

# How to Run the Project

- Go to the [Streamlit app](http://192.168.1.8:8501)
- Input Data: Provide information like average heart rate, session duration, age, gender, fat percentage, and workout type in the sidebar.
- Predict Calories Burned: Click the Predict button to get an estimate of the calories burned for the given

# Contributions

Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or raise an issue.


