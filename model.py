import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
data = pd.read_csv('athlete_performance_large.csv')

# Data preprocessing
label_encoders = {}
categorical_cols = ['Gender', 'Sport_Type', 'Exercise_Type', 'Intensity_Level', 'Hydration_Level']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and targets
features = ['Age', 'Gender', 'Sport_Type', 'Exercise_Type', 'Duration_Minutes', 
            'Intensity_Level', 'Heart_Rate', 'Hydration_Level', 'Fatigue_Score', 'Speed_Score', 'Strength_Score']

# Handle Injury History with NaN and unexpected values
data['Injury_History_Binary'] = (
    data['Injury_History']
    .fillna('Minor')  # Fill missing values
    .str.strip()  # Remove whitespace
    .str.title()  # Standardize casing
    .map({'Minor': 0, 'Moderate': 1, 'Severe': 2})
    .fillna(0)  # Handle unexpected values
)

# Prepare the data
X = data[features]
y_endurance = data['Endurance_Score']
y_injury = data['Injury_History_Binary']
y_calories = data['Calories_Consumed']
y_sleep = data['Sleep_Hours']
y_recovery = data['Recovery_Time_Days']
y_protein = data['Protein_g']
y_carbs = data['Carbs_g']
y_fats = data['Fats_g']

# Split data for each target
X_train_end, X_test_end, y_train_end, y_test_end = train_test_split(X, y_endurance, test_size=0.2, random_state=42)
X_train_inj, X_test_inj, y_train_inj, y_test_inj = train_test_split(X, y_injury, test_size=0.2, random_state=42)
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X, y_calories, test_size=0.2, random_state=42)
X_train_slp, X_test_slp, y_train_slp, y_test_slp = train_test_split(X, y_sleep, test_size=0.2, random_state=42)
X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(X, y_recovery, test_size=0.2, random_state=42)
X_train_pro, X_test_pro, y_train_pro, y_test_pro = train_test_split(X, y_protein, test_size=0.2, random_state=42)
X_train_carbs, X_test_carbs, y_train_carbs, y_test_carbs = train_test_split(X, y_carbs, test_size=0.2, random_state=42)
X_train_fats, X_test_fats, y_train_fats, y_test_fats = train_test_split(X, y_fats, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = [f for f in features if f not in categorical_cols]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_cols)])

# Build models
# Endurance Score (Regression)
endurance_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
endurance_model.fit(X_train_end, y_train_end)

# Injury History (Classification)
injury_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    ))])
injury_model.fit(X_train_inj, y_train_inj)

# Calories Consumed (Regression)
calories_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
calories_model.fit(X_train_cal, y_train_cal)

# Sleep Hours (Regression)
sleep_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
sleep_model.fit(X_train_slp, y_train_slp)

recovery_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
recovery_model.fit(X_train_rec, y_train_rec)

# Protein Prediction Model
protein_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))])
protein_model.fit(X_train_pro, y_train_pro)

# Carbs Prediction Model
carbs_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))])
carbs_model.fit(X_train_carbs, y_train_carbs)

# Fats Prediction Model 
fats_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))])
fats_model.fit(X_train_fats, y_train_fats)

# Evaluate models
def evaluate_model(model, X_test, y_test, model_type):
    y_pred = model.predict(X_test)
    if model_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Sample prediction: True={y_test.iloc[0]}, Predicted={y_pred[0]:.2f}")
    else:
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))
        print(f"Sample prediction: True={y_test.iloc[0]}, Predicted={y_pred[0]}")

print("Endurance Score Model Evaluation:")
evaluate_model(endurance_model, X_test_end, y_test_end, 'regression')

print("\nInjury History Model Evaluation:")
evaluate_model(injury_model, X_test_inj, y_test_inj, 'classification')

print("\nCalories Consumed Model Evaluation:")
evaluate_model(calories_model, X_test_cal, y_test_cal, 'regression')

print("\nSleep Hours Model Evaluation:")
evaluate_model(sleep_model, X_test_slp, y_test_slp, 'regression')

print("\nRecovery Time Model Evaluation:")
evaluate_model(recovery_model, X_test_rec, y_test_rec, 'regression')

print("\nProtein Model Evaluation:")
evaluate_model(protein_model, X_test_pro, y_test_pro, 'regression')

print("\nCarbs Model Evaluation:")
evaluate_model(carbs_model, X_test_carbs, y_test_carbs, 'regression')

print("\nFats Model Evaluation:")
evaluate_model(fats_model, X_test_fats, y_test_fats, 'regression')

# Save models
joblib.dump(endurance_model, 'models/endurance_model.pkl')
joblib.dump(injury_model, 'models/injury_model.pkl')
joblib.dump(calories_model, 'models/calories_model.pkl')
joblib.dump(sleep_model, 'models/sleep_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(protein_model, 'models/protein_model.pkl')
joblib.dump(carbs_model, 'models/carbs_model.pkl')
joblib.dump(fats_model, 'models/fats_model.pkl')

# Prediction function
def predict_athlete_performance(user_input):
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in user_input:
            try:
                le = label_encoders[col]
                input_df[col] = le.transform([user_input[col]])[0]
            except ValueError:
                input_df[col] = 0
    
    # Get predictions
    endurance = endurance_model.predict(input_df[features])[0]
    calories = calories_model.predict(input_df[features])[0]
    sleep = sleep_model.predict(input_df[features])[0]
    
    # Get injury probability and classify based on thresholds
    injury_proba = injury_model.predict_proba(input_df[features])[0]
    max_prob = max(injury_proba)  # Get highest probability among all classes
    
    if 0.25 <= max_prob < 0.5:
        injury_pred = "Minor"
    elif 0.5 <= max_prob < 0.75:
        injury_pred = "Moderate"
    else:
        injury_pred = "Severe"
    
    return {
        'Endurance_Score': round(endurance, 2),
        'Injury_History': injury_pred,
        'Calories_Consumed': round(calories),
        'Sleep_Hours': round(sleep, 1),
        'Protein_g': round(protein_model.predict(input_df)[0]),
        'Carbs_g': round(carbs_model.predict(input_df)[0]),
        'Fats_g': round(fats_model.predict(input_df)[0])
    }

# Example usage
example_input = {
    'Age': 33,
    'Gender': 'Female',
    'Sport_Type': 'Basketball',
    'Exercise_Type': 'Strength',
    'Duration_Minutes': 41,
    'Intensity_Level': 'Low',
    'Heart_Rate': 163,
    'Protein_g': 96,
    'Carbs_g': 106,
    'Fats_g': 85,
    'Hydration_Level': 'Moderate',
    'Fatigue_Score': 8,
    'Speed_Score': 13.65,
    'Strength_Score': 3.08
}

print("\nExample Prediction:")
print(predict_athlete_performance(example_input))