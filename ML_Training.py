import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train the spoilage prediction model
def train_model(df):
    # Feature selection
    features = df[['BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm', 'time_seconds']]
    target = df['spoilage_label']  # Assuming spoilage_label is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse:.2f}')

    return model

# Save the trained model
def save_model(model, model_path):
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # File paths
    data_file_path = '../data/raw_data.csv'
    model_file_path = '../models/spoilage_model.pkl'

    # Load data
    data = load_data(data_file_path)

    # Train the model
    trained_model = train_model(data)

    # Save the model
    save_model(trained_model, model_file_path)