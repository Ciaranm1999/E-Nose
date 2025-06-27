import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def debug_100_percent_accuracy():
    """Debug why models are getting 100% training accuracy"""
    
    print("="*60)
    print("DEBUGGING 100% TRAINING ACCURACY ISSUE")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
    test_df = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
    
    # Truncate at NaN
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    def truncate_at_nan(df, cols):
        nan_mask = df[cols].isnull().any(axis=1)
        if nan_mask.any():
            first_nan_idx = nan_mask.idxmax()
            return df.loc[:first_nan_idx - 1].reset_index(drop=True)
        return df
    
    train_df = truncate_at_nan(train_df, feature_cols)
    test_df = truncate_at_nan(test_df, feature_cols)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Create time-based labels
    def create_time_labels(df):
        if 'Time_to_Spoilage_Minutes' not in df.columns:
            print("❌ No Time_to_Spoilage_Minutes column found!")
            return None
            
        def label_time_to_spoilage(time_to_spoilage):
            if time_to_spoilage > 48 * 60:  # > 48 hours
                return 0  # Fresh
            elif time_to_spoilage > 24 * 60:  # 24-48 hours
                return 1  # Spoiling
            else:  # < 24 hours
                return 2  # Spoiled
        
        return df['Time_to_Spoilage_Minutes'].apply(label_time_to_spoilage)
    
    # Check if the time column exists
    print(f"Available columns in train_df: {list(train_df.columns)}")
    print(f"Available columns in test_df: {list(test_df.columns)}")
    
    # Create labels
    y_train = create_time_labels(train_df)
    y_test = create_time_labels(test_df)
    
    if y_train is None or y_test is None:
        print("❌ Cannot create time-based labels. Stopping.")
        return
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].mean())
    
    print(f"\nLabel distributions:")
    print(f"Train labels: {np.bincount(y_train)}")
    print(f"Test labels: {np.bincount(y_test)}")
    
    # Check if labels are too sequential/predictable
    print(f"\nFirst 20 training labels: {y_train[:20].tolist()}")
    print(f"Last 20 training labels: {y_train[-20:].tolist()}")
    
    # Check feature ranges
    print(f"\nFeature statistics:")
    for col in feature_cols:
        print(f"{col}:")
        print(f"  Train: {X_train[col].min():.2f} to {X_train[col].max():.2f} (mean: {X_train[col].mean():.2f})")
        print(f"  Test:  {X_test[col].min():.2f} to {X_test[col].max():.2f} (mean: {X_test[col].mean():.2f})")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test simple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=100)
    }
    
    print(f"\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Get predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"  Training Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
        print(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        print(f"  Overfitting Gap: {train_acc - test_acc:.3f}")
        
        if train_acc == 1.0:
            print("  ⚠️  PERFECT TRAINING ACCURACY - LIKELY OVERFITTING!")
            
        # Show confusion on test set
        print(f"  Test predictions distribution: {np.bincount(test_pred)}")
    
    # Check for data leakage - are the features too predictive?
    print(f"\n" + "="*60)
    print("CHECKING FOR DATA LEAKAGE")
    print("="*60)
    
    # Plot time-series of labels vs key features
    plt.figure(figsize=(15, 10))
    
    # Calculate time from start
    train_df['Hours_from_start'] = (train_df['Timestamp'] - train_df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    plt.subplot(3, 1, 1)
    plt.plot(train_df['Hours_from_start'], y_train, 'o-', markersize=2)
    plt.title('Labels Over Time (0=Fresh, 1=Spoiling, 2=Spoiled)')
    plt.ylabel('Label')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(train_df['Hours_from_start'], train_df['MQ3_Top_PPM'], 'b-', alpha=0.7, label='MQ3 Top')
    plt.plot(train_df['Hours_from_start'], train_df['MQ3_Bottom_PPM'], 'r-', alpha=0.7, label='MQ3 Bottom')
    plt.title('MQ3 Sensors Over Time')
    plt.ylabel('PPM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    colors = ['green', 'orange', 'red']
    for i in range(3):
        mask = y_train == i
        if mask.any():
            plt.scatter(train_df.loc[mask, 'Hours_from_start'], 
                       train_df.loc[mask, 'MQ3_Top_PPM'], 
                       c=colors[i], alpha=0.6, s=10, label=f'Class {i}')
    plt.title('MQ3 Top PPM Colored by Label')
    plt.xlabel('Hours from Start')
    plt.ylabel('MQ3 Top PPM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 ANALYSIS COMPLETE!")
    print("If training accuracy is 100%, possible causes:")
    print("1. Time-based labels are too predictable (sequential pattern)")
    print("2. Features have clear temporal trends that perfectly separate classes")
    print("3. Small dataset allows perfect memorization")
    print("4. Data leakage - features contain information that shouldn't be available")

if __name__ == "__main__":
    debug_100_percent_accuracy()
