import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from datetime import datetime, timedelta
# Added supervised learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load both datasets"""
    train_df = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
    test_df = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
    
    # Truncate at NaN
    def truncate_at_nan(df, feature_cols):
        feature_cols_available = [col for col in feature_cols if col in df.columns]
        nan_mask = df[feature_cols_available].isnull().any(axis=1)
        if nan_mask.any():
            first_nan_idx = nan_mask.idxmax()
            return df.loc[:first_nan_idx - 1].reset_index(drop=True)
        return df
    
    # Key features for visualization
    key_features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    train_df = truncate_at_nan(train_df, key_features)
    test_df = truncate_at_nan(test_df, key_features)
    
    return train_df, test_df

def visualize_mq3_over_time(df, batch_name="Batch", spoilage_ranges=None):
    """Visualize MQ3 sensor data over time to identify spoilage patterns
    
    Args:
        df: DataFrame with sensor data
        batch_name: Name of the batch for titles
        spoilage_ranges: Dict with 'start' and 'end' datetime strings for known spoilage periods
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'{batch_name} - Sensor Data Over Time Analysis', fontsize=16)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Much better spacing between plots
    
    # Calculate relative time in hours from start
    df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    # Convert spoilage ranges to hours if provided
    spoilage_hours = None
    if spoilage_ranges and 'start' in spoilage_ranges and 'end' in spoilage_ranges:
        try:
            start_dt = pd.to_datetime(spoilage_ranges['start'])
            end_dt = pd.to_datetime(spoilage_ranges['end'])
            
            # Calculate hours from experiment start
            start_hours = (start_dt - df['Timestamp'].iloc[0]).total_seconds() / 3600
            end_hours = (end_dt - df['Timestamp'].iloc[0]).total_seconds() / 3600
            
            spoilage_hours = {'start': start_hours, 'end': end_hours}
            print(f"Hand-written spoilage certainty range: {start_hours:.1f}h to {end_hours:.1f}h")
        except Exception as e:
            print(f"Warning: Could not parse spoilage ranges: {e}")
    # 1. MQ3 Bottom PPM over time (changed from Top)
    # Only plot valid (non-NaN) MQ3 data to show where sensors cut out
    valid_mq3_bottom = df['MQ3_Bottom_PPM'].notna()
    axes[0,0].plot(df.loc[valid_mq3_bottom, 'Hours_from_start'], 
                   df.loc[valid_mq3_bottom, 'MQ3_Bottom_PPM'], 
                   'r-', linewidth=2, label='MQ3 Bottom')
    axes[0,0].set_title('MQ3 Bottom PPM vs Time (Primary Sensor)')
    axes[0,0].set_xlabel('Hours from Start')
    axes[0,0].set_ylabel('PPM')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Add spoilage certainty range if provided
    if spoilage_hours:
        axes[0,0].axvspan(spoilage_hours['start'], spoilage_hours['end'], 
                         alpha=0.3, color='red', label='Spoilage Certainty Range')
        axes[0,0].legend()
    
    # Add vertical line to show end of spoiled classification (where MQ3 data cuts out)
    if valid_mq3_bottom.any():
        last_valid_hour = df.loc[valid_mq3_bottom, 'Hours_from_start'].iloc[-1]
        axes[0,0].axvline(x=last_valid_hour, color='red', linestyle='--', alpha=0.7)
    # 2. MQ3 Top PPM over time (swapped with Bottom)
    # Only plot valid (non-NaN) MQ3 data to show where sensors cut out
    valid_mq3_top = df['MQ3_Top_PPM'].notna()
    axes[0,1].plot(df.loc[valid_mq3_top, 'Hours_from_start'], 
                   df.loc[valid_mq3_top, 'MQ3_Top_PPM'], 
                   'b-', linewidth=2, label='MQ3 Top')
    axes[0,1].set_title('MQ3 Top PPM vs Time')
    axes[0,1].set_xlabel('Hours from Start')
    axes[0,1].set_ylabel('PPM')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Add spoilage certainty range
    if spoilage_hours:
        axes[0,1].axvspan(spoilage_hours['start'], spoilage_hours['end'], 
                         alpha=0.3, color='red', label='Spoilage Certainty Range')
        axes[0,1].legend()
    
    # Add vertical line to show end of spoiled classification (where MQ3 data cuts out)
    if valid_mq3_top.any():
        last_valid_hour = df.loc[valid_mq3_top, 'Hours_from_start'].iloc[-1]
        axes[0,1].axvline(x=last_valid_hour, color='red', linestyle='--', alpha=0.7)
    
    # 3. Both MQ3 sensors together
    # Only plot valid (non-NaN) MQ3 data for both sensors
    valid_mq3_top = df['MQ3_Top_PPM'].notna()
    valid_mq3_bottom = df['MQ3_Bottom_PPM'].notna()
    
    axes[1,0].plot(df.loc[valid_mq3_top, 'Hours_from_start'], 
                   df.loc[valid_mq3_top, 'MQ3_Top_PPM'], 
                   'b-', linewidth=2, label='MQ3 Top', alpha=0.8)
    axes[1,0].plot(df.loc[valid_mq3_bottom, 'Hours_from_start'], 
                   df.loc[valid_mq3_bottom, 'MQ3_Bottom_PPM'], 
                   'r-', linewidth=2, label='MQ3 Bottom', alpha=0.8)
    axes[1,0].set_title('Both MQ3 Sensors Comparison')
    axes[1,0].set_xlabel('Hours from Start')
    axes[1,0].set_ylabel('PPM')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()    # Add spoilage certainty range
    if spoilage_hours:
        axes[1,0].axvspan(spoilage_hours['start'], spoilage_hours['end'], 
                         alpha=0.3, color='red', label='Spoilage Certainty Range')
        axes[1,0].legend()
    
    # Add vertical line to show end of spoiled classification (where MQ3 data cuts out)
    if valid_mq3_top.any() or valid_mq3_bottom.any():
        # Use the latest cutoff point from either sensor
        last_valid_hours = []
        if valid_mq3_top.any():
            last_valid_hours.append(df.loc[valid_mq3_top, 'Hours_from_start'].iloc[-1])
        if valid_mq3_bottom.any():
            last_valid_hours.append(df.loc[valid_mq3_bottom, 'Hours_from_start'].iloc[-1])
        
        last_valid_hour = max(last_valid_hours)
        axes[1,0].axvline(x=last_valid_hour, color='red', linestyle='--', alpha=0.7)
    
    # 4. Environmental factors
    axes[1,1].plot(df['Hours_from_start'], df['BME_Temp'], 'g-', linewidth=2, label='Temperature', alpha=0.8)
    ax2 = axes[1,1].twinx()
    ax2.plot(df['Hours_from_start'], df['BME_Humidity'], 'purple', linewidth=2, label='Humidity', alpha=0.8)
    axes[1,1].set_title('Environmental Conditions')
    axes[1,1].set_xlabel('Hours from Start')
    axes[1,1].set_ylabel('Temperature (°C)', color='g')
    ax2.set_ylabel('Humidity (%)', color='purple')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 5. VOC sensor
    axes[2,0].plot(df['Hours_from_start'], df['BME_VOC_Ohm'], 'brown', linewidth=2, label='VOC Resistance')
    axes[2,0].set_title('VOC Sensor Over Time')
    axes[2,0].set_xlabel('Hours from Start')
    axes[2,0].set_ylabel('Resistance (Ohm)')
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].legend()
    
    # 6. Rate of change in MQ3 Bottom (derivative) - changed from Top
    if len(df) > 1:
        mq3_diff = np.diff(df['MQ3_Bottom_PPM'])
        time_diff = np.diff(df['Hours_from_start'])
        rate_of_change = mq3_diff / time_diff
        
        axes[2,1].plot(df['Hours_from_start'][1:], rate_of_change, 'orange', linewidth=2, label='Rate of Change')
        axes[2,1].set_title('MQ3 Bottom Rate of Change (Derivative)')
        axes[2,1].set_xlabel('Hours from Start')
        axes[2,1].set_ylabel('PPM/Hour')
        axes[2,1].grid(True, alpha=0.3)
        axes[2,1].legend()
        axes[2,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def compare_labeling_methods(df, batch_name="Batch"):
    """Compare K-means clustering with time-based labeling"""
    
    print(f"\n{'='*80}")
    print(f"LABELING METHODS COMPARISON - {batch_name}")
    print(f"{'='*80}")
    
    # Prepare features for clustering
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    X = df[feature_cols].fillna(df[feature_cols].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based labels (if available)
    if 'Time_to_Spoilage_Minutes' in df.columns:
        def create_time_labels(time_to_spoilage):
            if time_to_spoilage > 48 * 60:  # > 48 hours
                return 0  # Fresh
            elif time_to_spoilage > 24 * 60:  # 24-48 hours
                return 1  # Spoiling
            else:  # < 24 hours
                return 2  # Spoiled
        
        time_labels = df['Time_to_Spoilage_Minutes'].apply(create_time_labels)
        
        print(f"Time-based label distribution:")
        print(f"Fresh (0): {(time_labels == 0).sum()}")
        print(f"Spoiling (1): {(time_labels == 1).sum()}")
        print(f"Spoiled (2): {(time_labels == 2).sum()}")
    
    # K-means clustering with different numbers of clusters
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        if k == 3:  # Save 3-cluster result for comparison
            kmeans_3_labels = cluster_labels
            kmeans_3_centers = kmeans.cluster_centers_
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    plt.subplot(2, 2, 1)
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    plt.legend()
    
    # Compare time-based vs K-means (3 clusters)
    if 'Time_to_Spoilage_Minutes' in df.columns:
        plt.subplot(2, 2, 2)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Time_Hours': df['Hours_from_start'],
            'Time_Labels': time_labels,
            'KMeans_Labels': kmeans_3_labels,
            'MQ3_Bottom': df['MQ3_Bottom_PPM']  # Changed from Top to Bottom
        })
        
        # Plot time-based labels
        colors_time = ['green', 'orange', 'red']
        for i in range(3):
            mask = comparison_df['Time_Labels'] == i
            if mask.any():
                plt.scatter(comparison_df.loc[mask, 'Time_Hours'], 
                          comparison_df.loc[mask, 'MQ3_Bottom'], 
                          c=colors_time[i], alpha=0.6, s=20,
                          label=f'Time-based {i}')
        
        plt.title('Time-based Labels vs MQ3 Bottom')
        plt.xlabel('Hours from Start')
        plt.ylabel('MQ3 Bottom PPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        
        # Plot K-means labels
        colors_kmeans = ['blue', 'purple', 'brown']
        for i in range(3):
            mask = comparison_df['KMeans_Labels'] == i
            if mask.any():
                plt.scatter(comparison_df.loc[mask, 'Time_Hours'], 
                          comparison_df.loc[mask, 'MQ3_Bottom'], 
                          c=colors_kmeans[i], alpha=0.6, s=20,
                          label=f'K-means {i}')
        
        plt.title('K-means Labels vs MQ3 Bottom')
        plt.xlabel('Hours from Start')
        plt.ylabel('MQ3 Bottom PPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate agreement between methods
        rand_score = adjusted_rand_score(time_labels, kmeans_3_labels)
        
        plt.subplot(2, 2, 4)
        
        # Confusion matrix between methods
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(time_labels, kmeans_3_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Time vs K-means Agreement\nAdjusted Rand Score: {rand_score:.3f}')
        plt.xlabel('K-means Labels')
        plt.ylabel('Time-based Labels')
        
        print(f"\nMethod Comparison:")
        print(f"Adjusted Rand Score (agreement): {rand_score:.3f}")
        print(f"Higher score (closer to 1) means better agreement")
        
        if rand_score < 0.3:
            print("⚠️  LOW AGREEMENT: K-means finds different patterns than time-based labels")
        elif rand_score < 0.6:
            print("📊 MODERATE AGREEMENT: Some similarity between methods")
        else:
            print("✅ HIGH AGREEMENT: Methods largely agree")
    
    plt.tight_layout()
    plt.show()
    
    return kmeans_3_labels, silhouette_scores

def create_interactive_spoilage_detection(df, batch_name="Batch"):
    """Create interactive plots to help identify spoilage onset"""
    
    print(f"\n{'='*60}")
    print(f"INTERACTIVE SPOILAGE DETECTION - {batch_name}")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{batch_name} - Identify Spoilage Onset Points', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Calculate moving averages and derivatives
    window = max(5, len(df) // 20)  # Adaptive window size
    df['MQ3_Bottom_MA'] = df['MQ3_Bottom_PPM'].rolling(window=window, center=True).mean()
    df['MQ3_Top_MA'] = df['MQ3_Top_PPM'].rolling(window=window, center=True).mean()
    
    # 1. Raw data with moving average (changed to MQ3 Bottom)
    axes[0,0].plot(df['Hours_from_start'], df['MQ3_Bottom_PPM'], 'lightcoral', alpha=0.5, label='Raw MQ3 Bottom')
    axes[0,0].plot(df['Hours_from_start'], df['MQ3_Bottom_MA'], 'red', linewidth=3, label='Moving Average')
    axes[0,0].set_title('MQ3 Bottom: Raw vs Smoothed')
    axes[0,0].set_xlabel('Hours from Start')
    axes[0,0].set_ylabel('PPM')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Derivative analysis (changed to MQ3 Bottom)
    if len(df) > window:
        derivative = np.gradient(df['MQ3_Bottom_MA'].fillna(method='bfill').fillna(method='ffill'))
        axes[0,1].plot(df['Hours_from_start'], derivative, 'red', linewidth=2, label='Rate of Change')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,1].set_title('Rate of Change Analysis (MQ3 Bottom)')
        axes[0,1].set_xlabel('Hours from Start')
        axes[0,1].set_ylabel('PPM/Hour')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Find significant change points
        threshold = np.std(derivative) * 1.5
        change_points = np.where(np.abs(derivative) > threshold)[0]
        
        if len(change_points) > 0:
            for cp in change_points[:5]:  # Show first 5 change points
                axes[0,1].axvline(x=df['Hours_from_start'].iloc[cp], color='orange', 
                                linestyle='--', alpha=0.7)
            
            print(f"Significant change points detected at hours: {df['Hours_from_start'].iloc[change_points[:5]].values}")
    
    # 3. Cumulative change (changed to MQ3 Bottom)
    baseline = df['MQ3_Bottom_PPM'].iloc[:min(10, len(df))].mean()
    cumulative_change = df['MQ3_Bottom_PPM'] - baseline
    axes[1,0].plot(df['Hours_from_start'], cumulative_change, 'green', linewidth=2, label='Cumulative Change')
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Baseline')
    axes[1,0].set_title('Cumulative Change from Baseline (MQ3 Bottom)')
    axes[1,0].set_xlabel('Hours from Start')
    axes[1,0].set_ylabel('PPM Change')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Sensor correlation over time (Bottom-Top correlation)
    if len(df) > 1:
        # Rolling correlation between bottom and top sensors
        rolling_corr = df['MQ3_Bottom_PPM'].rolling(window=window).corr(df['MQ3_Top_PPM'])
        axes[1,1].plot(df['Hours_from_start'], rolling_corr, 'purple', linewidth=2, label='Bottom-Top Correlation')
        axes[1,1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Correlation')
        axes[1,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Correlation')
        axes[1,1].set_title('Sensor Correlation Over Time')
        axes[1,1].set_xlabel('Hours from Start')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def determine_optimal_labeling_method(train_df, test_df):
    """Determine the best labeling method based on data analysis"""
    
    print(f"\n{'='*80}")
    print("DETERMINING OPTIMAL LABELING METHOD")
    print(f"{'='*80}")
    
    # Analyze clustering performance
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    results = {}
    for name, df in [("Batch 1", train_df), ("Batch 2", test_df)]:
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X_scaled = StandardScaler().fit_transform(X)
        
        # Test different cluster numbers
        silhouette_scores = []
        for k in range(2, 6):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(0)
        
        best_k = range(2, 6)[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        
        # Check time-based labeling quality if available
        time_quality = "N/A"
        if 'Time_to_Spoilage_Minutes' in df.columns:
            time_labels = create_time_based_labels(df)
            if len(np.unique(time_labels)) > 1:
                # Compare with best K-means
                kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                kmeans_labels = kmeans_best.fit_predict(X_scaled)
                agreement = adjusted_rand_score(time_labels, kmeans_labels)
                time_quality = f"Agreement with K-means: {agreement:.3f}"
        
        results[name] = {
            'best_k': best_k,
            'best_silhouette': best_score,
            'time_quality': time_quality
        }
    
    # Display results
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Best K-means clusters: {result['best_k']}")
        print(f"  Silhouette score: {result['best_silhouette']:.3f}")
        print(f"  Time-based labeling: {result['time_quality']}")
    
    # Make recommendation
    avg_silhouette = np.mean([r['best_silhouette'] for r in results.values()])
    
    print(f"\n🎯 OPTIMAL LABELING RECOMMENDATION:")
    print("-" * 50)
    
    if avg_silhouette > 0.4:
        optimal_method = "K-means clustering"
        print(f"✅ Use K-MEANS CLUSTERING (3 clusters)")
        print(f"   Reason: High silhouette scores ({avg_silhouette:.3f}) indicate natural data groupings")
    elif 'Time_to_Spoilage_Minutes' in train_df.columns:
        optimal_method = "time-based"
        print(f"✅ Use TIME-BASED LABELING")
        print(f"   Reason: Moderate clustering performance, time-based labels available")
    else:
        optimal_method = "hybrid"
        print(f"✅ Use HYBRID APPROACH")
        print(f"   Reason: Manual labeling with visual inspection recommended")
    
    return optimal_method, results

def create_time_based_labels(df, verbose=False):
    """Create spoilage labels based on time to spoilage"""
    if 'Time_to_Spoilage_Minutes' not in df.columns:
        if verbose:
            print("⚠️  No Time_to_Spoilage_Minutes column found")
        return None
        
    def label_time_to_spoilage(time_to_spoilage):
        if time_to_spoilage > 48 * 60:  # > 48 hours in minutes
            return 0  # Fresh
        elif time_to_spoilage > 24 * 60:  # 24-48 hours
            return 1  # Spoiling
        else:  # < 24 hours
            return 2  # Spoiled
    
    labels = df['Time_to_Spoilage_Minutes'].apply(label_time_to_spoilage)
    
    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"📊 Time-based label distribution:")
        for label, count in zip(unique, counts):
            class_name = ['Fresh', 'Spoiling', 'Spoiled'][int(label)]
            print(f"   {class_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels

def create_optimal_labels(df, method="time-based"):
    """Create labels using the optimal method"""
    
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    if method == "time-based" and 'Time_to_Spoilage_Minutes' in df.columns:
        return create_time_based_labels(df)
    
    elif method == "K-means clustering":
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        return kmeans.fit_predict(X_scaled)
    
    else:  # hybrid approach - use time-based if available, else K-means
        if 'Time_to_Spoilage_Minutes' in df.columns:
            return create_time_based_labels(df)
        else:
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            return kmeans.fit_predict(X_scaled)

def get_classifiers():
    """Define all classifiers to test"""
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    return classifiers

def evaluate_supervised_models(train_df, test_df, optimal_method, overfitting_threshold=0.2):
    """Evaluate all supervised learning models with overfitting detection"""
    
    print(f"\n{'='*80}")
    print("SUPERVISED LEARNING MODEL EVALUATION")
    print(f"{'='*80}")
    
    print(f"🏷️  LABELING METHOD BEING USED: {optimal_method.upper()}")
    print("-" * 50)
    
    if optimal_method == "time-based":
        print("Using TIME-BASED LABELING:")
        print("• Fresh: > 48 hours before predicted spoilage")
        print("• Spoiling: 24-48 hours before predicted spoilage")  
        print("• Spoiled: < 24 hours before predicted spoilage")
    elif optimal_method == "K-means clustering":
        print("Using K-MEANS CLUSTERING:")
        print("• Data-driven clusters based on sensor patterns")
        print("• 3 clusters automatically identified")
    else:
        print("Using HYBRID/MANUAL LABELING:")
        print("• Combination of time-based and visual inspection")
        print("• Manually adjusted based on sensor patterns")
    
    print("-" * 50)
    
    # Prepare features
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    # Create optimal labels
    y_train = create_optimal_labels(train_df, optimal_method)
    y_test = create_optimal_labels(test_df, optimal_method)
    
    if y_train is None or y_test is None:
        print("❌ Cannot create labels with the specified method")
        return None, None
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Using labeling method: {optimal_method}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    # Debug: Check for potential data leakage or overly simple patterns
    print(f"\n🔍 DEBUGGING TRAINING DATA:")
    print(f"Feature ranges in training data:")
    for col in feature_cols:
        print(f"  {col}: {X_train[col].min():.2f} to {X_train[col].max():.2f}")
    
    # Check if labels are too predictable based on simple rules
    print(f"\nLabel transitions in time order:")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    print("-" * 50)
    
    # Get classifiers
    classifiers = get_classifiers()
    results = {}
    trained_models = {}
    
    # Evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nEvaluating {name}...")
        
        try:
            # Train model
            clf.fit(X_train_scaled, y_train)
            
            # Get predictions
            train_pred = clf.predict(X_train_scaled)
            test_pred = clf.predict(X_test_scaled)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Cross-validation on training set
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
              # Overfitting detection
            # For small datasets like ours (~3500 samples), a 20% difference between
            # training and test accuracy is a reasonable threshold to detect overfitting.
            # This threshold accounts for:
            # 1. Natural variance in small datasets
            # 2. Class imbalance effects
            # 3. Limited test set size
            # You can adjust this threshold based on your requirements:
            # - Stricter: 0.1 (10%) - flags more models as overfitting            # - More lenient: 0.3 (30%) - allows larger train-test gaps
            overfitting_score = train_acc - test_acc
            is_overfitting = overfitting_score > overfitting_threshold
            
            # F1 scores
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_score': overfitting_score,
                'is_overfitting': is_overfitting,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'test_predictions': test_pred,
                'train_predictions': train_pred
            }
            
            trained_models[name] = clf
            
            # Print results
            print(f"  Train Accuracy: {train_acc:.3f}")
            print(f"  Test Accuracy: {test_acc:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Overfitting Score: {overfitting_score:.3f}")
            if is_overfitting:
                print(f"  ⚠️  OVERFITTING DETECTED!")
            else:
                print(f"  ✅ Good generalization")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = None
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        # Best model based on test accuracy, penalized for overfitting
        best_model = max(valid_results.keys(), 
                        key=lambda x: valid_results[x]['test_accuracy'] - 
                                    (0.1 * valid_results[x]['overfitting_score'] if valid_results[x]['is_overfitting'] else 0))
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Test Accuracy: {valid_results[best_model]['test_accuracy']:.3f}")
        print(f"   Overfitting: {'Yes' if valid_results[best_model]['is_overfitting'] else 'No'}")
    
    return results, trained_models, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

def visualize_model_performance(results):
    """Visualize model performance comparison"""
    
    if not results or not any(results.values()):
        print("❌ No valid results to visualize")
        return
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    model_names = list(valid_results.keys())
    
    # 1. Train vs Test Accuracy
    train_accs = [valid_results[name]['train_accuracy'] for name in model_names]
    test_accs = [valid_results[name]['test_accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0,0].bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    axes[0,0].bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    axes[0,0].set_xlabel('Models')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('Train vs Test Accuracy')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(model_names, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Overfitting Analysis
    overfitting_scores = [valid_results[name]['overfitting_score'] for name in model_names]
    colors = ['red' if valid_results[name]['is_overfitting'] else 'green' for name in model_names]
    
    axes[0,1].bar(model_names, overfitting_scores, color=colors, alpha=0.7)
    axes[0,1].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    axes[0,1].set_xlabel('Models')
    axes[0,1].set_ylabel('Train - Test Accuracy')
    axes[0,1].set_title('Overfitting Analysis')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Cross-validation scores
    cv_means = [valid_results[name]['cv_mean'] for name in model_names]
    cv_stds = [valid_results[name]['cv_std'] for name in model_names]
    
    axes[1,0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
    axes[1,0].set_xlabel('Models')
    axes[1,0].set_ylabel('CV Accuracy')
    axes[1,0].set_title('Cross-Validation Performance')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. F1 Scores
    test_f1s = [valid_results[name]['test_f1'] for name in model_names]
    
    axes[1,1].bar(model_names, test_f1s, alpha=0.8, color='purple')
    axes[1,1].set_xlabel('Models')
    axes[1,1].set_ylabel('F1 Score')
    axes[1,1].set_title('Test F1 Scores')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_model_predictions_on_time_series(train_df, test_df, results, spoilage_ranges):
    """Visualize model predictions overlaid on MQ3 vs time plots"""
    
    if not results or not any(results.values()):
        print("❌ No valid results to visualize")
        return
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Get best models (top 3)
    sorted_models = sorted(valid_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Model Predictions on MQ3 vs Time', fontsize=16)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    datasets = [
        (train_df, "Batch 1 (Training)", spoilage_ranges.get('batch1')), 
        (test_df, "Batch 2 (Test)", spoilage_ranges.get('batch2'))
    ]
    
    colors = ['green', 'orange', 'red']  # Fresh, Spoiling, Spoiled
    labels = ['Fresh', 'Spoiling', 'Spoiled']
    
    for model_idx, (model_name, model_results) in enumerate(sorted_models):
        for dataset_idx, (df, dataset_name, spoilage_range) in enumerate(datasets):
            ax = axes[model_idx, dataset_idx]
            
            # Calculate hours from start
            df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
            
            # Get predictions
            if dataset_idx == 0:  # Training data
                predictions = model_results['train_predictions']
            else:  # Test data
                predictions = model_results['test_predictions']
              # Plot MQ3 data with color-coded predictions
            # Use MQ3 Bottom for both batches since it shows clearer patterns
            mq3_data = df['MQ3_Bottom_PPM']
            mq3_label = 'MQ3 Bottom PPM'
            
            for class_label in range(3):
                mask = predictions == class_label
                if mask.any():
                    ax.scatter(df['Hours_from_start'][mask], mq3_data[mask], 
                             c=colors[class_label], alpha=0.6, s=30, 
                             label=f'{labels[class_label]} ({mask.sum()} points)')
              # Add spoilage certainty range if provided
            if spoilage_range:
                try:
                    start_dt = pd.to_datetime(spoilage_range['start'])
                    end_dt = pd.to_datetime(spoilage_range['end'])
                    start_hours = (start_dt - df['Timestamp'].iloc[0]).total_seconds() / 3600
                    end_hours = (end_dt - df['Timestamp'].iloc[0]).total_seconds() / 3600
                    
                    ax.axvspan(start_hours, end_hours, alpha=0.2, color='black', 
                             label='Spoilage Certainty Range')
                except:
                    pass
            
            ax.set_title(f'{model_name} - {dataset_name}')
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel(mq3_label)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def suggest_optimal_labeling_strategy(train_df, test_df):
    """Suggest the best labeling strategy based on analysis"""
    
    print(f"\n{'='*80}")
    print("OPTIMAL LABELING STRATEGY RECOMMENDATION")
    print(f"{'='*80}")
    
    # Analyze both datasets
    results = {}
    
    for name, df in [("Batch 1 (Training)", train_df), ("Batch 2 (Test)", test_df)]:
        feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X_scaled = StandardScaler().fit_transform(X)
        
        # Try different numbers of clusters
        best_k = 2
        best_score = -1
        
        for k in range(2, 6):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        results[name] = {'best_k': best_k, 'best_score': best_score}
    
    print(f"\nCluster Analysis Results:")
    for name, result in results.items():
        print(f"{name}: Optimal clusters = {result['best_k']}, Silhouette score = {result['best_score']:.3f}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 50)
    
    print("1. HYBRID APPROACH (Recommended):")
    print("   • Use your known spoilage time ranges as anchors")
    print("   • Apply K-means to find natural sensor-based groups")
    print("   • Manually adjust boundaries based on visual inspection")
    print("   • Benefits: Combines domain knowledge with data patterns")
    
    print("\n2. PURE K-MEANS APPROACH:")
    print("   • Let data determine natural groupings")
    print("   • Good if time-based labels are unreliable")
    print("   • Risk: Clusters might not match spoilage stages")
    
    print("\n3. REFINED TIME-BASED APPROACH:")
    print("   • Keep current time-based labels")
    print("   • Adjust boundaries based on visual inspection")
    print("   • Use derivative analysis to identify transition points")
    
    print("\n4. ENSEMBLE LABELING:")
    print("   • Create multiple label sets (time-based, K-means, manual)")
    print("   • Train models on each labeling scheme")
    print("   • Use voting or averaging for final predictions")
    
    print(f"\n💡 NEXT STEPS:")
    print("-" * 30)
    print("1. Examine the time-series plots carefully")
    print("2. Identify where you visually see spoilage starting")
    print("3. Compare with current time-based boundaries")
    print("4. Consider creating custom boundaries based on sensor patterns")
    print("5. Test different labeling schemes and compare model performance")

def compare_labeling_methods_with_models(train_df, test_df):
    """Compare time-based vs K-means labeling using supervised learning models"""
    
    print(f"\n{'='*80}")
    print("LABELING METHODS COMPARISON WITH SUPERVISED MODELS")
    print(f"{'='*80}")
    
    # Prepare features
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create labels with both methods
    y_train_time = create_optimal_labels(train_df, "time-based")
    y_test_time = create_optimal_labels(test_df, "time-based")
    
    y_train_kmeans = create_optimal_labels(train_df, "K-means clustering")
    y_test_kmeans = create_optimal_labels(test_df, "K-means clustering")
    
    if y_train_time is None or y_train_kmeans is None:
        print("❌ Cannot create labels with one of the methods")
        return None
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results_comparison = {}
    
    # Test both labeling methods
    for method_name, (y_train, y_test) in [
        ("Time-based", (y_train_time, y_test_time)),
        ("K-means", (y_train_kmeans, y_test_kmeans))
    ]:
        print(f"\n{'='*60}")
        print(f"TESTING {method_name.upper()} LABELING")
        print(f"{'='*60}")
        
        print(f"Training label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        method_results = {}
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Overfitting check
                overfitting_score = train_acc - test_acc
                is_overfitting = overfitting_score > 0.2
                
                method_results[model_name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'test_f1': test_f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'overfitting_score': overfitting_score,
                    'is_overfitting': is_overfitting,
                    'train_predictions': y_train_pred,
                    'test_predictions': y_test_pred
                }
                
                print(f"{model_name:20} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | F1: {test_f1:.3f} | CV: {cv_mean:.3f}±{cv_std:.3f} | {'⚠️ Overfitting' if is_overfitting else '✅ Good'}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                method_results[model_name] = None
        
        results_comparison[method_name] = method_results
    
    # Compare results
    print(f"\n{'='*80}")
    print("LABELING METHODS COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_table = []
    for model_name in models.keys():
        time_result = results_comparison["Time-based"].get(model_name)
        kmeans_result = results_comparison["K-means"].get(model_name)
        
        if time_result and kmeans_result:
            time_acc = time_result['test_accuracy']
            kmeans_acc = kmeans_result['test_accuracy']
            time_f1 = time_result['test_f1']
            kmeans_f1 = kmeans_result['test_f1']
            
            better_acc = "K-means" if kmeans_acc > time_acc else "Time-based"
            better_f1 = "K-means" if kmeans_f1 > time_f1 else "Time-based"
            
            comparison_table.append({
                'Model': model_name,
                'Time_Acc': time_acc,
                'Kmeans_Acc': kmeans_acc,
                'Better_Acc': better_acc,
                'Time_F1': time_f1,
                'Kmeans_F1': kmeans_f1,
                'Better_F1': better_f1
            })
    
    # Print comparison table
    print(f"{'Model':<20} | {'Time Acc':<8} | {'K-means Acc':<11} | {'Better Acc':<12} | {'Time F1':<8} | {'K-means F1':<10} | {'Better F1':<9}")
    print("-" * 100)
    for row in comparison_table:
        print(f"{row['Model']:<20} | {row['Time_Acc']:<8.3f} | {row['Kmeans_Acc']:<11.3f} | {row['Better_Acc']:<12} | {row['Time_F1']:<8.3f} | {row['Kmeans_F1']:<10.3f} | {row['Better_F1']:<9}")
    
    # Overall winner
    time_wins = sum(1 for row in comparison_table if row['Better_Acc'] == "Time-based")
    kmeans_wins = sum(1 for row in comparison_table if row['Better_Acc'] == "K-means")
    
    print(f"\n🏆 OVERALL WINNER (by accuracy): ", end="")
    if kmeans_wins > time_wins:
        print(f"K-MEANS CLUSTERING ({kmeans_wins}/{len(comparison_table)} models)")
        recommended_method = "K-means clustering"
    elif time_wins > kmeans_wins:
        print(f"TIME-BASED LABELING ({time_wins}/{len(comparison_table)} models)")
        recommended_method = "time-based"
    else:
        print(f"TIE ({time_wins}-{kmeans_wins})")
        recommended_method = "hybrid"
    
    print(f"🎯 RECOMMENDED LABELING METHOD: {recommended_method.upper()}")
    
    return results_comparison, recommended_method

def refine_labeling_for_spoiling_class(train_df, test_df):
    """Improve labeling to better capture the 'spoiling' class"""
    
    print(f"\n{'='*80}")
    print("REFINING LABELING FOR BETTER SPOILING CLASS DETECTION")
    print(f"{'='*80}")
    
    # Try different time-based thresholds
    refined_methods = {
        'Original': {'fresh': 48*60, 'spoiling': 24*60},  # Original thresholds
        'Conservative': {'fresh': 60*60, 'spoiling': 12*60},  # Longer fresh period
        'Aggressive': {'fresh': 36*60, 'spoiling': 18*60},  # Shorter fresh period
        'Extended_Spoiling': {'fresh': 54*60, 'spoiling': 18*60},  # Longer spoiling period
    }
    
    def create_refined_time_labels(df, fresh_threshold, spoiling_threshold):
        """Create labels with custom thresholds"""
        if 'Time_to_Spoilage_Minutes' not in df.columns:
            return None
            
        def label_func(time_to_spoilage):
            if time_to_spoilage > fresh_threshold:
                return 0  # Fresh
            elif time_to_spoilage > spoiling_threshold:
                return 1  # Spoiling
            else:
                return 2  # Spoiled
        
        return df['Time_to_Spoilage_Minutes'].apply(label_func)
    
    # Test different approaches
    results = {}
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].mean())
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTesting different labeling approaches:")
    print("-" * 50)
    
    for method_name, thresholds in refined_methods.items():
        # Create labels
        y_train = create_refined_time_labels(train_df, thresholds['fresh'], thresholds['spoiling'])
        y_test = create_refined_time_labels(test_df, thresholds['fresh'], thresholds['spoiling'])
        
        if y_train is None:
            continue
            
        # Check class distribution
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)
        
        print(f"\n{method_name}:")
        print(f"  Fresh: >{thresholds['fresh']/60:.0f}h, Spoiling: {thresholds['spoiling']/60:.0f}-{thresholds['fresh']/60:.0f}h, Spoiled: <{thresholds['spoiling']/60:.0f}h")
        print(f"  Train distribution: Fresh={train_dist[0]}, Spoiling={train_dist[1]}, Spoiled={train_dist[2]}")
        print(f"  Test distribution: Fresh={test_dist[0]}, Spoiling={test_dist[1]}, Spoiled={test_dist[2]}")
        
        # Quick test with Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            
            # Calculate per-class metrics
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            spoiling_precision = report['1']['precision']
            spoiling_recall = report['1']['recall']
            spoiling_f1 = report['1']['f1-score']
            
            print(f"  Spoiling class - Precision: {spoiling_precision:.3f}, Recall: {spoiling_recall:.3f}, F1: {spoiling_f1:.3f}")
            
            results[method_name] = {
                'spoiling_f1': spoiling_f1,
                'spoiling_precision': spoiling_precision,
                'spoiling_recall': spoiling_recall,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Find best method for spoiling class
    if results:
        best_method = max(results.keys(), key=lambda x: results[x]['spoiling_f1'])
        print(f"\n🏆 BEST METHOD FOR SPOILING CLASS: {best_method}")
        print(f"   Spoiling F1-score: {results[best_method]['spoiling_f1']:.3f}")
        
        return results[best_method], best_method
    
    return None, None

def try_advanced_classification_techniques(train_df, test_df, optimal_method):
    """Try advanced techniques to improve spoiling class detection"""
    
    print(f"\n{'='*80}")
    print("ADVANCED CLASSIFICATION TECHNIQUES FOR SPOILING CLASS")
    print(f"{'='*80}")
    
    # Prepare data
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].mean())
    
    y_train = create_optimal_labels(train_df, optimal_method)
    y_test = create_optimal_labels(test_df, optimal_method)
    
    if y_train is None or y_test is None:
        print("❌ Cannot create labels")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    techniques = {}
    
    # 1. Class balancing techniques
    print("\n1. CLASS BALANCING TECHNIQUES:")
    print("-" * 40)
    
    # Balanced Random Forest
    balanced_rf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced',  # This helps with imbalanced classes
        max_depth=10,
        min_samples_split=5
    )
    balanced_rf.fit(X_train_scaled, y_train)
    rf_pred = balanced_rf.predict(X_test_scaled)
    techniques['Balanced RF'] = rf_pred
    
    # 2. Ensemble with focus on spoiling class
    print("\n2. ENSEMBLE METHODS:")
    print("-" * 40)
    
    from sklearn.ensemble import VotingClassifier
    
    # Create ensemble focused on minority class
    ensemble = VotingClassifier([
        ('rf_balanced', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('svm_balanced', SVC(class_weight='balanced', probability=True, random_state=42)),
        ('gb_balanced', GradientBoostingClassifier(random_state=42))
    ], voting='soft')
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    techniques['Ensemble'] = ensemble_pred
    
    # 3. Feature engineering
    print("\n3. FEATURE ENGINEERING:")
    print("-" * 40)
    
    # Add derived features
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # Ratios and differences
    X_train_enhanced['MQ3_Ratio'] = X_train['MQ3_Top_PPM'] / (X_train['MQ3_Bottom_PPM'] + 1e-6)
    X_test_enhanced['MQ3_Ratio'] = X_test['MQ3_Top_PPM'] / (X_test['MQ3_Bottom_PPM'] + 1e-6)
    
    X_train_enhanced['MQ3_Diff'] = X_train['MQ3_Top_PPM'] - X_train['MQ3_Bottom_PPM']
    X_test_enhanced['MQ3_Diff'] = X_test['MQ3_Top_PPM'] - X_test['MQ3_Bottom_PPM']
    
    # Temperature-humidity interaction
    X_train_enhanced['Temp_Humid_Interaction'] = X_train['BME_Temp'] * X_train['BME_Humidity']
    X_test_enhanced['Temp_Humid_Interaction'] = X_test['BME_Temp'] * X_test['BME_Humidity']
    
    # Scale enhanced features
    scaler_enhanced = StandardScaler()
    X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
    X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)
    
    # Train with enhanced features
    rf_enhanced = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_enhanced.fit(X_train_enhanced_scaled, y_train)
    enhanced_pred = rf_enhanced.predict(X_test_enhanced_scaled)
    techniques['Enhanced Features'] = enhanced_pred
    
    # Evaluate all techniques
    print("\n4. RESULTS COMPARISON:")
    print("-" * 40)
    print(f"{'Method':<20} | {'Spoiling Precision':<18} | {'Spoiling Recall':<15} | {'Spoiling F1':<12}")
    print("-" * 70)
    
    best_method = None
    best_f1 = 0
    
    for method_name, predictions in techniques.items():
        try:
            from sklearn.metrics import classification_report
            report = classification_report(y_test, predictions, output_dict=True)
            
            if '1' in report:  # Check if spoiling class exists
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
                
                print(f"{method_name:<20} | {precision:<18.3f} | {recall:<15.3f} | {f1:<12.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = method_name
                    
        except Exception as e:
            print(f"{method_name:<20} | Error: {str(e)}")
    
    if best_method:
        print(f"\n🏆 BEST METHOD FOR SPOILING CLASS: {best_method}")
        print(f"   Best Spoiling F1-score: {best_f1:.3f}")
    
    return techniques, best_method

def optimize_spoiling_thresholds_advanced(df, verbose=True):
    """
    Advanced optimization of time-based thresholds to maximize spoiling class detection.
    Uses MQ3 Bottom sensor data and multiple criteria.
    """
    if verbose:
        print("🔧 ADVANCED SPOILING THRESHOLD OPTIMIZATION")
        print("=" * 50)
    
    # Calculate hours from start
    df = df.copy()
    df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    # Use MQ3 Bottom as primary sensor
    mq3_data = df['MQ3_Bottom_PPM'].dropna()
    hours_data = df.loc[mq3_data.index, 'Hours_from_start']
    
    if len(mq3_data) < 20:
        if verbose:
            print("⚠️  Insufficient MQ3 Bottom data for optimization")
        return None
    
    # Calculate various change indicators
    baseline = mq3_data.iloc[:min(10, len(mq3_data))].mean()
    cumulative_change = mq3_data - baseline
    rolling_std = mq3_data.rolling(window=10, center=True).std()
    gradient = np.gradient(mq3_data.values)
    
    # Find potential transition points
    threshold_candidates = []
    
    # Method 1: Significant rate of change
    high_gradient_points = np.where(np.abs(gradient) > np.percentile(np.abs(gradient), 75))[0]
    if len(high_gradient_points) > 0:
        threshold_candidates.extend(hours_data.iloc[high_gradient_points].tolist())
    
    # Method 2: Standard deviation peaks (instability)
    if not rolling_std.isna().all():
        std_peaks = rolling_std > np.percentile(rolling_std.dropna(), 80)
        threshold_candidates.extend(hours_data.loc[std_peaks].tolist())
    
    # Method 3: Cumulative change thresholds
    change_percentiles = [50, 60, 70, 80]
    for percentile in change_percentiles:
        threshold_val = np.percentile(cumulative_change, percentile)
        transition_idx = np.where(cumulative_change >= threshold_val)[0]
        if len(transition_idx) > 0:
            threshold_candidates.append(hours_data.iloc[transition_idx[0]])
    
    # Remove duplicates and sort
    threshold_candidates = sorted(list(set(threshold_candidates)))
    
    if verbose:
        print(f"📊 Generated {len(threshold_candidates)} threshold candidates")
    
    # Evaluate each threshold combination
    best_config = None
    best_score = 0
    max_hours = hours_data.max()
    
    for fresh_to_spoiling in threshold_candidates:
        for spoiling_to_spoiled in threshold_candidates:
            if spoiling_to_spoiled <= fresh_to_spoiling:
                continue
            if spoiling_to_spoiled >= max_hours - 5:  # Leave some time for spoiled class
                continue
                
            # Create labels with this configuration
            labels = np.zeros(len(df))
            labels[(df['Hours_from_start'] >= fresh_to_spoiling) & 
                   (df['Hours_from_start'] < spoiling_to_spoiled)] = 1  # Spoiling
            labels[df['Hours_from_start'] >= spoiling_to_spoiled] = 2   # Spoiled
            
            # Calculate quality metrics
            unique, counts = np.unique(labels, return_counts=True)
            
            if len(unique) == 3:  # All three classes present
                spoiling_ratio = counts[1] / len(labels)
                
                # Prefer configurations with 10-30% spoiling class
                if 0.05 <= spoiling_ratio <= 0.35:
                    # Score based on balanced representation and reasonable transition timing
                    balance_score = 1 - abs(spoiling_ratio - 0.15)  # Target ~15% spoiling
                    timing_score = 1 - abs((fresh_to_spoiling / max_hours) - 0.6)  # Target ~60% through
                    
                    total_score = balance_score * 0.7 + timing_score * 0.3
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_config = {
                            'fresh_to_spoiling': fresh_to_spoiling,
                            'spoiling_to_spoiled': spoiling_to_spoiled,
                            'spoiling_ratio': spoiling_ratio,
                            'score': total_score
                        }
    
    if best_config and verbose:
        print(f"🎯 OPTIMAL THRESHOLDS FOUND:")
        print(f"   Fresh → Spoiling: {best_config['fresh_to_spoiling']:.1f}h")
        print(f"   Spoiling → Spoiled: {best_config['spoiling_to_spoiled']:.1f}h")
        print(f"   Spoiling class ratio: {best_config['spoiling_ratio']:.1%}")
        print(f"   Quality score: {best_config['score']:.3f}")
    
    return best_config

def create_optimized_time_labels(df, verbose=True):
    """
    Create time-based labels using optimized thresholds for better spoiling class detection.
    """
    optimal_config = optimize_spoiling_thresholds_advanced(df, verbose)
    
    if optimal_config is None:
        if verbose:
            print("⚠️  Using fallback threshold method")
        return create_time_based_labels(df, verbose)
    
    # Calculate hours from start
    df = df.copy()
    df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    # Create optimized labels
    labels = np.zeros(len(df))
    labels[(df['Hours_from_start'] >= optimal_config['fresh_to_spoiling']) & 
           (df['Hours_from_start'] < optimal_config['spoiling_to_spoiled'])] = 1  # Spoiling
    labels[df['Hours_from_start'] >= optimal_config['spoiling_to_spoiled']] = 2   # Spoiled
    
    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n🏷️  OPTIMIZED LABEL DISTRIBUTION:")
        for label, count in zip(unique, counts):
            class_name = ['Fresh', 'Spoiling', 'Spoiled'][int(label)]
            print(f"   {class_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels

def advanced_spoiling_detection_pipeline(X, y, test_size=0.3, verbose=True):
    """
    Complete pipeline for advanced spoiling class detection using multiple techniques.
    """
    if verbose:
        print("🚀 ADVANCED SPOILING DETECTION PIPELINE")
        print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    import seaborn as sns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    if verbose:
        print(f"📊 Training Set Class Distribution:")
        for label, count in zip(unique, counts):
            class_name = ['Fresh', 'Spoiling', 'Spoiled'][int(label)]
            print(f"   {class_name}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Enhanced feature engineering
    X_train_enhanced = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_enhanced = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Add interaction features
    if 'MQ3_Top_PPM' in X.columns and 'MQ3_Bottom_PPM' in X.columns:
        X_train_enhanced['MQ3_Ratio'] = X_train_enhanced['MQ3_Top_PPM'] / (X_train_enhanced['MQ3_Bottom_PPM'] + 1e-6)
        X_test_enhanced['MQ3_Ratio'] = X_test_enhanced['MQ3_Top_PPM'] / (X_test_enhanced['MQ3_Bottom_PPM'] + 1e-6)
        
        X_train_enhanced['MQ3_Diff'] = X_train_enhanced['MQ3_Top_PPM'] - X_train_enhanced['MQ3_Bottom_PPM']
        X_test_enhanced['MQ3_Diff'] = X_test_enhanced['MQ3_Top_PPM'] - X_test_enhanced['MQ3_Bottom_PPM']
    
    if 'BME_Temp' in X.columns and 'BME_Humidity' in X.columns:
        X_train_enhanced['Temp_Humid_Interaction'] = X_train_enhanced['BME_Temp'] * X_train_enhanced['BME_Humidity']
        X_test_enhanced['Temp_Humid_Interaction'] = X_test_enhanced['BME_Temp'] * X_test_enhanced['BME_Humidity']
    
    models = {}
    results = {}
    
    # 1. Balanced Random Forest
    if verbose:
        print(f"\n1️⃣  Training Balanced Random Forest...")
    
    rf_balanced = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_balanced.fit(X_train_enhanced, y_train)
    rf_pred = rf_balanced.predict(X_test_enhanced)
    models['Balanced RF'] = rf_balanced
    results['Balanced RF'] = rf_pred
    
    # 2. Gradient Boosting with class weights
    if verbose:
        print(f"2️⃣  Training Gradient Boosting...")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[int(label)] for label in y_train])
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    gb_model.fit(X_train_enhanced, y_train, sample_weight=sample_weights)
    gb_pred = gb_model.predict(X_test_enhanced)
    models['Weighted GB'] = gb_model
    results['Weighted GB'] = gb_pred
    
    # 3. Ensemble Voting Classifier
    if verbose:
        print(f"3️⃣  Training Ensemble Classifier...")
    
    ensemble = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('svm', SVC(class_weight='balanced', probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ], voting='soft')
    
    ensemble.fit(X_train_enhanced, y_train)
    ensemble_pred = ensemble.predict(X_test_enhanced)
    models['Ensemble'] = ensemble
    results['Ensemble'] = ensemble_pred
    
    # Evaluate all models
    if verbose:
        print(f"\n📈 MODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<15} | {'Spoiling Precision':<18} | {'Spoiling Recall':<15} | {'Spoiling F1':<12} | {'Overall Accuracy':<16}")
        print("-" * 80)
    
    best_model = None
    best_spoiling_f1 = 0
    
    for model_name, predictions in results.items():
        try:
            report = classification_report(y_test, predictions, output_dict=True)
            overall_accuracy = (predictions == y_test).mean()
            
            if '1' in report:  # Spoiling class exists
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
                
                if verbose:
                    print(f"{model_name:<15} | {precision:<18.3f} | {recall:<15.3f} | {f1:<12.3f} | {overall_accuracy:<16.3f}")
                
                if f1 > best_spoiling_f1:
                    best_spoiling_f1 = f1
                    best_model = model_name
            else:
                if verbose:
                    print(f"{model_name:<15} | {'No spoiling class':<47} | {overall_accuracy:<16.3f}")
                    
        except Exception as e:
            if verbose:
                print(f"{model_name:<15} | Error: {str(e)}")
    
    if best_model and verbose:
        print(f"\n🏆 BEST MODEL FOR SPOILING DETECTION: {best_model}")
        print(f"   Spoiling F1-score: {best_spoiling_f1:.3f}")
        
        # Show detailed confusion matrix for best model
        print(f"\n📊 CONFUSION MATRIX FOR {best_model}:")
        cm = confusion_matrix(y_test, results[best_model])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fresh', 'Spoiling', 'Spoiled'],
                   yticklabels=['Fresh', 'Spoiling', 'Spoiled'])
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    return {
        'models': models,
        'results': results,
        'best_model': best_model,
        'best_spoiling_f1': best_spoiling_f1,
        'scaler': scaler,
        'y_test': y_test
    }

def main():
    """Main analysis function"""
    
    print("Loading data...")
    train_df, test_df = load_data()
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
      # Define hand-written spoilage certainty ranges (corrected to 2025)
    # Batch 1 and 2 end times adjusted to when MQ3 data cuts out
    spoilage_ranges = {
        'batch1': {
            'start': '2025-06-15 20:00:00',
            'end': '2025-06-16 10:21:00'  # Adjusted to when MQ3 data actually cuts out
        },        'batch2': {
            'start': '2025-06-19 20:00:00', 
            'end': '2025-06-20 05:40:00'  # Adjusted to when MQ3 data actually cuts out
        }
    }
    
    # Visualize MQ3 data over time for both batches with spoilage ranges
    print("\n" + "="*60)
    print("VISUAL ANALYSIS OF SENSOR DATA OVER TIME")
    print("="*60)
    
    train_df_enhanced = visualize_mq3_over_time(train_df, "Batch 1 (Training)", 
                                              spoilage_ranges['batch1'])
    test_df_enhanced = visualize_mq3_over_time(test_df, "Batch 2 (Test)", 
                                             spoilage_ranges['batch2'])
    
    # Determine optimal labeling method
    optimal_method, labeling_results = determine_optimal_labeling_method(train_df, test_df)
    
    # Advanced spoiling class optimization
    print("\n" + "="*60)
    print("ADVANCED SPOILING CLASS OPTIMIZATION")
    print("="*60)
    
    # Try optimized time-based labeling for better spoiling detection
    print("\n🔧 Optimizing time-based labeling thresholds...")
    optimized_train_labels = create_optimized_time_labels(train_df, verbose=True)
    optimized_test_labels = create_optimized_time_labels(test_df, verbose=True)
    
    # Prepare features for advanced pipeline
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    # Get complete data (no NaN values)
    train_complete = train_df[feature_cols].dropna()
    
    if len(train_complete) > 100:  # Need sufficient data
        print(f"\n🚀 Running advanced spoiling detection pipeline...")
        print(f"   Complete training samples: {len(train_complete)}")
        
        # Use optimized labels for complete data indices
        train_labels_complete = optimized_train_labels[:len(train_complete)]
        
        # Run advanced pipeline
        pipeline_results = advanced_spoiling_detection_pipeline(
            train_complete, train_labels_complete, 
            test_size=0.3, verbose=True
        )
        
        print(f"\n✅ Advanced pipeline completed!")
        if pipeline_results['best_model']:
            print(f"   Best model: {pipeline_results['best_model']}")
            print(f"   Best spoiling F1-score: {pipeline_results['best_spoiling_f1']:.3f}")
    else:
        print(f"⚠️  Insufficient complete data for advanced pipeline")
        print(f"   Training: {len(train_complete)} samples (need >100)")
      # Compare labeling methods
    print("\nComparing labeling methods...")
    train_kmeans_labels, train_silhouette = compare_labeling_methods(train_df_enhanced, "Batch 1")
    test_kmeans_labels, test_silhouette = compare_labeling_methods(test_df_enhanced, "Batch 2")
    
    # Compare labeling methods with supervised models
    print("\nComparing labeling methods with supervised models...")
    labeling_comparison, recommended_method = compare_labeling_methods_with_models(train_df, test_df)
    
    # Refine labeling for better spoiling class detection
    print("\nRefining labeling for better spoiling class detection...")
    refined_result, refined_method = refine_labeling_for_spoiling_class(train_df, test_df)
    
    # Interactive spoilage detection
    print("\nCreating interactive spoilage detection plots...")
    train_df_final = create_interactive_spoilage_detection(train_df_enhanced, "Batch 1")
    test_df_final = create_interactive_spoilage_detection(test_df_enhanced, "Batch 2")
    
    # Use the recommended method for final model evaluation
    final_method = recommended_method if recommended_method != "hybrid" else optimal_method
    print(f"\n🎯 Using {final_method.upper()} for final model evaluation...")
    
    # Try advanced classification techniques
    print("\nTrying advanced classification techniques...")
    advanced_techniques, best_advanced = try_advanced_classification_techniques(train_df, test_df, final_method)
    
    # Evaluate supervised learning models
    model_results, trained_models, scaler, data_splits = evaluate_supervised_models(
        train_df, test_df, final_method)
    
    if model_results:
        # Visualize model performance
        visualize_model_performance(model_results)
        
        # Visualize model predictions on time series
        visualize_model_predictions_on_time_series(train_df, test_df, model_results, spoilage_ranges)
      # Suggest optimal strategy
    suggest_optimal_labeling_strategy(train_df, test_df)
    
    return train_df_final, test_df_final, model_results, labeling_comparison

if __name__ == "__main__":
    train_enhanced, test_enhanced, model_results, labeling_comparison = main()
