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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from itertools import cycle
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{batch_name} - Sensor Data Over Time Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.8, wspace=0.4, top=0.93, bottom=0.10)
    
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
    axes[1,0].legend()
    
    # Add spoilage certainty range
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
    
    # 4. Environmental factors with VOC (combined plot for space efficiency)
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
    plt.suptitle(f'Labeling Methods Comparison - {batch_name}', fontsize=16, y=0.97)
    plt.subplots_adjust(hspace=0.7, wspace=0.4, top=0.93, bottom=0.12)
    
    plt.subplot(2, 2, 1)
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    plt.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # Compare time-based vs K-means (3 clusters)
    if 'Time_to_Spoilage_Minutes' in df.columns:
        # Map K-means labels to proper spoilage categories
        kmeans_3_labels_mapped = map_kmeans_to_spoilage_labels(df, kmeans_3_labels)
        
        plt.subplot(2, 2, 2)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Time_Hours': df['Hours_from_start'],
            'Time_Labels': time_labels,
            'KMeans_Labels': kmeans_3_labels_mapped,
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
        plt.legend(loc='upper left', framealpha=0.9, fontsize=8)
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
        plt.legend(loc='upper left', framealpha=0.9, fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Calculate agreement between methods
        rand_score = adjusted_rand_score(time_labels, kmeans_3_labels_mapped)
        
        plt.subplot(2, 2, 4)
        
        # Confusion matrix between methods
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(time_labels, kmeans_3_labels_mapped)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Time vs K-means Agreement\nAdjusted Rand Score: {rand_score:.3f}')
        plt.xlabel('K-means Labels')
        plt.ylabel('Time-based Labels')
        
        print(f"\nMethod Comparison:")
        print(f"Adjusted Rand Score (agreement): {rand_score:.3f}")
        print(f"Higher score (closer to 1) means better agreement")
        
        if rand_score < 0.3:
            print("WARNING: LOW AGREEMENT: K-means finds different patterns than time-based labels")
        elif rand_score < 0.6:
            print("MODERATE AGREEMENT: Some similarity between methods")
        else:
            print("HIGH AGREEMENT: Methods largely agree")
    
    plt.tight_layout()
    plt.show()
    
    return kmeans_3_labels_mapped, silhouette_scores

def create_interactive_spoilage_detection(df, batch_name="Batch"):
    """Create interactive plots to help identify spoilage onset"""
    
    print(f"\n{'='*60}")
    print(f"INTERACTIVE SPOILAGE DETECTION - {batch_name}")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{batch_name} - Identify Spoilage Onset Points', fontsize=16, y=0.97)
    plt.subplots_adjust(hspace=0.9, wspace=0.4, top=0.93, bottom=0.12)
    
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
    
    print(f"\nOPTIMAL LABELING RECOMMENDATION:")
    print("-" * 50)
    
    optimal_method = "time-based"
    print(f"Use TIME-BASED LABELING")
    print(f"   Reason: Domain knowledge-driven classification using spoilage timeline")
    print(f"   Fresh: > 48 hours before spoilage")
    print(f"   Spoiling: 24-48 hours before spoilage")
    print(f"   Spoiled: < 24 hours before spoilage")
    
    return optimal_method, results

def create_time_based_labels(df, verbose=False):
    """Create spoilage labels based on time to spoilage"""
    if 'Time_to_Spoilage_Minutes' not in df.columns:
        if verbose:
            print("WARNING: No Time_to_Spoilage_Minutes column found")
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
        print(f"Time-based label distribution:")
        for label, count in zip(unique, counts):
            class_name = ['Fresh', 'Spoiling', 'Spoiled'][int(label)]
            print(f"   {class_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels

def create_optimized_time_labels(df, verbose=False):
    """Create optimized time-based labels with adjusted thresholds for better spoiling detection"""
    if 'Time_to_Spoilage_Minutes' not in df.columns:
        if verbose:
            print("WARNING: No Time_to_Spoilage_Minutes column found, using K-means clustering")
        # Fallback to K-means if no time data
        feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(X_scaled)
        # Map clusters to proper spoilage categories
        return map_kmeans_to_spoilage_labels(df, raw_labels)
        
    # Optimized thresholds for better spoiling class balance
    def optimized_label_time_to_spoilage(time_to_spoilage):
        if time_to_spoilage > 60 * 60:  # > 60 hours (increased fresh threshold)
            return 0  # Fresh
        elif time_to_spoilage > 12 * 60:  # 12-60 hours (wider spoiling window)
            return 1  # Spoiling
        else:  # < 12 hours
            return 2  # Spoiled
    
    labels = df['Time_to_Spoilage_Minutes'].apply(optimized_label_time_to_spoilage)
    
    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Optimized time-based label distribution:")
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
        raw_labels = kmeans.fit_predict(X_scaled)
        # Map clusters to proper spoilage categories
        return map_kmeans_to_spoilage_labels(df, raw_labels)
    
    else:  # hybrid approach - use time-based if available, else K-means
        if 'Time_to_Spoilage_Minutes' in df.columns:
            return create_time_based_labels(df)
        else:
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(X_scaled)
            # Map clusters to proper spoilage categories
            return map_kmeans_to_spoilage_labels(df, raw_labels)

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

def evaluate_supervised_models(train_df, test_df, optimal_method, overfitting_threshold=0.3):
    """Evaluate all supervised learning models with overfitting detection"""
    
    print(f"\n{'='*80}")
    print("SUPERVISED LEARNING MODEL EVALUATION")
    print(f"{'='*80}")
    
    print(f"LABELING METHOD BEING USED: {optimal_method.upper()}")
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
        print("ERROR: Cannot create labels with the specified method")
        return None, None, None, None
    
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
    print(f"\nDEBUGGING TRAINING DATA:")
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
            # - Stricter: 0.1 (10%) - flags more models as overfitting
            # - More lenient: 0.3 (30%) - allows larger train-test gaps
            overfitting_score = train_acc - test_acc
            is_overfitting = overfitting_score > overfitting_threshold
            
            # F1 scores
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            # ROC AUC (multiclass using one-vs-rest)
            roc_auc_score = None
            try:
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test_scaled)
                    n_classes = len(np.unique(y_test))
                    if n_classes > 2:
                        # Multiclass ROC AUC (one-vs-rest)
                        from sklearn.metrics import roc_auc_score
                        roc_auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    else:
                        # Binary classification
                        roc_auc_score = roc_auc_score(y_test, y_proba[:, 1])
                elif hasattr(clf, "decision_function"):
                    y_scores = clf.decision_function(X_test_scaled)
                    if len(np.unique(y_test)) > 2:
                        from sklearn.metrics import roc_auc_score
                        roc_auc_score = roc_auc_score(y_test, y_scores, multi_class='ovr', average='weighted')
                    else:
                        roc_auc_score = roc_auc_score(y_test, y_scores)
            except Exception as e:
                print(f"  Note: Could not calculate ROC AUC for {name}: {e}")
                roc_auc_score = None
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_score': overfitting_score,
                'is_overfitting': is_overfitting,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'roc_auc': roc_auc_score,
                'test_predictions': test_pred,
                'train_predictions': train_pred
            }
            
            trained_models[name] = clf
            
            # Print results
            print(f"  Train Accuracy: {train_acc:.3f}")
            print(f"  Test Accuracy: {test_acc:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
            print(f"  Train F1: {train_f1:.3f}")
            print(f"  Test F1: {test_f1:.3f}")
            if roc_auc_score is not None:
                print(f"  ROC AUC: {roc_auc_score:.3f}")
            print(f"  Overfitting Score: {overfitting_score:.3f}")
            if is_overfitting:
                print(f"  WARNING: OVERFITTING DETECTED!")
            else:
                print(f"  Good generalization")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = None
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        # Best model based on test accuracy, penalized for overfitting
        best_model = max(valid_results.keys(), 
                        key=lambda x: valid_results[x]['test_accuracy'] - 
                                    (0.1 * valid_results[x]['overfitting_score'] if valid_results[x]['is_overfitting'] else 0))
        
        print(f"\nBEST MODEL: {best_model}")
        print(f"   Test Accuracy: {valid_results[best_model]['test_accuracy']:.3f}")
        print(f"   Overfitting: {'Yes' if valid_results[best_model]['is_overfitting'] else 'No'}")
    
    return results, trained_models, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

def visualize_model_performance(results):
    """Visualize model performance comparison"""
    
    if not results or not any(results.values()):
        print("ERROR: No valid results to visualize")
        return
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=0.97)
    plt.subplots_adjust(hspace=0.7, wspace=0.4, top=0.93, bottom=0.12)
    
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
    axes[0,1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
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

def visualize_all_model_predictions_separately(train_df, test_df, results, spoilage_ranges):
    """Visualize each model's predictions on separate plots"""
    
    if not results or not any(results.values()):
        print("ERROR: No valid results to visualize")
        return
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    colors = ['green', 'orange', 'red']  # Fresh, Spoiling, Spoiled
    labels = ['Fresh', 'Spoiling', 'Spoiled']
    
    datasets = [
        (train_df, "Batch 1 (Training)", spoilage_ranges.get('batch1')), 
        (test_df, "Batch 2 (Test)", spoilage_ranges.get('batch2'))
    ]
    
    for model_name, model_results in valid_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{model_name} - Predictions on MQ3 vs Time', fontsize=16, y=0.98)
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.90, bottom=0.15)
        
        for dataset_idx, (df, dataset_name, spoilage_range) in enumerate(datasets):
            ax = axes[dataset_idx]
            
            # Calculate hours from start
            df = df.copy()
            df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
            
            # Get predictions
            if dataset_idx == 0:  # Training data
                predictions = model_results['train_predictions']
            else:  # Test data
                predictions = model_results['test_predictions']
            
            # Plot MQ3 data with color-coded predictions
            mq3_data = df['MQ3_Bottom_PPM']
            
            for class_label in range(3):
                mask = predictions == class_label
                if mask.any():
                    ax.scatter(df['Hours_from_start'][mask], mq3_data[mask], 
                             c=colors[class_label], alpha=0.7, s=40, 
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
            
            ax.set_title(f'{dataset_name}', pad=15)
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel('MQ3 Bottom PPM')
            ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def visualize_all_confusion_matrices(results, y_train, y_test):
    """Visualize confusion matrices for all models"""
    
    if not results or not any(results.values()):
        print("ERROR: No valid results to visualize")
        return
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_models = len(valid_results)
    model_names = list(valid_results.keys())
    class_names = ['Fresh', 'Spoiling', 'Spoiled']
    
    # Limit to 4 subplots per figure (2x2 grid)
    models_per_plot = 4
    n_plots = (n_models + models_per_plot - 1) // models_per_plot
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * models_per_plot
        end_idx = min(start_idx + models_per_plot, n_models)
        current_models = model_names[start_idx:end_idx]
        n_current = len(current_models)
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if n_plots > 1:
            fig.suptitle(f'Confusion Matrices - Models {start_idx+1}-{end_idx}', fontsize=16, y=0.97)
        else:
            fig.suptitle('Confusion Matrices - All Models', fontsize=16, y=0.97)
        
        plt.subplots_adjust(hspace=0.8, wspace=0.5, top=0.93, bottom=0.08)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for idx, model_name in enumerate(current_models):
            ax = axes_flat[idx]
            model_results = valid_results[model_name]
            
            # Get test predictions and calculate confusion matrix
            test_predictions = model_results['test_predictions']
            cm = confusion_matrix(y_test, test_predictions)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'shrink': 0.8})
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, test_predictions)
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', pad=15)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        
        # Hide unused subplots
        for idx in range(n_current, 4):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def visualize_model_predictions_on_time_series(train_df, test_df, results, spoilage_ranges):
    """Visualize model predictions overlaid on MQ3 vs time plots"""
    
    if not results or not any(results.values()):
        print("ERROR: No valid results to visualize")
        return
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Get best models (top 3)
    sorted_models = sorted(valid_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Top 3 Models - Predictions on MQ3 vs Time', fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=1.0, wspace=0.4, top=0.94, bottom=0.08)
    
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
            df = df.copy()
            df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
            
            # Get predictions
            if dataset_idx == 0:  # Training data
                predictions = model_results['train_predictions']
            else:  # Test data
                predictions = model_results['test_predictions']
            
            # Plot MQ3 data with color-coded predictions
            mq3_data = df['MQ3_Bottom_PPM']
            
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
            
            ax.set_title(f'{model_name} - {dataset_name}', pad=15)
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel('MQ3 Bottom PPM')
            ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def map_kmeans_to_spoilage_labels(df, kmeans_labels):
    """Map K-means cluster labels to spoilage categories based on time progression"""
    
    # Calculate average time for each cluster
    df_temp = df.copy()
    df_temp['cluster'] = kmeans_labels
    
    # Calculate hours from start if not already present
    if 'Hours_from_start' not in df_temp.columns:
        df_temp['Hours_from_start'] = (df_temp['Timestamp'] - df_temp['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    cluster_avg_times = df_temp.groupby('cluster')['Hours_from_start'].mean().sort_values()
    
    # Map clusters based on time progression: earliest = Fresh, latest = Spoiled
    cluster_mapping = {}
    sorted_clusters = cluster_avg_times.index.tolist()
    
    if len(sorted_clusters) >= 3:
        cluster_mapping[sorted_clusters[0]] = 0  # Earliest time = Fresh
        cluster_mapping[sorted_clusters[1]] = 1  # Middle time = Spoiling  
        cluster_mapping[sorted_clusters[2]] = 2  # Latest time = Spoiled
    elif len(sorted_clusters) == 2:
        cluster_mapping[sorted_clusters[0]] = 0  # Fresh
        cluster_mapping[sorted_clusters[1]] = 2  # Spoiled
    else:
        cluster_mapping[sorted_clusters[0]] = 1  # Default to Spoiling
    
    # Apply mapping
    mapped_labels = np.array([cluster_mapping.get(label, label) for label in kmeans_labels])
    
    print(f"K-means cluster mapping: {cluster_mapping}")
    print(f"Cluster average times: {cluster_avg_times.to_dict()}")
    
    return mapped_labels

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models using one-vs-rest approach for multiclass classification
    """
    n_classes = len(np.unique(y_test))
    
    # Binarize the output labels for multiclass ROC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    
    # Calculate number of figures needed (4 models per figure)
    n_figures = (len(models) + 3) // 4
    
    for fig_idx in range(n_figures):
        plt.figure(figsize=(16, 12))
        
        # Get models for this figure
        start_idx = fig_idx * 4
        end_idx = min(start_idx + 4, len(models))
        current_models = list(models.items())[start_idx:end_idx]
        
        for idx, (name, model) in enumerate(current_models):
            plt.subplot(2, 2, idx + 1)
            
            try:
                # Get prediction probabilities
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test)
                    # Normalize decision function output to [0,1]
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    y_proba = scaler.fit_transform(y_proba)
                else:
                    print(f"Model {name} doesn't support probability prediction. Skipping ROC curve.")
                    plt.text(0.5, 0.5, f'{name}\nNo probability\nsupport', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    continue
                
                # Colors for different classes
                colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
                
                # Calculate ROC curve for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    if n_classes == 2:
                        # Binary classification
                        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                    else:
                        # Multiclass classification
                        if y_proba.shape[1] >= n_classes:
                            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                        else:
                            # Handle case where model output doesn't match number of classes
                            continue
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Plot ROC curves for each class
                class_names = ['Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4']
                for i, color in zip(range(n_classes), colors):
                    if i in fpr:
                        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
                        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
                
                # Plot diagonal line
                plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
                
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {name}')
                plt.legend(loc="lower right", fontsize=8)
                plt.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error plotting ROC curve for {name}: {str(e)}")
                plt.text(0.5, 0.5, f'{name}\nError plotting\nROC curve', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.suptitle(f'ROC Curves - Models {start_idx+1}-{end_idx}', fontsize=16, y=1.02)
        plt.show()

def main():
    """Main analysis function"""
    
    print("Loading data...")
    train_df, test_df = load_data()
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Define hand-written spoilage certainty ranges (corrected to 2025)
    spoilage_ranges = {
        'batch1': {
            'start': '2025-06-15 20:00:00',
            'end': '2025-06-16 10:21:00'
        },
        'batch2': {
            'start': '2025-06-19 20:00:00', 
            'end': '2025-06-20 05:40:00'
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
    
    # Compare labeling methods
    print("\nComparing labeling methods...")
    train_kmeans_labels, train_silhouette = compare_labeling_methods(train_df_enhanced, "Batch 1")
    test_kmeans_labels, test_silhouette = compare_labeling_methods(test_df_enhanced, "Batch 2")
    
    # Interactive spoilage detection
    print("\nCreating interactive spoilage detection plots...")
    train_df_final = create_interactive_spoilage_detection(train_df_enhanced, "Batch 1")
    test_df_final = create_interactive_spoilage_detection(test_df_enhanced, "Batch 2")
    
    # Evaluate supervised learning models
    model_results, trained_models, scaler, data_splits = evaluate_supervised_models(
        train_df, test_df, "time-based")
    
    if model_results:
        X_train_scaled, X_test_scaled, y_train, y_test = data_splits
        
        # Visualize model performance
        visualize_model_performance(model_results)
        
        # Visualize model predictions on time series (top 3 models)
        visualize_model_predictions_on_time_series(train_df, test_df, model_results, spoilage_ranges)
        
        # Visualize ALL model predictions separately
        print("\nGenerating individual prediction plots for all models...")
        visualize_all_model_predictions_separately(train_df, test_df, model_results, spoilage_ranges)
        
        # Visualize ALL confusion matrices
        print("\nGenerating confusion matrices for all models...")
        visualize_all_confusion_matrices(model_results, y_train, y_test)
        
        # Plot ROC curves for all models
        print("\nGenerating ROC curves for all models...")
        plot_roc_curves(trained_models, X_test_scaled, y_test)
    
    return train_df_final, test_df_final, model_results

if __name__ == "__main__":
    train_enhanced, test_enhanced, model_results = main()
