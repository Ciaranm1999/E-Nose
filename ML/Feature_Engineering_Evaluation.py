import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare both datasets"""
    train_df = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
    test_df = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
    
    # Truncate at NaN
    def truncate_at_nan(df, feature_cols):
        nan_mask = df[feature_cols].isnull().any(axis=1)
        if nan_mask.any():
            first_nan_idx = nan_mask.idxmax()
            return df.loc[:first_nan_idx - 1].reset_index(drop=True)
        return df
    
    # Basic MQ3 features for NaN checking
    basic_features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'MQ3_Top_Analog', 'MQ3_Bottom_Analog']
    train_df = truncate_at_nan(train_df, basic_features)
    test_df = truncate_at_nan(test_df, basic_features)
    
    return train_df, test_df

def create_engineered_features(df, lag_values=[60, 120, 180]):
    """Create engineered features from raw sensor data"""
    df_eng = df.copy()
    
    # 1. Original sensor features
    sensor_features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'MQ3_Top_Analog', 'MQ3_Bottom_Analog', 
                      'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    # 2. Ratio features
    df_eng['MQ3_PPM_Ratio'] = df_eng['MQ3_Top_PPM'] / (df_eng['MQ3_Bottom_PPM'] + 1e-6)
    df_eng['MQ3_Analog_Ratio'] = df_eng['MQ3_Top_Analog'] / (df_eng['MQ3_Bottom_Analog'] + 1e-6)
    df_eng['PPM_to_Analog_Top'] = df_eng['MQ3_Top_PPM'] / (df_eng['MQ3_Top_Analog'] + 1e-6)
    df_eng['PPM_to_Analog_Bottom'] = df_eng['MQ3_Bottom_PPM'] / (df_eng['MQ3_Bottom_Analog'] + 1e-6)
    
    # 3. Difference features
    df_eng['MQ3_PPM_Diff'] = df_eng['MQ3_Top_PPM'] - df_eng['MQ3_Bottom_PPM']
    df_eng['MQ3_Analog_Diff'] = df_eng['MQ3_Top_Analog'] - df_eng['MQ3_Bottom_Analog']
    
    # 4. Sum features
    df_eng['MQ3_PPM_Sum'] = df_eng['MQ3_Top_PPM'] + df_eng['MQ3_Bottom_PPM']
    df_eng['MQ3_Analog_Sum'] = df_eng['MQ3_Top_Analog'] + df_eng['MQ3_Bottom_Analog']
    
    # 5. Moving averages (rolling windows)
    for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']:
        if col in df_eng.columns:
            df_eng[f'{col}_MA_5'] = df_eng[col].rolling(window=5, min_periods=1).mean()
            df_eng[f'{col}_MA_10'] = df_eng[col].rolling(window=10, min_periods=1).mean()
            df_eng[f'{col}_MA_30'] = df_eng[col].rolling(window=30, min_periods=1).mean()
            
            # Standard deviation features
            df_eng[f'{col}_STD_5'] = df_eng[col].rolling(window=5, min_periods=1).std().fillna(0)
            df_eng[f'{col}_STD_10'] = df_eng[col].rolling(window=10, min_periods=1).std().fillna(0)
    
    # 6. Lag features
    for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']:
        if col in df_eng.columns:
            for lag in lag_values:
                if lag < len(df_eng):
                    df_eng[f'{col}_lag_{lag}'] = df_eng[col].shift(lag).fillna(method='bfill')
    
    # 7. Rate of change features (derivatives)
    for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity']:
        if col in df_eng.columns:
            df_eng[f'{col}_diff'] = df_eng[col].diff().fillna(0)
            df_eng[f'{col}_pct_change'] = df_eng[col].pct_change().fillna(0)
    
    # 8. Interaction features
    if 'BME_Temp' in df_eng.columns and 'BME_Humidity' in df_eng.columns:
        df_eng['Temp_Humidity_Product'] = df_eng['BME_Temp'] * df_eng['BME_Humidity']
        df_eng['Temp_Humidity_Ratio'] = df_eng['BME_Temp'] / (df_eng['BME_Humidity'] + 1e-6)
    
    # 9. Cumulative features
    for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']:
        if col in df_eng.columns:
            df_eng[f'{col}_cumsum'] = df_eng[col].cumsum()
            df_eng[f'{col}_cummax'] = df_eng[col].cummax()
            df_eng[f'{col}_cummin'] = df_eng[col].cummin()
    
    # 10. Statistical features over time windows
    for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']:
        if col in df_eng.columns:
            df_eng[f'{col}_max_5'] = df_eng[col].rolling(window=5, min_periods=1).max()
            df_eng[f'{col}_min_5'] = df_eng[col].rolling(window=5, min_periods=1).min()
            df_eng[f'{col}_range_5'] = df_eng[f'{col}_max_5'] - df_eng[f'{col}_min_5']
    
    return df_eng

def create_labels(df):
    """Create labels using K-means clustering"""
    # Features for clustering
    feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) == 0:
        # Fallback to basic MQ3 features
        available_cols = [col for col in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM'] if col in df.columns]
    
    X_cluster = df[available_cols].fillna(df[available_cols].mean())
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Try different numbers of clusters and select best
    best_score = -1
    best_k = 3
    best_labels = None
    
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        if len(set(labels)) > 1:  # More than one cluster
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    
    print(f"Best clustering: {best_k} clusters with silhouette score: {best_score:.3f}")
    
    return best_labels, best_k, best_score

def evaluate_feature_importance(df, labels):
    """Evaluate feature importance using multiple methods"""
    # Get all engineered features
    feature_cols = [col for col in df.columns if col not in ['Timestamp', 'labels']]
    X = df[feature_cols].fillna(df[feature_cols].mean())
    
    # Remove constant or near-constant features
    feature_variances = X.var()
    variable_features = feature_variances[feature_variances > 1e-6].index.tolist()
    X = X[variable_features]
    
    print(f"Using {len(variable_features)} variable features out of {len(feature_cols)} total features")
    
    # 1. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, labels)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # 2. Univariate Statistical Tests
    selector_f = SelectKBest(score_func=f_classif, k='all')
    selector_f.fit(X, labels)
    f_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    # 3. Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
    selector_mi.fit(X, labels)
    mi_scores = pd.DataFrame({
        'feature': X.columns,
        'mi_score': selector_mi.scores_
    }).sort_values('mi_score', ascending=False)
    
    # Combine results
    importance_df = rf_importance.merge(f_scores, on='feature').merge(mi_scores, on='feature')
    
    # Calculate composite score (normalized sum)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    importance_df[['rf_norm', 'f_norm', 'mi_norm']] = scaler.fit_transform(
        importance_df[['rf_importance', 'f_score', 'mi_score']]
    )
    importance_df['composite_score'] = (
        importance_df['rf_norm'] + importance_df['f_norm'] + importance_df['mi_norm']
    ) / 3
    importance_df = importance_df.sort_values('composite_score', ascending=False)
    
    return importance_df

def visualize_feature_importance(importance_df, top_n=20):
    """Visualize feature importance results"""
    top_features = importance_df.head(top_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analysis', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. Random Forest Importance
    axes[0,0].barh(range(len(top_features)), top_features['rf_importance'])
    axes[0,0].set_yticks(range(len(top_features)))
    axes[0,0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,0].set_title('Random Forest Importance')
    axes[0,0].set_xlabel('Importance Score')
    axes[0,0].invert_yaxis()
    
    # 2. F-Score
    axes[0,1].barh(range(len(top_features)), top_features['f_score'])
    axes[0,1].set_yticks(range(len(top_features)))
    axes[0,1].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,1].set_title('F-Score (ANOVA)')
    axes[0,1].set_xlabel('F-Score')
    axes[0,1].invert_yaxis()
    
    # 3. Mutual Information
    axes[1,0].barh(range(len(top_features)), top_features['mi_score'])
    axes[1,0].set_yticks(range(len(top_features)))
    axes[1,0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[1,0].set_title('Mutual Information Score')
    axes[1,0].set_xlabel('MI Score')
    axes[1,0].invert_yaxis()
    
    # 4. Composite Score
    axes[1,1].barh(range(len(top_features)), top_features['composite_score'])
    axes[1,1].set_yticks(range(len(top_features)))
    axes[1,1].set_yticklabels(top_features['feature'], fontsize=8)
    axes[1,1].set_title('Composite Score (Combined)')
    axes[1,1].set_xlabel('Composite Score')
    axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def select_best_features(importance_df, threshold_percentile=80):
    """Select the best features based on composite score"""
    threshold = np.percentile(importance_df['composite_score'], threshold_percentile)
    best_features = importance_df[importance_df['composite_score'] >= threshold]['feature'].tolist()
    
    print(f"\nSelected {len(best_features)} features above {threshold_percentile}th percentile:")
    for i, feature in enumerate(best_features, 1):
        score = importance_df[importance_df['feature'] == feature]['composite_score'].iloc[0]
        print(f"{i:2d}. {feature:<25} (score: {score:.3f})")
    
    return best_features

def main():
    """Main execution function"""
    print("="*80)
    print("FEATURE ENGINEERING AND EVALUATION")
    print("="*80)
    
    # Load data
    print("\n1. Loading and preparing data...")
    train_df, test_df = load_and_prepare_data()
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Engineer features for both datasets
    print("\n2. Engineering features...")
    train_eng = create_engineered_features(train_df)
    test_eng = create_engineered_features(test_df)
    print(f"Engineered features - Train: {train_eng.shape}, Test: {test_eng.shape}")
    
    # Create labels using clustering
    print("\n3. Creating labels using K-means clustering...")
    train_labels, best_k, score = create_labels(train_eng)
    train_eng['labels'] = train_labels
    
    print(f"Label distribution in training data:")
    for i in range(best_k):
        count = (train_labels == i).sum()
        print(f"  Cluster {i}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    # Evaluate feature importance
    print("\n4. Evaluating feature importance...")
    importance_df = evaluate_feature_importance(train_eng, train_labels)
    
    # Visualize results
    print("\n5. Visualizing feature importance...")
    visualize_feature_importance(importance_df)
    
    # Select best features
    print("\n6. Selecting best features...")
    best_features = select_best_features(importance_df, threshold_percentile=80)
    
    # Show top 10 features with their types
    print(f"\nTop 10 Features by Composite Score:")
    print("-" * 60)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        feature_type = "Original" if row['feature'] in ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm'] else "Engineered"
        print(f"{i:2d}. {row['feature']:<25} ({feature_type:<10}) Score: {row['composite_score']:.3f}")
    
    # Clustering quality assessment
    print(f"\n7. Clustering Quality Assessment:")
    print(f"   - Silhouette Score: {score:.3f}")
    
    feature_cols = [col for col in train_eng.columns if col not in ['Timestamp', 'labels']]
    X_cluster = train_eng[feature_cols].fillna(train_eng[feature_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    db_score = davies_bouldin_score(X_scaled, train_labels)
    print(f"   - Davies-Bouldin Score: {db_score:.3f} (lower is better)")
    
    print(f"\n8. Feature Engineering Summary:")
    print(f"   - Original features: {len([col for col in train_df.columns if col != 'Timestamp'])}")
    print(f"   - Engineered features: {len([col for col in train_eng.columns if col not in train_df.columns and col != 'labels'])}")
    print(f"   - Total features: {len(feature_cols)}")
    print(f"   - Selected best features: {len(best_features)}")
    print(f"   - Feature reduction: {(1 - len(best_features)/len(feature_cols))*100:.1f}%")
    
    return train_eng, test_eng, importance_df, best_features

if __name__ == "__main__":
    train_data, test_data, feature_importance, selected_features = main()
