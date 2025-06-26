import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for model visualization"""
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
    
    features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    train_df = truncate_at_nan(train_df, features)
    test_df = truncate_at_nan(test_df, features)
    
    # Add time features
    train_df['Hours_from_start'] = (train_df['Timestamp'] - train_df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    test_df['Hours_from_start'] = (test_df['Timestamp'] - test_df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    return train_df, test_df, features

def create_time_based_labels(df):
    """Create spoilage labels based on time to spoilage"""
    def label_time_to_spoilage(time_to_spoilage):
        if time_to_spoilage > 48 * 60:  # > 48 hours
            return 0  # Fresh
        elif time_to_spoilage > 24 * 60:  # 24-48 hours
            return 1  # Spoiling
        else:  # < 24 hours
            return 2  # Spoiled
    
    return df['Time_to_Spoilage_Minutes'].apply(label_time_to_spoilage)

def get_classifiers():
    """Get all classifiers for comparison"""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=1000)
    }

def visualize_model_classifications_on_timeseries(train_df, test_df, features):
    """
    Visualize how different models classify data points on MQ3 vs time graphs
    """
    
    print("🎯 MODEL CLASSIFICATION VISUALIZATION ON TIME SERIES")
    print("="*70)
    
    # Prepare data
    y_train = create_time_based_labels(train_df)
    y_test = create_time_based_labels(test_df)
    
    # Scale features
    scaler = StandardScaler()
    all_data = pd.concat([train_df[features], test_df[features]])
    scaler.fit(all_data)
    
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    classifiers = get_classifiers()
    
    # Create a large figure for all models
    n_models = len(classifiers)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Model Classifications on MQ3 Top vs Time', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Improved spacing
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Colors for classes
    colors = ['green', 'orange', 'red']
    class_names = ['Fresh', 'Spoiling', 'Spoiled']
    
    model_accuracies = {}
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        try:
            # Train model
            clf.fit(X_train, y_train)
            
            # Get predictions
            train_pred = clf.predict(X_train)
            test_pred = clf.predict(X_test)
            
            # Calculate accuracy
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            model_accuracies[name] = {'train': train_acc, 'test': test_acc}
            
            # Plot training data with true labels (smaller points, transparent)
            for class_idx in range(3):
                mask = y_train == class_idx
                if mask.any():
                    ax.scatter(train_df[mask]['Hours_from_start'], 
                             train_df[mask]['MQ3_Top_PPM'],
                             c=colors[class_idx], alpha=0.3, s=15, 
                             label=f'Train {class_names[class_idx]}' if idx == 0 else "")
            
            # Plot test data with predictions (larger points, distinct markers)
            markers = ['o', 's', '^']
            for class_idx in range(3):
                mask = test_pred == class_idx
                if mask.any():
                    ax.scatter(test_df[mask]['Hours_from_start'], 
                             test_df[mask]['MQ3_Top_PPM'],
                             c=colors[class_idx], alpha=0.8, s=50, 
                             marker=markers[class_idx], edgecolor='black', linewidth=1,
                             label=f'Pred {class_names[class_idx]}' if idx == 0 else "")
            
            # Add misclassifications as red X's
            misclassified = test_pred != y_test
            if misclassified.any():
                ax.scatter(test_df[misclassified]['Hours_from_start'], 
                         test_df[misclassified]['MQ3_Top_PPM'],
                         marker='x', c='red', s=100, linewidth=3, alpha=0.8,
                         label='Misclassified' if idx == 0 else "")
            
            ax.set_title(f'{name}\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}')
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel('MQ3 Top PPM')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Add legend only to first plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{name} - Error')
    
    # Hide unused subplots
    for idx in range(len(classifiers), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return model_accuracies

def visualize_decision_boundaries_2d(train_df, test_df, features):
    """
    Visualize decision boundaries in 2D space (MQ3 Top vs MQ3 Bottom)
    """
    
    print("\n🎯 MODEL DECISION BOUNDARIES (MQ3 Top vs MQ3 Bottom)")
    print("="*70)
    
    # Prepare data
    y_train = create_time_based_labels(train_df)
    y_test = create_time_based_labels(test_df)
    
    # Use only MQ3 features for 2D visualization
    mq3_features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']
    X_train_2d = train_df[mq3_features].fillna(train_df[mq3_features].mean())
    X_test_2d = test_df[mq3_features].fillna(test_df[mq3_features].mean())
    
    # Scale
    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)
    
    # Select top 4 models for decision boundary visualization
    top_models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Decision Boundaries in MQ3 Feature Space', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    axes_flat = axes.flatten()
    colors = ['green', 'orange', 'red']
    class_names = ['Fresh', 'Spoiling', 'Spoiled']
    
    for idx, (name, clf) in enumerate(top_models.items()):
        ax = axes_flat[idx]
        
        try:
            # Train model
            clf.fit(X_train_2d_scaled, y_train)
            
            # Create a mesh to plot decision boundary
            h = 0.02  # step size in the mesh
            x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
            y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Make predictions on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3, colors=['lightgreen', 'lightyellow', 'lightcoral'])
            
            # Plot training points
            for class_idx in range(3):
                mask = y_train == class_idx
                if mask.any():
                    ax.scatter(X_train_2d_scaled[mask, 0], X_train_2d_scaled[mask, 1],
                             c=colors[class_idx], alpha=0.6, s=30, 
                             label=f'{class_names[class_idx]}')
            
            # Plot test points with different markers
            test_pred = clf.predict(X_test_2d_scaled)
            for class_idx in range(3):
                mask = test_pred == class_idx
                if mask.any():
                    ax.scatter(X_test_2d_scaled[mask, 0], X_test_2d_scaled[mask, 1],
                             c=colors[class_idx], alpha=0.8, s=50, marker='s',
                             edgecolor='black', linewidth=1)
            
            # Highlight misclassifications
            misclassified = test_pred != y_test
            if misclassified.any():
                ax.scatter(X_test_2d_scaled[misclassified, 0], X_test_2d_scaled[misclassified, 1],
                         marker='x', c='red', s=100, linewidth=3, alpha=0.9)
            
            test_acc = accuracy_score(y_test, test_pred)
            ax.set_title(f'{name} (Test Acc: {test_acc:.3f})')
            ax.set_xlabel('MQ3 Top PPM (scaled)')
            ax.set_ylabel('MQ3 Bottom PPM (scaled)')
            
            if idx == 0:
                ax.legend()
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def visualize_prediction_confidence_over_time(train_df, test_df, features):
    """
    Visualize prediction confidence/probability over time
    """
    
    print("\n🎯 PREDICTION CONFIDENCE OVER TIME")
    print("="*50)
    
    # Prepare data
    y_train = create_time_based_labels(train_df)
    y_test = create_time_based_labels(test_df)
    
    scaler = StandardScaler()
    all_data = pd.concat([train_df[features], test_df[features]])
    scaler.fit(all_data)
    
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    # Models that support probability prediction
    prob_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Prediction Confidence Over Time', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    axes_flat = axes.flatten()
    
    for idx, (name, clf) in enumerate(prob_models.items()):
        ax = axes_flat[idx]
        
        try:
            # Train model
            clf.fit(X_train, y_train)
            
            # Get prediction probabilities
            test_proba = clf.predict_proba(X_test)
            test_pred = clf.predict(X_test)
            
            # Plot confidence for each class
            for class_idx in range(3):
                class_proba = test_proba[:, class_idx]
                ax.scatter(test_df['Hours_from_start'], class_proba, 
                         alpha=0.6, s=30, label=f'P({["Fresh", "Spoiling", "Spoiled"][class_idx]})')
            
            # Add actual predictions as line
            ax.scatter(test_df['Hours_from_start'], test_pred, 
                     c='black', marker='_', s=100, alpha=0.8, label='Prediction')
            
            # Add true labels
            ax.scatter(test_df['Hours_from_start'], y_test, 
                     c='red', marker='|', s=100, alpha=0.8, label='True Label')
            
            test_acc = accuracy_score(y_test, test_pred)
            ax.set_title(f'{name} (Acc: {test_acc:.3f})')
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel('Probability / Class')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 3.1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def analyze_misclassifications(train_df, test_df, features):
    """
    Detailed analysis of misclassifications
    """
    
    print("\n🎯 MISCLASSIFICATION ANALYSIS")
    print("="*50)
    
    # Prepare data
    y_train = create_time_based_labels(train_df)
    y_test = create_time_based_labels(test_df)
    
    scaler = StandardScaler()
    all_data = pd.concat([train_df[features], test_df[features]])
    scaler.fit(all_data)
    
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    # Use Random Forest as example
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    test_pred = rf.predict(X_test)
    
    # Find misclassifications
    misclassified = test_pred != y_test
    
    if misclassified.any():
        print(f"Number of misclassifications: {misclassified.sum()}/{len(y_test)} ({misclassified.sum()/len(y_test)*100:.1f}%)")
        
        # Create detailed misclassification plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3)
        
        # Plot 1: Misclassifications over time
        axes[0].scatter(test_df[~misclassified]['Hours_from_start'], 
                       test_df[~misclassified]['MQ3_Top_PPM'],
                       c='green', alpha=0.6, s=30, label='Correct')
        axes[0].scatter(test_df[misclassified]['Hours_from_start'], 
                       test_df[misclassified]['MQ3_Top_PPM'],
                       c='red', alpha=0.8, s=60, marker='x', label='Misclassified')
        axes[0].set_title('Misclassifications in Time Series')
        axes[0].set_xlabel('Hours from Start')
        axes[0].set_ylabel('MQ3 Top PPM')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Feature space analysis
        axes[1].scatter(test_df[~misclassified]['MQ3_Top_PPM'], 
                       test_df[~misclassified]['MQ3_Bottom_PPM'],
                       c='green', alpha=0.6, s=30, label='Correct')
        axes[1].scatter(test_df[misclassified]['MQ3_Top_PPM'], 
                       test_df[misclassified]['MQ3_Bottom_PPM'],
                       c='red', alpha=0.8, s=60, marker='x', label='Misclassified')
        axes[1].set_title('Misclassifications in Feature Space')
        axes[1].set_xlabel('MQ3 Top PPM')
        axes[1].set_ylabel('MQ3 Bottom PPM')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   xticklabels=['Fresh', 'Spoiling', 'Spoiled'],
                   yticklabels=['Fresh', 'Spoiling', 'Spoiled'])
        axes[2].set_title('Confusion Matrix')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('True')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze patterns in misclassifications
        print("\nMisclassification patterns:")
        for true_class in range(3):
            for pred_class in range(3):
                if true_class != pred_class:
                    count = ((y_test == true_class) & (test_pred == pred_class)).sum()
                    if count > 0:
                        class_names = ['Fresh', 'Spoiling', 'Spoiled']
                        print(f"  {class_names[true_class]} → {class_names[pred_class]}: {count} cases")
    
    else:
        print("Perfect classification! No misclassifications found.")

def suggest_improvements():
    """
    Suggest comprehensive improvements to the classification system
    """
    
    print(f"\n{'='*80}")
    print("🚀 COMPREHENSIVE IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    print("\n1. 📊 DATA COLLECTION IMPROVEMENTS:")
    print("-" * 40)
    print("• 🕒 Higher temporal resolution: Sample every 15-30 minutes instead of hourly")
    print("• 🔬 Additional sensors: pH, moisture content, gas chromatography validation")
    print("• 🌡️ Environmental logging: Ambient temperature, humidity, air circulation")
    print("• 📝 Better labeling: Visual spoilage assessment alongside time-based labels")
    print("• 🍓 Multiple strawberry varieties: Test generalization across cultivars")
    print("• 📦 Storage conditions: Different temperatures, packaging, batch sizes")
    
    print("\n2. 🛠️ FEATURE ENGINEERING ENHANCEMENTS:")
    print("-" * 40)
    print("• 📈 Temporal features: Rate of change, acceleration, trend analysis")
    print("• 🎯 Threshold features: Time spent above/below certain PPM levels")
    print("• 🔄 Cyclical features: Daily patterns, circadian rhythms")
    print("• 📊 Statistical features: Rolling statistics (mean, std, skewness)")
    print("• 🏃 Dynamic features: Velocity and acceleration of sensor changes")
    print("• 🔗 Cross-sensor features: Ratios, correlations between sensors")
    
    print("\n3. 🤖 ADVANCED MODELING TECHNIQUES:")
    print("-" * 40)
    print("• 🕒 Time series models: LSTM, GRU for temporal dependencies")
    print("• 🎪 Ensemble methods: Voting classifiers, stacking, blending")
    print("• 📊 Semi-supervised learning: Use unlabeled data for better boundaries")
    print("• 🎯 Active learning: Iteratively select most informative samples")
    print("• 🔄 Transfer learning: Pre-train on related food spoilage datasets")
    print("• 🎮 Anomaly detection: Identify unusual spoilage patterns")
    
    print("\n4. 🏷️ IMPROVED LABELING STRATEGIES:")
    print("-" * 40)
    print("• 👁️ Visual assessment: Human experts rate spoilage stages")
    print("• 🔬 Multi-modal labeling: Combine visual, chemical, and temporal data")
    print("• 📊 Probabilistic labels: Confidence scores instead of hard classes")
    print("• 🎯 Fine-grained classes: More than 3 spoilage stages")
    print("• 🔄 Dynamic boundaries: Adaptive thresholds based on batch characteristics")
    print("• 🤝 Consensus labeling: Multiple expert annotations")
    
    print("\n5. 📈 EVALUATION AND VALIDATION:")
    print("-" * 40)
    print("• 🕒 Temporal validation: Train on early batches, test on later ones")
    print("• 🎯 Class-balanced metrics: F1-score, balanced accuracy")
    print("• 📊 Uncertainty quantification: Prediction confidence intervals")
    print("• 🔍 Error analysis: Systematic study of failure modes")
    print("• 📉 Learning curves: Determine optimal dataset size")
    print("• 🎪 Cross-validation strategies: Group-based, time-aware splits")
    
    print("\n6. 🔧 SYSTEM OPTIMIZATION:")
    print("-" * 40)
    print("• ⚡ Real-time processing: Edge computing for instant predictions")
    print("• 📱 Mobile integration: Smartphone app for field deployment")
    print("• 🔔 Alert system: Notifications when spoilage detected")
    print("• 📊 Dashboard: Real-time monitoring and historical trends")
    print("• 🔄 Continuous learning: Model updates with new data")
    print("• 🛡️ Robustness: Handle sensor failures, missing data")
    
    print("\n7. 🎯 DOMAIN-SPECIFIC ENHANCEMENTS:")
    print("-" * 40)
    print("• 🍓 Strawberry biology: Incorporate knowledge of spoilage mechanisms")
    print("• 🦠 Microbiology: Model bacterial/fungal growth patterns")
    print("• 🌡️ Storage science: Integrate temperature/humidity effects")
    print("• 📦 Packaging impact: Study different container materials")
    print("• 🚚 Supply chain: Model transport and storage conditions")
    print("• 💰 Economic factors: Cost-benefit analysis of intervention timing")
    
    print("\n8. 📊 ADVANCED VISUALIZATION:")
    print("-" * 40)
    print("• 🎮 Interactive plots: Zoom, filter, explore data dynamically")
    print("• 🗺️ Feature importance maps: Show which sensors matter when")
    print("• 🎯 Decision trees visualization: Understand model logic")
    print("• 📈 Prediction trajectories: Show spoilage progression paths")
    print("• 🔄 Animation: Time-lapse of spoilage development")
    print("• 📱 Mobile-friendly: Responsive visualizations for field use")
    
    print("\n9. 🧪 EXPERIMENTAL DESIGN:")
    print("-" * 40)
    print("• 🎯 Controlled experiments: Vary single factors systematically")
    print("• 📊 A/B testing: Compare different sensor configurations")
    print("• 🔄 Longitudinal studies: Track same strawberries over time")
    print("• 🎪 Multi-center validation: Test across different facilities")
    print("• 📏 Calibration studies: Ensure sensor consistency")
    print("• 🔬 Validation studies: Compare with gold-standard methods")
    
    print("\n10. 🚀 DEPLOYMENT CONSIDERATIONS:")
    print("-" * 40)
    print("• 💰 Cost analysis: ROI calculations for different accuracy levels")
    print("• 🔧 Maintenance: Sensor calibration and replacement schedules")
    print("• 👥 User training: Interface design for non-technical users")
    print("• 📊 Performance monitoring: Track model drift and degradation")
    print("• 🔄 Update mechanisms: Safe model deployment and rollback")
    print("• 📏 Scalability: Handle thousands of monitoring units")
    
    print(f"\n💡 IMMEDIATE ACTIONABLE IMPROVEMENTS:")
    print("="*50)
    print("1. 🎯 Implement the hybrid labeling approach we discussed")
    print("2. 📊 Add temporal features (rate of change, acceleration)")
    print("3. 🤖 Try ensemble methods (Random Forest + SVM voting)")
    print("4. 🔍 Implement the visualization tools we created")
    print("5. 📈 Collect more frequent measurements (every 15-30 minutes)")
    print("6. 🔄 Set up continuous model monitoring and updating")

def main():
    """Main function to run all visualizations"""
    
    print("🎯 ADVANCED MODEL EVALUATION VISUALIZATION")
    print("="*70)
    
    # Load data
    train_df, test_df, features = load_and_prepare_data()
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Test data: {len(test_df)} samples")
    print(f"Features: {features}")
    
    # Run all visualizations
    print("\n1. Model Classifications on Time Series...")
    model_accuracies = visualize_model_classifications_on_timeseries(train_df, test_df, features)
    
    print("\n2. Decision Boundaries in 2D...")
    visualize_decision_boundaries_2d(train_df, test_df, features)
    
    print("\n3. Prediction Confidence Over Time...")
    visualize_prediction_confidence_over_time(train_df, test_df, features)
    
    print("\n4. Misclassification Analysis...")
    analyze_misclassifications(train_df, test_df, features)
    
    # Print model performance summary
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    for model, scores in model_accuracies.items():
        print(f"{model:<18}: Train={scores['train']:.3f}, Test={scores['test']:.3f}, Gap={scores['train']-scores['test']:.3f}")
    
    # Suggest improvements
    suggest_improvements()
    
    return model_accuracies

if __name__ == "__main__":
    results = main()
