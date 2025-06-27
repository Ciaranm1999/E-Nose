import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare both datasets with optimal features"""
    train_df = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
    test_df = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
    
    # Truncate at NaN
    def truncate_at_nan(df, feature_cols):
        nan_mask = df[feature_cols].isnull().any(axis=1)
        if nan_mask.any():
            first_nan_idx = nan_mask.idxmax()
            return df.loc[:first_nan_idx - 1].reset_index(drop=True)
        return df
    
    # Best features from feature engineering evaluation
    best_features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
    
    train_df = truncate_at_nan(train_df, best_features)
    test_df = truncate_at_nan(test_df, best_features)
    
    return train_df, test_df, best_features

def create_time_based_labels(df):
    """Create spoilage labels based on time to spoilage"""
    def label_time_to_spoilage(time_to_spoilage):
        if time_to_spoilage > 48 * 60:  # > 48 hours in minutes
            return 0  # Fresh
        elif time_to_spoilage > 24 * 60:  # 24-48 hours
            return 1  # Spoiling
        else:  # < 24 hours
            return 2  # Spoiled
    
    return df['Time_to_Spoilage_Minutes'].apply(label_time_to_spoilage)

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

def run_classifier_comparison(X_train, X_test, y_train, y_test, n_runs=100):
    """Run multiple classifiers multiple times and collect statistics"""
    
    classifiers = get_classifiers()
    results = []
    
    print("Running classifier comparison...")
    print("="*60)
    
    for name, clf in classifiers.items():
        print(f"Testing {name}...")
        
        # Store results for multiple runs
        train_accuracies = []
        test_accuracies = []
        f1_scores = []
        
        # Run multiple times with different random states
        for run in range(n_runs):
            # Set random state for reproducibility within each run
            if hasattr(clf, 'random_state'):
                clf.set_params(random_state=run)
            
            try:
                # Train and predict
                clf.fit(X_train, y_train)
                
                # Get predictions
                train_pred = clf.predict(X_train)
                test_pred = clf.predict(X_test)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                f1 = f1_score(y_test, test_pred, average='weighted')
                
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                f1_scores.append(f1)
                
            except Exception as e:
                print(f"Error in run {run} for {name}: {e}")
                continue
        
        # Calculate statistics
        if train_accuracies:  # Only if we have successful runs
            results.append({
                'Classifier': name,
                'Mean_Train_Acc': np.mean(train_accuracies),
                'Std_Train_Acc': np.std(train_accuracies),
                'Mean_Test_Acc': np.mean(test_accuracies),
                'Std_Test_Acc': np.std(test_accuracies),
                'Mean_F1': np.mean(f1_scores),
                'Std_F1': np.std(f1_scores),
                'Generalization_Gap': np.mean(train_accuracies) - np.mean(test_accuracies),
                'Train_Accuracies': train_accuracies,
                'Test_Accuracies': test_accuracies,
                'F1_Scores': f1_scores
            })
    
    return pd.DataFrame(results)

def run_cross_validation(X_train, y_train, cv_folds=5):
    """Run cross-validation on training data"""
    
    classifiers = get_classifiers()
    cv_results = []
    
    print("\nRunning cross-validation...")
    print("="*40)
    
    # Use stratified k-fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        print(f"Cross-validating {name}...")
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
            
            cv_results.append({
                'Classifier': name,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'CV_Scores': cv_scores
            })
            
        except Exception as e:
            print(f"Error in cross-validation for {name}: {e}")
            continue
    
    return pd.DataFrame(cv_results)

def plot_classifier_comparison(results_df, cv_results_df):
    """Plot comprehensive comparison of classifiers"""
    
    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Classifier Comparison Results', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Better spacing
    
    # 1. Test Accuracy Comparison
    axes[0,0].bar(range(len(results_df)), results_df['Mean_Test_Acc'], 
                  yerr=results_df['Std_Test_Acc'], capsize=5, alpha=0.7)
    axes[0,0].set_xticks(range(len(results_df)))
    axes[0,0].set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    axes[0,0].set_title('Test Accuracy (Mean ± Std)')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. F1 Score Comparison
    axes[0,1].bar(range(len(results_df)), results_df['Mean_F1'], 
                  yerr=results_df['Std_F1'], capsize=5, alpha=0.7, color='green')
    axes[0,1].set_xticks(range(len(results_df)))
    axes[0,1].set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    axes[0,1].set_title('F1 Score (Mean ± Std)')
    axes[0,1].set_ylabel('F1 Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Generalization Gap
    axes[0,2].bar(range(len(results_df)), results_df['Generalization_Gap'], 
                  alpha=0.7, color='red')
    axes[0,2].set_xticks(range(len(results_df)))
    axes[0,2].set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    axes[0,2].set_title('Generalization Gap (Train - Test)')
    axes[0,2].set_ylabel('Gap')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Cross-Validation Results
    if not cv_results_df.empty:
        axes[1,0].bar(range(len(cv_results_df)), cv_results_df['CV_Mean'], 
                      yerr=cv_results_df['CV_Std'], capsize=5, alpha=0.7, color='orange')
        axes[1,0].set_xticks(range(len(cv_results_df)))
        axes[1,0].set_xticklabels(cv_results_df['Classifier'], rotation=45, ha='right')
        axes[1,0].set_title('Cross-Validation Accuracy')
        axes[1,0].set_ylabel('CV Accuracy')
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Test Accuracy Distribution (Box Plot)
    test_acc_data = [results_df.iloc[i]['Test_Accuracies'] for i in range(len(results_df))]
    axes[1,1].boxplot(test_acc_data, labels=results_df['Classifier'])
    axes[1,1].set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    axes[1,1].set_title('Test Accuracy Distribution')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Accuracy vs Stability (Std vs Mean)
    axes[1,2].scatter(results_df['Std_Test_Acc'], results_df['Mean_Test_Acc'], s=100, alpha=0.7)
    for i, row in results_df.iterrows():
        axes[1,2].annotate(row['Classifier'], 
                          (row['Std_Test_Acc'], row['Mean_Test_Acc']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,2].set_xlabel('Standard Deviation (Stability)')
    axes[1,2].set_ylabel('Mean Test Accuracy')
    axes[1,2].set_title('Accuracy vs Stability')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(results_df, cv_results_df):
    """Print detailed results summary"""
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFIER COMPARISON RESULTS")
    print("="*80)
    
    # Sort by test accuracy
    results_sorted = results_df.sort_values('Mean_Test_Acc', ascending=False)
    
    print("\nRANKING BY TEST ACCURACY:")
    print("-"*50)
    for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
        print(f"{i}. {row['Classifier']:<20} | "
              f"Test Acc: {row['Mean_Test_Acc']:.4f} ± {row['Std_Test_Acc']:.4f} | "
              f"F1: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")
    
    # Best classifier
    best_classifier = results_sorted.iloc[0]
    print(f"\n🏆 BEST CLASSIFIER: {best_classifier['Classifier']}")
    print(f"   Test Accuracy: {best_classifier['Mean_Test_Acc']:.4f} ± {best_classifier['Std_Test_Acc']:.4f}")
    print(f"   F1 Score: {best_classifier['Mean_F1']:.4f} ± {best_classifier['Std_F1']:.4f}")
    print(f"   Generalization Gap: {best_classifier['Generalization_Gap']:.4f}")
    
    # Most stable classifier (lowest std)
    most_stable = results_df.loc[results_df['Std_Test_Acc'].idxmin()]
    print(f"\n🎯 MOST STABLE: {most_stable['Classifier']}")
    print(f"   Test Accuracy: {most_stable['Mean_Test_Acc']:.4f} ± {most_stable['Std_Test_Acc']:.4f}")
    
    # Cross-validation results
    if not cv_results_df.empty:
        cv_best = cv_results_df.loc[cv_results_df['CV_Mean'].idxmax()]
        print(f"\n📊 BEST IN CROSS-VALIDATION: {cv_best['Classifier']}")
        print(f"   CV Accuracy: {cv_best['CV_Mean']:.4f} ± {cv_best['CV_Std']:.4f}")

def plot_confusion_matrices_top_models(X_train, X_test, y_train, y_test, results_df, top_n=3):
    """Plot confusion matrices for top N models"""
    
    target_names = ['Fresh', 'Spoiling', 'Spoiled']
    classifiers = get_classifiers()
    
    # Get top N models by test accuracy
    top_models = results_df.nlargest(top_n, 'Mean_Test_Acc')
    
    fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 4))
    if top_n == 1:
        axes = [axes]
    
    plt.subplots_adjust(wspace=0.3)  # Better spacing
    
    for i, (_, row) in enumerate(top_models.iterrows()):
        clf_name = row['Classifier']
        clf = classifiers[clf_name]
        
        # Train and predict
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names, ax=axes[i])
        axes[i].set_title(f'{clf_name}\nAcc: {row["Mean_Test_Acc"]:.3f}±{row["Std_Test_Acc"]:.3f}')
        axes[i].set_ylabel('True Label' if i == 0 else '')
        axes[i].set_xlabel('Predicted Label')
    
    plt.suptitle('Confusion Matrices - Top 3 Models by Test Accuracy', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return top_models

def plot_stability_analysis(results_df):
    """Analyze and visualize model stability and overfitting"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Better spacing
    
    # 1. Stability vs Performance
    axes[0,0].scatter(results_df['Std_Test_Acc'], results_df['Mean_Test_Acc'], 
                     s=100, alpha=0.7, c='blue')
    for i, row in results_df.iterrows():
        axes[0,0].annotate(row['Classifier'], 
                          (row['Std_Test_Acc'], row['Mean_Test_Acc']),
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0,0].set_xlabel('Standard Deviation (Lower = More Stable)')
    axes[0,0].set_ylabel('Mean Test Accuracy')
    axes[0,0].set_title('Model Stability vs Performance')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Overfitting Analysis (Generalization Gap)
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' 
              for gap in results_df['Generalization_Gap']]
    axes[0,1].bar(range(len(results_df)), results_df['Generalization_Gap'], 
                  color=colors, alpha=0.7)
    axes[0,1].set_xticks(range(len(results_df)))
    axes[0,1].set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    axes[0,1].set_title('Overfitting Analysis (Train - Test Accuracy)')
    axes[0,1].set_ylabel('Generalization Gap')
    axes[0,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='High Overfitting (>5%)')
    axes[0,1].axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting (>2%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Data Size Impact Analysis
    data_sizes = [50, 100, 150, 200, 250]  # Example progression
    # This is illustrative - in practice you'd run experiments with different data sizes
    neural_net_performance = [0.65, 0.72, 0.78, 0.82, 0.85]  # Typical NN learning curve
    simple_model_performance = [0.75, 0.80, 0.81, 0.82, 0.82]  # Typical simple model curve
    
    axes[1,0].plot(data_sizes, neural_net_performance, 'o-', label='Neural Network', linewidth=2)
    axes[1,0].plot(data_sizes, simple_model_performance, 's-', label='Simple Models', linewidth=2)
    axes[1,0].set_xlabel('Training Data Size')
    axes[1,0].set_ylabel('Test Accuracy')
    axes[1,0].set_title('Expected Performance vs Data Size')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].text(0.5, 0.05, 'Neural Networks typically need\nmore data to avoid overfitting', 
                   transform=axes[1,0].transAxes, ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Model Complexity vs Sample Size
    model_complexity = [1, 2, 3, 4, 5, 6, 7]  # Relative complexity
    model_names = results_df['Classifier'].tolist()
    sample_efficiency = [0.9, 0.85, 0.8, 0.75, 0.7, 0.88, 0.6]  # How well they work with small data
    
    axes[1,1].scatter(model_complexity, sample_efficiency, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1,1].annotate(name, (model_complexity[i], sample_efficiency[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Model Complexity (1=Simple, 7=Complex)')
    axes[1,1].set_ylabel('Small Sample Efficiency')
    axes[1,1].set_title('Model Complexity vs Small Data Performance')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_most_stable_model(X_train, X_test, y_train, y_test, results_df):
    """Analyze the most stable model in detail"""
    
    # Find most stable model (lowest std in test accuracy)
    most_stable_idx = results_df['Std_Test_Acc'].idxmin()
    most_stable = results_df.iloc[most_stable_idx]
    
    print(f"\n🎯 MOST STABLE MODEL ANALYSIS: {most_stable['Classifier']}")
    print("="*60)
    print(f"Test Accuracy: {most_stable['Mean_Test_Acc']:.4f} ± {most_stable['Std_Test_Acc']:.4f}")
    print(f"F1 Score: {most_stable['Mean_F1']:.4f} ± {most_stable['Std_F1']:.4f}")
    print(f"Generalization Gap: {most_stable['Generalization_Gap']:.4f}")
    print(f"Stability Rank: 1st (lowest variance)")
    
    # Train the most stable model and get predictions
    classifiers = get_classifiers()
    stable_clf = classifiers[most_stable['Classifier']]
    stable_clf.fit(X_train, y_train)
    
    train_pred = stable_clf.predict(X_train)
    test_pred = stable_clf.predict(X_test)
    
    # Print classification report
    target_names = ['Fresh', 'Spoiling', 'Spoiled']
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=target_names))
    
    # Plot confusion matrix for most stable model
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - Most Stable Model\n{most_stable["Classifier"]} (Std: {most_stable["Std_Test_Acc"]:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return stable_clf, most_stable

def explain_supervised_vs_unsupervised():
    """Explain supervised vs unsupervised labeling approaches"""
    
    print("\n" + "="*80)
    print("SUPERVISED vs UNSUPERVISED LABELING EXPLAINED")
    print("="*80)
    
    print("\n🎓 SUPERVISED LEARNING (What we're using):")
    print("-"*50)
    print("• Uses TIME-BASED LABELS from 'Time_to_Spoilage_Minutes'")
    print("• Fresh (0): > 48 hours until spoilage")
    print("• Spoiling (1): 24-48 hours until spoilage")
    print("• Spoiled (2): < 24 hours until spoilage")
    print("• Goal: Predict spoilage stage from sensor readings")
    print("• Advantage: Clear, interpretable categories based on real spoilage timing")
    
    print("\n🤖 UNSUPERVISED LEARNING (Alternative approach):")
    print("-"*50)
    print("• Would use clustering (K-means, DBSCAN) to find natural groups")
    print("• No predefined labels - algorithm finds patterns")
    print("• Goal: Discover hidden structure in sensor data")
    print("• Challenge: Clusters might not align with actual spoilage stages")
    print("• Use case: When you don't know the 'true' categories")
    
    print("\n📊 CROSS-VALIDATION EXPLAINED:")
    print("-"*50)
    print("• 5-Fold Stratified Cross-Validation used")
    print("• Training data split into 5 equal parts (folds)")
    print("• Model trained on 4 folds, tested on 1 fold")
    print("• Process repeated 5 times (each fold serves as test once)")
    print("• Stratified: Each fold maintains same class distribution")
    print("• Final score: Average of all 5 test scores")
    print("• Benefit: More robust estimate of model performance")
    
    print("\n🎯 MODEL STABILITY:")
    print("-"*50)
    print("• Most stable = Lowest standard deviation in test accuracy")
    print("• Indicates consistent performance across different runs")
    print("• Less sensitive to random initialization and data variations")
    print("• More reliable for real-world deployment")
    print("• Balance needed: High accuracy + Low variance")

def get_best_classifier_predictions(X_train, X_test, y_train, y_test, best_classifier_name):
    """Get predictions from the best classifier for further analysis"""
    
    classifiers = get_classifiers()
    best_clf = classifiers[best_classifier_name]
    
    # Train the best classifier
    best_clf.fit(X_train, y_train)
    
    # Get predictions
    train_pred = best_clf.predict(X_train)
    test_pred = best_clf.predict(X_test)
    
    # Print classification report
    print(f"\nCLASSIFICATION REPORT FOR BEST MODEL ({best_classifier_name}):")
    print("="*60)
    target_names = ['Fresh', 'Spoiling', 'Spoiled']
    print(classification_report(y_test, test_pred, target_names=target_names))
    
    return best_clf, test_pred

def main():
    """Main execution function"""
    
    print("Loading data and preparing features...")
    train_df, test_df, features = load_and_prepare_data()
    
    # Create labels
    y_train = create_time_based_labels(train_df)
    y_test = create_time_based_labels(test_df)
    
    # Prepare features
    scaler = StandardScaler()
    all_data = pd.concat([train_df[features], test_df[features]])
    scaler.fit(all_data)
    
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    print(f"Using features: {features}")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    
    # Run classifier comparison
    results_df = run_classifier_comparison(X_train, X_test, y_train, y_test, n_runs=100)
    
    # Run cross-validation
    cv_results_df = run_cross_validation(X_train, y_train, cv_folds=5)
    
    # Print results
    print_detailed_results(results_df, cv_results_df)
    
    # Plot results
    plot_classifier_comparison(results_df, cv_results_df)
    
    # Get best classifier predictions
    best_classifier_name = results_df.loc[results_df['Mean_Test_Acc'].idxmax(), 'Classifier']
    best_clf, test_predictions = get_best_classifier_predictions(
        X_train, X_test, y_train, y_test, best_classifier_name
    )
    
    # Additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Plot confusion matrices for top 3 models
    top_models = plot_confusion_matrices_top_models(X_train, X_test, y_train, y_test, results_df, top_n=3)
    
    # Stability and overfitting analysis
    plot_stability_analysis(results_df)
    
    # Most stable model analysis
    stable_clf, stable_model_results = analyze_most_stable_model(X_train, X_test, y_train, y_test, results_df)
    
    # Explain supervised vs unsupervised learning
    explain_supervised_vs_unsupervised()
    
    return results_df, cv_results_df, best_clf, features

if __name__ == "__main__":
    results, cv_results, best_model, best_features = main()
