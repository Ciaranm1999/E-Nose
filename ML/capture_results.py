import json
import pandas as pd
from Data_Visualization_and_Labeling_Analysis import main

def capture_and_save_results():
    """Run the analysis and capture results to JSON file"""
    
    print("Running E-Nose analysis and capturing results...")
    
    # Run the main analysis
    train_enhanced, test_enhanced, model_results = main()
    
    if model_results:
        # Extract key metrics for each model
        results_summary = {}
        
        for model_name, results in model_results.items():
            if results is not None:
                results_summary[model_name] = {
                    'test_accuracy': float(results['test_accuracy']),
                    'train_accuracy': float(results['train_accuracy']),
                    'test_f1': float(results['test_f1']),
                    'train_f1': float(results['train_f1']),
                    'cv_mean': float(results['cv_mean']),
                    'cv_std': float(results['cv_std']),
                    'overfitting_score': float(results['overfitting_score']),
                    'is_overfitting': bool(results['is_overfitting'])
                }
        
        # Save to JSON file
        with open('actual_model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to 'actual_model_results.json'")
        print("Summary of results:")
        print("="*50)
        
        # Print summary table
        for model_name, metrics in results_summary.items():
            print(f"\n{model_name}:")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
            print(f"  Test F1 Score: {metrics['test_f1']:.3f}")
            print(f"  CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
            print(f"  Overfitting: {'Yes' if metrics['is_overfitting'] else 'No'}")
        
        return results_summary
    else:
        print("ERROR: No model results to capture")
        return None

if __name__ == "__main__":
    actual_results = capture_and_save_results()
