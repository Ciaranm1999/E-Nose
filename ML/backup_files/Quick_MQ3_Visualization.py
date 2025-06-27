import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

def quick_load_and_visualize():
    """Quick visualization of MQ3 data to identify spoilage patterns"""
    
    print("Loading data...")
    try:
        # Load data
        batch1 = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
        batch2 = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
        
        print(f"Batch 1: {len(batch1)} samples")
        print(f"Batch 2: {len(batch2)} samples")
        
        for batch_name, df in [("Batch 1", batch1), ("Batch 2", batch2)]:
            print(f"\n{batch_name} Analysis:")
            print("-" * 40)
            
            # Calculate hours from start
            df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
              # Create comprehensive visualization with better spacing
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{batch_name} - MQ3 Sensor Analysis for Spoilage Detection', fontsize=16)
            plt.subplots_adjust(hspace=0.35, wspace=0.25)  # Improved spacing between plots
            
            # 1. MQ3 Top over time
            axes[0,0].plot(df['Hours_from_start'], df['MQ3_Top_PPM'], 'b-', linewidth=2)
            axes[0,0].set_title('MQ3 Top PPM vs Time')
            axes[0,0].set_xlabel('Hours from Start')
            axes[0,0].set_ylabel('PPM')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add spoilage boundaries if available
            if 'Time_to_Spoilage_Minutes' in df.columns:
                max_time = df['Time_to_Spoilage_Minutes'].max()
                spoilage_time = max_time / 60  # Convert to hours
                
                # Current time-based boundaries
                axes[0,0].axvline(x=spoilage_time, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Spoilage Point')
                axes[0,0].axvline(x=spoilage_time-24, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Spoiling Start (24h before)')
                axes[0,0].axvline(x=spoilage_time-48, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Fresh End (48h before)')
                axes[0,0].legend()
                
                print(f"  Spoilage at: {spoilage_time:.1f} hours")
                print(f"  Spoiling starts: {spoilage_time-24:.1f} hours")
                print(f"  Fresh period ends: {spoilage_time-48:.1f} hours")
            
            # 2. MQ3 Bottom over time
            axes[0,1].plot(df['Hours_from_start'], df['MQ3_Bottom_PPM'], 'r-', linewidth=2)
            axes[0,1].set_title('MQ3 Bottom PPM vs Time')
            axes[0,1].set_xlabel('Hours from Start')
            axes[0,1].set_ylabel('PPM')
            axes[0,1].grid(True, alpha=0.3)
            
            if 'Time_to_Spoilage_Minutes' in df.columns:
                axes[0,1].axvline(x=spoilage_time, color='red', linestyle='-', linewidth=2, alpha=0.8)
                axes[0,1].axvline(x=spoilage_time-24, color='orange', linestyle='--', linewidth=2, alpha=0.8)
                axes[0,1].axvline(x=spoilage_time-48, color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            # 3. Both sensors together with current labels
            axes[0,2].plot(df['Hours_from_start'], df['MQ3_Top_PPM'], 'b-', linewidth=2, label='MQ3 Top', alpha=0.8)
            axes[0,2].plot(df['Hours_from_start'], df['MQ3_Bottom_PPM'], 'r-', linewidth=2, label='MQ3 Bottom', alpha=0.8)
            
            # Color background by current time-based labels
            if 'Time_to_Spoilage_Minutes' in df.columns:
                # Create time-based labels
                def create_labels(time_to_spoilage):
                    if time_to_spoilage > 48 * 60:
                        return 0  # Fresh
                    elif time_to_spoilage > 24 * 60:
                        return 1  # Spoiling
                    else:
                        return 2  # Spoiled
                
                labels = df['Time_to_Spoilage_Minutes'].apply(create_labels)
                
                # Color regions
                fresh_mask = labels == 0
                spoiling_mask = labels == 1
                spoiled_mask = labels == 2
                
                if fresh_mask.any():
                    axes[0,2].axvspan(df[fresh_mask]['Hours_from_start'].min(), 
                                    df[fresh_mask]['Hours_from_start'].max(), 
                                    alpha=0.2, color='green', label='Fresh Period')
                if spoiling_mask.any():
                    axes[0,2].axvspan(df[spoiling_mask]['Hours_from_start'].min(), 
                                    df[spoiling_mask]['Hours_from_start'].max(), 
                                    alpha=0.2, color='orange', label='Spoiling Period')
                if spoiled_mask.any():
                    axes[0,2].axvspan(df[spoiled_mask]['Hours_from_start'].min(), 
                                    df[spoiled_mask]['Hours_from_start'].max(), 
                                    alpha=0.2, color='red', label='Spoiled Period')
            
            axes[0,2].set_title('Both Sensors with Current Time-based Labels')
            axes[0,2].set_xlabel('Hours from Start')
            axes[0,2].set_ylabel('PPM')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # 4. Rate of change analysis
            window = max(3, len(df) // 30)
            df['MQ3_Top_smooth'] = df['MQ3_Top_PPM'].rolling(window=window, center=True).mean()
            
            if len(df) > window:
                rate_of_change = np.gradient(df['MQ3_Top_smooth'].fillna(method='bfill').fillna(method='ffill'))
                axes[1,0].plot(df['Hours_from_start'], rate_of_change, 'purple', linewidth=2)
                axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1,0].set_title('Rate of Change in MQ3 Top')
                axes[1,0].set_xlabel('Hours from Start')
                axes[1,0].set_ylabel('PPM/Hour')
                axes[1,0].grid(True, alpha=0.3)
                
                # Highlight significant changes
                threshold = np.std(rate_of_change) * 1.5
                significant_changes = np.abs(rate_of_change) > threshold
                if significant_changes.any():
                    axes[1,0].scatter(df[significant_changes]['Hours_from_start'], 
                                    rate_of_change[significant_changes], 
                                    color='red', s=50, alpha=0.8, label='Significant Changes')
                    axes[1,0].legend()
            
            # 5. K-means clustering comparison
            feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            if len(X) > 3:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Try 3 clusters
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                
                # Plot K-means results
                colors = ['blue', 'green', 'red']
                for i in range(3):
                    mask = kmeans_labels == i
                    if mask.any():
                        axes[1,1].scatter(df[mask]['Hours_from_start'], 
                                        df[mask]['MQ3_Top_PPM'], 
                                        c=colors[i], alpha=0.6, s=30, label=f'Cluster {i}')
                
                axes[1,1].set_title('K-means Clustering (3 clusters)')
                axes[1,1].set_xlabel('Hours from Start')
                axes[1,1].set_ylabel('MQ3 Top PPM')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
                
                # Compare with time-based labels
                if 'Time_to_Spoilage_Minutes' in df.columns:
                    agreement = adjusted_rand_score(labels, kmeans_labels)
                    axes[1,1].text(0.02, 0.98, f'Agreement with time labels: {agreement:.3f}', 
                                  transform=axes[1,1].transAxes, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    print(f"  K-means vs Time-based agreement: {agreement:.3f}")
                    
                    # Distribution of clusters
                    print(f"  K-means cluster sizes: {np.bincount(kmeans_labels)}")
                    print(f"  Time-based label sizes: {np.bincount(labels)}")
            
            # 6. Suggested manual boundaries
            axes[1,2].plot(df['Hours_from_start'], df['MQ3_Top_PPM'], 'b-', linewidth=2, label='MQ3 Top')
            
            # Find potential transition points using change detection
            if len(df) > 10:
                # Moving average
                window = max(5, len(df) // 20)
                smooth_data = df['MQ3_Top_PPM'].rolling(window=window, center=True).mean()
                
                # Find points where slope changes significantly
                if len(smooth_data.dropna()) > 5:
                    diff = np.diff(smooth_data.dropna())
                    diff2 = np.diff(diff)  # Second derivative
                    
                    # Find inflection points
                    threshold = np.std(diff2) * 1.2
                    inflection_points = np.where(np.abs(diff2) > threshold)[0]
                    
                    if len(inflection_points) > 0:
                        # Convert back to original indices
                        valid_indices = smooth_data.dropna().index[inflection_points]
                        
                        for idx in valid_indices[:3]:  # Show first 3 inflection points
                            if idx < len(df):
                                time_point = df.loc[idx, 'Hours_from_start']
                                axes[1,2].axvline(x=time_point, color='orange', 
                                                linestyle=':', linewidth=2, alpha=0.8)
                                axes[1,2].text(time_point, axes[1,2].get_ylim()[1]*0.9, 
                                              f'{time_point:.1f}h', 
                                              rotation=90, ha='right', va='top')
                        
                        print(f"  Suggested transition points (hours): {[df.loc[idx, 'Hours_from_start'] for idx in valid_indices[:3] if idx < len(df)]}")
            
            axes[1,2].set_title('Suggested Manual Boundaries')
            axes[1,2].set_xlabel('Hours from Start')
            axes[1,2].set_ylabel('MQ3 Top PPM')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].legend()
            
            plt.tight_layout()
            plt.show()
            
            print(f"  Visual inspection questions:")
            print(f"  - Do you see clear transitions in the MQ3 data?")
            print(f"  - Do the current time-based boundaries align with sensor changes?")
            print(f"  - Would you adjust the boundaries based on what you see?")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the data files exist in the correct location.")

def compare_labeling_approaches():
    """Compare different labeling approaches"""
    
    print("\n" + "="*80)
    print("LABELING APPROACH COMPARISON")
    print("="*80)
    
    print("\n🕐 TIME-BASED LABELING (Current Approach):")
    print("   Pros:")
    print("   ✅ Based on actual spoilage timing")
    print("   ✅ Clear, interpretable categories")
    print("   ✅ Consistent across batches")
    print("   ✅ Domain knowledge incorporated")
    
    print("   Cons:")
    print("   ❌ May not match sensor patterns")
    print("   ❌ Assumes linear spoilage progression")
    print("   ❌ Fixed boundaries may be suboptimal")
    
    print("\n🤖 K-MEANS CLUSTERING:")
    print("   Pros:")
    print("   ✅ Finds natural sensor patterns")
    print("   ✅ Data-driven boundaries")
    print("   ✅ May discover unexpected groupings")
    print("   ✅ No assumptions about timing")
    
    print("   Cons:")
    print("   ❌ Clusters may not match spoilage stages")
    print("   ❌ Less interpretable")
    print("   ❌ May vary between batches")
    print("   ❌ Ignores domain knowledge")
    
    print("\n🔧 HYBRID APPROACH (Recommended):")
    print("   Strategy:")
    print("   • Start with time-based labels as baseline")
    print("   • Use visual inspection to identify sensor transitions")
    print("   • Apply K-means to validate natural groupings")
    print("   • Manually adjust boundaries based on both approaches")
    print("   • Use your known spoilage time ranges as constraints")
    
    print("\n💡 SPECIFIC RECOMMENDATIONS FOR YOUR DATA:")
    print("   1. Examine the MQ3 plots above carefully")
    print("   2. Look for clear changes in sensor behavior")
    print("   3. Note where K-means disagrees with time labels")
    print("   4. Consider these key questions:")
    print("      - Do sensors show gradual or sudden changes?")
    print("      - Are there clear transition points?")
    print("      - Do both MQ3 sensors behave similarly?")
    print("      - Does the rate of change plot show inflection points?")
    
    print("\n🎯 REGARDING NEURAL NETWORKS:")
    print("   You're absolutely right to be cautious about neural networks!")
    print("   Reasons to avoid them for this application:")
    print("   • Small dataset size (typically need 1000+ samples per class)")
    print("   • High risk of overfitting")
    print("   • Less interpretable than simpler models")
    print("   • Overkill for this type of classification problem")
    print("   ")
    print("   Better alternatives:")
    print("   • Random Forest (excellent for small datasets)")
    print("   • SVM (good for high-dimensional small data)")
    print("   • Logistic Regression (interpretable and stable)")
    print("   • Gradient Boosting (if you have enough data)")

if __name__ == "__main__":
    quick_load_and_visualize()
    compare_labeling_approaches()
