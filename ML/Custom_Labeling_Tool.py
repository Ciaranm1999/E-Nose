import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def create_custom_labeling_tool():
    """
    Interactive tool to help you create custom labels based on visual inspection
    """
    
    print("CUSTOM LABELING TOOL")
    print("="*50)
    print("This tool helps you create custom spoilage labels based on your visual inspection.")
    print("First, run Quick_MQ3_Visualization.py to see the sensor data plots.")
    print("Then use this tool to implement your custom boundaries.\n")
    
    try:
        # Load data
        batch1 = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
        batch2 = pd.read_csv('../Data Processing/Data/batch_two/complete_data.csv', parse_dates=['Timestamp'])
        
        for batch_name, df in [("Batch 1", batch1), ("Batch 2", batch2)]:
            print(f"\n{batch_name} Analysis:")
            print("-" * 30)
            
            # Calculate hours from start
            df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
            
            # Show current time-based boundaries
            if 'Time_to_Spoilage_Minutes' in df.columns:
                max_time = df['Time_to_Spoilage_Minutes'].max()
                spoilage_time = max_time / 60
                spoiling_start = spoilage_time - 24
                fresh_end = spoilage_time - 48
                
                print(f"Current time-based boundaries:")
                print(f"  Fresh period: 0 to {fresh_end:.1f} hours")
                print(f"  Spoiling period: {fresh_end:.1f} to {spoiling_start:.1f} hours") 
                print(f"  Spoiled period: {spoiling_start:.1f} to {spoilage_time:.1f} hours")
                
                # Create original time-based labels
                def create_time_labels(time_to_spoilage):
                    if time_to_spoilage > 48 * 60:
                        return 0  # Fresh
                    elif time_to_spoilage > 24 * 60:
                        return 1  # Spoiling
                    else:
                        return 2  # Spoiled
                
                time_labels = df['Time_to_Spoilage_Minutes'].apply(create_time_labels)
                
                print(f"  Label distribution: Fresh={np.sum(time_labels==0)}, Spoiling={np.sum(time_labels==1)}, Spoiled={np.sum(time_labels==2)}")
            
            # K-means comparison
            feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            if len(X) > 3:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                
                # Map K-means clusters to temporal order
                cluster_times = {}
                for i in range(3):
                    mask = kmeans_labels == i
                    if mask.any():
                        cluster_times[i] = df[mask]['Hours_from_start'].mean()
                
                sorted_clusters = sorted(cluster_times.items(), key=lambda x: x[1])
                cluster_to_label = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}
                kmeans_ordered = np.array([cluster_to_label[c] for c in kmeans_labels])
                
                print(f"\nK-means clustering results:")
                print(f"  Label distribution: Fresh={np.sum(kmeans_ordered==0)}, Spoiling={np.sum(kmeans_ordered==1)}, Spoiled={np.sum(kmeans_ordered==2)}")
                
                if 'Time_to_Spoilage_Minutes' in df.columns:
                    from sklearn.metrics import adjusted_rand_score
                    agreement = adjusted_rand_score(time_labels, kmeans_ordered)
                    print(f"  Agreement with time labels: {agreement:.3f}")
                
                # Silhouette scores
                if 'Time_to_Spoilage_Minutes' in df.columns:
                    time_silhouette = silhouette_score(X_scaled, time_labels)
                    kmeans_silhouette = silhouette_score(X_scaled, kmeans_ordered)
                    print(f"\nSilhouette scores (higher is better):")
                    print(f"  Time-based labels: {time_silhouette:.3f}")
                    print(f"  K-means labels: {kmeans_silhouette:.3f}")
            
            print(f"\nTo create custom labels for {batch_name}, modify the function below:")
            print(f"def create_custom_labels_{batch_name.lower().replace(' ', '_')}(df):")
            print(f"    '''Custom labeling based on visual inspection'''")
            print(f"    def custom_rule(hours):")
            print(f"        if hours <= YOUR_FRESH_END_HOUR:     # e.g., 25")
            print(f"            return 0  # Fresh")
            print(f"        elif hours <= YOUR_SPOILING_END_HOUR: # e.g., 50") 
            print(f"            return 1  # Spoiling")
            print(f"        else:")
            print(f"            return 2  # Spoiled")
            print(f"    return df['Hours_from_start'].apply(custom_rule)")
            
        print(f"\n{'='*50}")
        print("EXAMPLE IMPLEMENTATION:")
        print("="*50)
        print("""
# After examining the plots, if you decide that:
# - Fresh period ends at 30 hours (instead of time-based boundary)
# - Spoiling period ends at 55 hours (instead of time-based boundary)

def create_your_custom_labels(df):
    '''Your custom labeling based on visual inspection'''
    
    def custom_rule(hours):
        if hours <= 30:      # YOUR observed fresh end
            return 0  # Fresh
        elif hours <= 55:    # YOUR observed spoiling end  
            return 1  # Spoiling
        else:
            return 2  # Spoiled
    
    return df['Hours_from_start'].apply(custom_rule)

# Then test it:
custom_labels = create_your_custom_labels(your_dataframe)

# And evaluate:
features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
X = your_dataframe[features].fillna(your_dataframe[features].mean())
X_scaled = StandardScaler().fit_transform(X)
custom_silhouette = silhouette_score(X_scaled, custom_labels)
print(f"Custom labeling silhouette score: {custom_silhouette:.3f}")
""")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Look at the MQ3 plots and identify clear transition points")
        print("2. Note the hours where you see sensor behavior change")  
        print("3. Create custom boundaries based on these observations")
        print("4. Test custom labels using the code above")
        print("5. Compare silhouette scores - higher is better")
        print("6. Use the labeling approach with the best score + visual validation")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure data files are in the correct location.")

def test_custom_labeling_example():
    """
    Example of how to implement and test custom labeling
    """
    
    print(f"\n{'='*60}")
    print("EXAMPLE: TESTING CUSTOM LABELING")
    print("="*60)
    
    try:
        # Load batch 1 for example
        df = pd.read_csv('../Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])
        df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
        
        # Example custom labeling (replace with your observations)
        def example_custom_labels(df):
            """Example custom labeling - replace with your boundaries"""
            def custom_rule(hours):
                # These are EXAMPLE boundaries - replace with YOUR observations!
                if hours <= 25:      # Example: fresh period based on visual inspection
                    return 0  # Fresh
                elif hours <= 50:    # Example: spoiling period based on visual inspection
                    return 1  # Spoiling
                else:
                    return 2  # Spoiled
            return df['Hours_from_start'].apply(custom_rule)
        
        # Create different label sets
        custom_labels = example_custom_labels(df)
        
        # Original time-based labels
        if 'Time_to_Spoilage_Minutes' in df.columns:
            def create_time_labels(time_to_spoilage):
                if time_to_spoilage > 48 * 60:
                    return 0
                elif time_to_spoilage > 24 * 60:
                    return 1
                else:
                    return 2
            time_labels = df['Time_to_Spoilage_Minutes'].apply(create_time_labels)
        
        # K-means labels
        feature_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
        X = df[feature_cols].fillna(df[feature_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Order K-means by time
        cluster_times = {}
        for i in range(3):
            mask = kmeans_labels == i
            if mask.any():
                cluster_times[i] = df[mask]['Hours_from_start'].mean()
        
        sorted_clusters = sorted(cluster_times.items(), key=lambda x: x[1])
        cluster_to_label = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}
        kmeans_ordered = np.array([cluster_to_label[c] for c in kmeans_labels])
        
        # Compare all approaches
        print("Label Distribution Comparison:")
        print(f"Custom labels:     Fresh={np.sum(custom_labels==0)}, Spoiling={np.sum(custom_labels==1)}, Spoiled={np.sum(custom_labels==2)}")
        if 'Time_to_Spoilage_Minutes' in df.columns:
            print(f"Time-based labels: Fresh={np.sum(time_labels==0)}, Spoiling={np.sum(time_labels==1)}, Spoiled={np.sum(time_labels==2)}")
        print(f"K-means labels:    Fresh={np.sum(kmeans_ordered==0)}, Spoiling={np.sum(kmeans_ordered==1)}, Spoiled={np.sum(kmeans_ordered==2)}")
        
        # Silhouette scores
        custom_silhouette = silhouette_score(X_scaled, custom_labels)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_ordered)
        
        print(f"\nSilhouette Scores (higher is better):")
        print(f"Custom labels: {custom_silhouette:.3f}")
        if 'Time_to_Spoilage_Minutes' in df.columns:
            time_silhouette = silhouette_score(X_scaled, time_labels)
            print(f"Time-based:    {time_silhouette:.3f}")
        print(f"K-means:       {kmeans_silhouette:.3f}")
        
        # Visualize custom labels
        plt.figure(figsize=(14, 8))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Better spacing
        
        plt.subplot(2, 2, 1)
        colors = ['green', 'orange', 'red']
        for i in range(3):
            mask = custom_labels == i
            if mask.any():
                plt.scatter(df[mask]['Hours_from_start'], df[mask]['MQ3_Top_PPM'], 
                          c=colors[i], alpha=0.6, s=30, label=f'Custom {i}')
        plt.title('Custom Labels')
        plt.xlabel('Hours from Start')
        plt.ylabel('MQ3 Top PPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if 'Time_to_Spoilage_Minutes' in df.columns:
            plt.subplot(2, 2, 2)
            for i in range(3):
                mask = time_labels == i
                if mask.any():
                    plt.scatter(df[mask]['Hours_from_start'], df[mask]['MQ3_Top_PPM'], 
                              c=colors[i], alpha=0.6, s=30, label=f'Time {i}')
            plt.title('Time-based Labels')
            plt.xlabel('Hours from Start')  
            plt.ylabel('MQ3 Top PPM')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        for i in range(3):
            mask = kmeans_ordered == i
            if mask.any():
                plt.scatter(df[mask]['Hours_from_start'], df[mask]['MQ3_Top_PPM'], 
                          c=colors[i], alpha=0.6, s=30, label=f'K-means {i}')
        plt.title('K-means Labels')
        plt.xlabel('Hours from Start')
        plt.ylabel('MQ3 Top PPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        methods = ['Custom', 'K-means']
        scores = [custom_silhouette, kmeans_silhouette]
        if 'Time_to_Spoilage_Minutes' in df.columns:
            methods.append('Time-based')
            scores.append(time_silhouette)
        
        bars = plt.bar(methods, scores, color=['blue', 'green', 'red'][:len(methods)], alpha=0.7)
        plt.title('Silhouette Score Comparison')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n🎯 RECOMMENDATION:")
        best_method = methods[np.argmax(scores)]
        print(f"Based on silhouette scores, {best_method} labeling performs best.")
        print("However, also consider:")
        print("• Visual validation (do the boundaries make sense?)")
        print("• Domain knowledge (do they align with spoilage biology?)")
        print("• Consistency across batches")
        
    except Exception as e:
        print(f"Error in example: {e}")

if __name__ == "__main__":
    create_custom_labeling_tool()
    test_custom_labeling_example()
