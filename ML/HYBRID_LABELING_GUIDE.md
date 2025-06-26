# HYBRID LABELING APPROACH - Implementation Guide

## Step-by-Step Implementation

### Step 1: Visual Inspection and Manual Boundary Detection

```python
# Run this to see your data:
python Quick_MQ3_Visualization.py
```

**Look for:**
- Clear changes in MQ3 sensor slopes
- Inflection points where rate of change shifts
- Differences between top and bottom sensors
- Natural "breakpoints" in the data

### Step 2: Create Custom Labels Based on Visual + K-means

```python
def create_hybrid_labels(df, manual_boundaries=None):
    """
    Create labels using hybrid approach:
    1. Visual inspection boundaries (manual_boundaries)
    2. K-means validation
    3. Known spoilage time constraints
    """
    
    # If you provide manual boundaries (in hours from start)
    if manual_boundaries:
        fresh_end, spoiling_end = manual_boundaries
        
        def hybrid_labeling(row):
            hours = row['Hours_from_start']
            if hours <= fresh_end:
                return 0  # Fresh
            elif hours <= spoiling_end:
                return 1  # Spoiling
            else:
                return 2  # Spoiled
        
        return df.apply(hybrid_labeling, axis=1)
    
    # Otherwise use K-means + time constraints
    else:
        # K-means on sensor data
        features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM']
        X = df[features].fillna(df[features].mean())
        X_scaled = StandardScaler().fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Map clusters to spoilage stages based on time progression
        # (cluster that occurs earliest = fresh, latest = spoiled)
        cluster_times = {}
        for i in range(3):
            mask = clusters == i
            if mask.any():
                cluster_times[i] = df[mask]['Hours_from_start'].mean()
        
        # Sort clusters by average time
        sorted_clusters = sorted(cluster_times.items(), key=lambda x: x[1])
        
        # Map to spoilage labels
        cluster_to_label = {}
        for idx, (cluster, _) in enumerate(sorted_clusters):
            cluster_to_label[cluster] = idx
        
        return np.array([cluster_to_label[c] for c in clusters])
```

### Step 3: Validate Your Labeling Choice

```python
def compare_labeling_performance(df, custom_labels):
    """Compare different labeling approaches"""
    
    # Original time-based labels
    time_labels = create_time_based_labels(df)
    
    # K-means labels
    kmeans_labels = create_kmeans_labels(df)
    
    # Your custom labels
    
    # Compare using silhouette score
    features = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'BME_Temp', 'BME_Humidity']
    X = df[features].fillna(df[features].mean())
    X_scaled = StandardScaler().fit_transform(X)
    
    time_score = silhouette_score(X_scaled, time_labels)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    custom_score = silhouette_score(X_scaled, custom_labels)
    
    print(f"Time-based labeling score: {time_score:.3f}")
    print(f"K-means labeling score: {kmeans_score:.3f}")
    print(f"Custom labeling score: {custom_score:.3f}")
    
    return custom_score > time_score
```

## Key Questions for You to Answer (from the plots):

### 🔍 **Visual Inspection Questions:**

1. **Looking at MQ3 Top PPM over time:**
   - Do you see a clear point where the slope changes dramatically?
   - Is there a gradual increase or sudden jumps?
   - Where would YOU draw the line for "spoiling starts"?

2. **Comparing current boundaries:**
   - Do the orange/red lines align with visible sensor changes?
   - Are the boundaries too early, too late, or about right?

3. **Rate of change plot:**
   - Are there clear spikes indicating rapid changes?
   - Do these align with your expected spoilage timeline?

4. **K-means vs time labels:**
   - Does K-means find different patterns than time-based labels?
   - Which approach seems to better capture sensor behavior?

## Practical Example:

```python
# Example of how to implement your custom boundaries
# (Replace these with YOUR observations from the plots)

def create_your_custom_labels(df):
    """
    Create labels based on YOUR visual inspection
    """
    
    # Example: If you observe from plots that:
    # - Fresh period: 0-30 hours (sensor stable)
    # - Spoiling starts: 30-55 hours (sensor increases)  
    # - Spoiled: 55+ hours (sensor high/unstable)
    
    def custom_labeling(hours):
        if hours <= 30:      # Adjust based on YOUR observation
            return 0  # Fresh
        elif hours <= 55:    # Adjust based on YOUR observation  
            return 1  # Spoiling
        else:
            return 2  # Spoiled
    
    return df['Hours_from_start'].apply(custom_labeling)

# Then test this labeling approach:
custom_labels = create_your_custom_labels(your_dataframe)

# Compare with machine learning performance
# Train models using custom_labels instead of time_labels
```

## Why This Approach is Better:

✅ **Combines domain expertise (your knowledge) with data patterns**
✅ **Uses actual sensor behavior instead of rigid time rules**  
✅ **Allows you to incorporate known spoilage timing as constraints**
✅ **More likely to generalize to new batches**
✅ **You can visually validate the boundaries make sense**

## Next Steps:

1. **Run the visualization scripts** and examine the plots carefully
2. **Identify transition points** where you see sensor behavior change
3. **Note the hours** where these transitions occur
4. **Create custom boundaries** based on your observations
5. **Test the new labels** with the classifier comparison script
6. **Compare performance** with original time-based approach

The key insight: **Trust your eyes and domain knowledge more than rigid time rules!**
