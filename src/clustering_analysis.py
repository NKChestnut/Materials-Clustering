import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_materials_data.csv")


print(f"Loading processed data from {PROCESSED_DATA_FILE}...")
try:
    df_processed = pd.read_csv(PROCESSED_DATA_FILE)
    print("Processed data loaded successfully.")
    print(f"Number of materials: {len(df_processed)}")
    print("\nProcessed DataFrame Head:")
    print(df_processed.head())
    print("\nProcessed DataFrame Info:")
    df_processed.info()
except FileNotFoundError:
    print(f"Error: {PROCESSED_DATA_FILE} not found. Please ensure data preprocessing was successful in Step 2.")
    print("Expected path:", os.path.abspath(PROCESSED_DATA_FILE))
    exit()
# Define the list of non-element numerical columns
numerical_cols = ['nelements', 'band_gap', 'formation_energy_per_atom', 'density']

# Infer all_elements_sorted from the processed DataFrame columns
# All columns except the numerical_cols and 'cluster'  are element columns
all_elements_sorted = [col for col in df_processed.columns if col not in numerical_cols and col != 'cluster']
print(f"Inferred {len(all_elements_sorted)} element columns for analysis.")

print("\n Step 3: Dimensionality Reduction and Determining Optimal Clusters")

# Data Scaling
print("Scaling data using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)
print("Data scaled successfully.")
print(f"Shape of scaled data: {X_scaled.shape}")

X_scaled_df = pd.DataFrame(X_scaled, columns=df_processed.columns)
print("\nScaled Data Head (first 5 rows):")
print(X_scaled_df.head())


# Apply PCA
print("\nPerforming PCA to analyze explained variance...")
pca = PCA()
pca.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance by Number of Components')
plt.grid(True)
plt.axvline(x=2, linestyle='--', color='red', label='2 Components')
plt.axvline(x=3, linestyle='--', color='green', label='3 Components')
plt.legend()
plt.tight_layout()
plt.show()

# PCA to get the desired number of components (e.g., 2 for 2D visualization)
n_components = 2
pca_2d = PCA(n_components=n_components)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"Data reduced to {n_components} dimensions using PCA.")
print(f"Explained variance by {n_components} components: {np.sum(pca_2d.explained_variance_ratio_):.2f}")
print(f"Shape of 2D PCA data: {X_pca_2d.shape}")

df_pca = pd.DataFrame(data=X_pca_2d, columns=[f'PC{i+1}' for i in range(n_components)], index=df_processed.index)
print("\nPCA 2D Data Head:")
print(df_pca.head())

# Determine Optimal K

# Elbow Method
print("\nPerforming Elbow Method to find optimal K...")
sse = []
k_range = range(2, 16)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# Silhouette Score
print("\nCalculating Silhouette Scores for various K...")
silhouette_scores = []
k_range_silhouette = k_range

for k in k_range_silhouette:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range_silhouette, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')
plt.xticks(k_range_silhouette)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nReview the plots to determine an optimal 'k'.")
print("Look for the 'elbow' point in the SSE plot and the highest score in the Silhouette plot.")

# Placeholder for optimal K:
optimal_k = 5 
print(f"\nPlaceholder for optimal K: {optimal_k} (Adjust this based on plot analysis)")


# Perform K-Means Clustering
print(f"\n Step 4: Clustering and Interpretation")
optimal_k = 10

print(f"Applying K-Means clustering with optimal K = {optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled) # Fit on the full scaled data

# Add cluster labels to the PCA DataFrame
df_pca['cluster'] = cluster_labels
# Also add cluster labels to the original processed DataFrame for interpretation
df_processed['cluster'] = cluster_labels

print(f"K-Means clustering complete. Assigned {optimal_k} clusters.")
print("Cluster distribution:")
print(df_pca['cluster'].value_counts().sort_index())

# Visualize Clusters
print("\nGenerating 2D PCA Scatter Plot of Clusters...")
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='tab10', # A distinct color palette for up to 10 clusters
    data=df_pca,
    legend='full',
    alpha=0.6,
    s=20 # Adjust point size
)
plt.title(f'K-Means Clusters (k={optimal_k}) in 2D PCA Space')
plt.xlabel(f'Principal Component 1 (explains {pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 (explains {pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.tight_layout()
plt.show() # Display the plot

# Interpret Cluster Characteristics
print("\nAnalyzing Cluster Characteristics...")

# Calculate mean values for each feature within each cluster
cluster_means = df_processed.groupby('cluster')[df_processed.columns[:-1]].mean() # Exclude the 'cluster' column itself
# Sort elements by their overall abundance for better readability
overall_element_abundance = df_processed[df_processed.columns[:-4]].mean().sort_values(ascending=False).index.tolist()

print("\nTop 10 most abundant elements by overall dataset:")
print(df_processed[df_processed.columns[:-4]].mean().sort_values(ascending=False).head(10))

# Print top N elements and average properties for each cluster
num_top_elements = 5 # Number of top elements to show per cluster
print(f"\nTop {num_top_elements} elements and average properties for each cluster:")
for i in range(optimal_k):
    print(f"\n Cluster {i} (Count: {df_processed['cluster'].value_counts().get(i, 0)})")
    
    # Get mean elemental percentages for this cluster, sort and print top N
    cluster_element_means = cluster_means.loc[i, all_elements_sorted].sort_values(ascending=False)
    print("  Most Abundant Elements:")
    for element, percentage in cluster_element_means.head(num_top_elements).items():
        print(f"    {element}: {percentage:.4f}")
    
    # Get mean of other numerical properties for this cluster
    print("  Average Properties:")
    for prop in numerical_cols:
        print(f"    {prop}: {cluster_means.loc[i, prop]:.4f}")

# Save clustered data
CLUSTERED_DATA_FILE = os.path.join(DATA_DIR, "clustered_materials_data.csv")
# Save the original DataFrame with the new 'cluster' column
df_processed.to_csv(CLUSTERED_DATA_FILE, index=False)
print(f"\nClustered data saved to {CLUSTERED_DATA_FILE}")

