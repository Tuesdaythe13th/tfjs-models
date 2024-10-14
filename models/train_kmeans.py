import json
from sklearn.cluster import KMeans
import numpy as np

# Create some dummy data (replace this with your actual feature data)
features = np.random.rand(100, 4)  # 100 samples, 4 features per sample

# Train the KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features)

# Convert the trained KMeans model to a JSON-serializable format
kmeans_data = {
    "n_clusters": kmeans.n_clusters,
    "random_state": kmeans.random_state,
    "centroids": kmeans.cluster_centers_.tolist()  # Centroids of clusters
}

# Save the KMeans model to a JSON file
with open("kmeans_model.json", "w") as json_file:
    json.dump(kmeans_data, json_file)

print("KMeans model saved to kmeans_model.json")
