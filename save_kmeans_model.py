import json
from sklearn.cluster import KMeans
import numpy as np

# Generate random data for KMeans (this is just an example)
X = np.random.rand(100, 4)

# Train the KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Save the model as a JSON file
kmeans_model = {
    "n_clusters": kmeans.n_clusters,
    "random_state": kmeans.random_state,
    "centroids": kmeans.cluster_centers_.tolist()  # Convert numpy array to a Python list
}

with open('models/kmeans_model.json', 'w') as f:
    json.dump(kmeans_model, f)
