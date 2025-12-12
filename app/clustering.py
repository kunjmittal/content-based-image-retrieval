"""
K-Means Clustering Module
Implements clustering of images based on their ORB features
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import pickle

class ImageClusterer:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize image clusterer
        
        Args:
            n_clusters: Number of clusters for k-means
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
        self.image_clusters = {}
        self.cluster_centers = None
        
    def prepare_feature_vectors(self, features_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert ORB descriptors to fixed-size feature vectors for clustering
        
        Args:
            features_dict: Dictionary mapping image names to ORB descriptors
            
        Returns:
            Tuple of (feature_matrix, image_names)
        """
        feature_vectors = []
        image_names = []
        
        for image_name, descriptors in features_dict.items():
            if descriptors is not None and len(descriptors) > 0:
                # Create a histogram of descriptor values (bag of visual words approach)
                # Flatten all descriptors and create statistical features
                flattened = descriptors.flatten()
                
                # Create statistical features: mean, std, min, max for each bit position
                feature_vector = []
                
                # Reshape descriptors to work with them
                desc_reshaped = descriptors.reshape(-1, descriptors.shape[-1])
                
                # Statistical features across all descriptors
                feature_vector.extend([
                    np.mean(desc_reshaped, axis=0).mean(),  # Overall mean
                    np.std(desc_reshaped, axis=0).mean(),   # Overall std
                    np.min(desc_reshaped, axis=0).mean(),   # Overall min
                    np.max(desc_reshaped, axis=0).mean(),   # Overall max
                ])
                
                # Histogram of descriptor values
                hist, _ = np.histogram(flattened, bins=50, range=(0, 255))
                feature_vector.extend(hist.astype(float) / len(flattened))  # Normalize
                
                # Number of keypoints (normalized)
                feature_vector.append(len(descriptors) / 1000.0)  # Normalize by 1000
                
                feature_vectors.append(feature_vector)
                image_names.append(image_name)
        
        return np.array(feature_vectors), image_names
    
    def fit_clustering(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Perform k-means clustering on image features
        
        Args:
            features_dict: Dictionary mapping image names to ORB descriptors
            
        Returns:
            Dictionary mapping image names to cluster IDs
        """
        # Prepare feature vectors
        feature_matrix, image_names = self.prepare_feature_vectors(features_dict)
        
        if len(feature_matrix) == 0:
            print("No valid features found for clustering")
            return {}
        
        print(f"Clustering {len(feature_matrix)} images into {self.n_clusters} clusters...")
        print(f"Feature vector dimension: {feature_matrix.shape[1]}")
        
        # Apply PCA for dimensionality reduction if needed
        if feature_matrix.shape[1] > 50:
            # Ensure n_components doesn't exceed min(n_samples, n_features)
            max_components = min(feature_matrix.shape[0] - 1, feature_matrix.shape[1], 50)
            self.pca = PCA(n_components=max_components, random_state=self.random_state)
            feature_matrix = self.pca.fit_transform(feature_matrix)
            print(f"Applied PCA, reduced to {feature_matrix.shape[1]} dimensions")
        
        # Perform k-means clustering
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(feature_matrix)), 
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(feature_matrix)
        
        # Create mapping from image names to cluster IDs
        self.image_clusters = {}
        for image_name, cluster_id in zip(image_names, cluster_labels):
            self.image_clusters[image_name] = int(cluster_id)
        
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Print cluster statistics
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        print("Cluster distribution:")
        for cluster_id, count in zip(unique_clusters, counts):
            print(f"  Cluster {cluster_id}: {count} images")
        
        return self.image_clusters
    
    def get_cluster_id(self, image_name: str) -> int:
        """Get cluster ID for a specific image"""
        return self.image_clusters.get(image_name, -1)
    
    def get_images_in_cluster(self, cluster_id: int) -> List[str]:
        """Get all images in a specific cluster"""
        return [img for img, cid in self.image_clusters.items() if cid == cluster_id]
    
    def predict_cluster(self, feature_vector: np.ndarray) -> int:
        """Predict cluster for a new feature vector"""
        if self.kmeans is None:
            return -1
        
        # Apply PCA if it was used during training
        if self.pca is not None:
            feature_vector = self.pca.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)
        
        return int(self.kmeans.predict(feature_vector)[0])
    
    def save_clustering_model(self, save_path: str):
        """Save clustering model and results"""
        model_data = {
            'kmeans': self.kmeans,
            'pca': self.pca,
            'image_clusters': self.image_clusters,
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Clustering model saved to {save_path}")
    
    def load_clustering_model(self, load_path: str):
        """Load clustering model and results"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans = model_data['kmeans']
        self.pca = model_data['pca']
        self.image_clusters = model_data['image_clusters']
        self.cluster_centers = model_data['cluster_centers']
        self.n_clusters = model_data['n_clusters']
        
        print(f"Clustering model loaded from {load_path}")