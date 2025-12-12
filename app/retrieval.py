"""
Image Retrieval Module
Implements similarity search using Hamming distance for ORB descriptors
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
import os

class ImageRetriever:
    def __init__(self, features_dict: Dict[str, np.ndarray], dataset_path: str):
        """
        Initialize image retriever
        
        Args:
            features_dict: Dictionary mapping image names to their ORB descriptors
            dataset_path: Path to dataset directory
        """
        self.features_dict = features_dict
        self.dataset_path = dataset_path
        self.image_names = list(features_dict.keys())
        
    def compute_hamming_distance(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Compute Hamming distance between two ORB descriptor sets
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            
        Returns:
            Average minimum Hamming distance
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return float('inf')
        
        # For each descriptor in desc1, find the minimum distance to any descriptor in desc2
        min_distances = []
        
        for d1 in desc1:
            distances = []
            for d2 in desc2:
                # Convert binary descriptors to binary strings and compute Hamming distance
                dist = np.sum(d1 != d2) / len(d1)  # Normalized Hamming distance
                distances.append(dist)
            min_distances.append(min(distances))
        
        # Return average of minimum distances
        return np.mean(min_distances) if min_distances else float('inf')
    
    def find_similar_images(self, query_descriptors: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar images to query image
        
        Args:
            query_descriptors: ORB descriptors of query image
            top_k: Number of similar images to return
            
        Returns:
            List of tuples (image_name, similarity_score)
        """
        similarities = []
        
        for image_name, descriptors in self.features_dict.items():
            distance = self.compute_hamming_distance(query_descriptors, descriptors)
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + distance)
            similarities.append((image_name, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_image_path(self, image_name: str) -> str:
        """Get full path to image"""
        return os.path.join(self.dataset_path, image_name)
    
    def retrieve_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Complete retrieval pipeline
        
        Args:
            query_image_path: Path to query image
            top_k: Number of similar images to return
            
        Returns:
            List of tuples (image_name, similarity_score, full_path)
        """
        from app.orb_features import ORBFeatureExtractor
        
        # Extract features from query image
        extractor = ORBFeatureExtractor()
        try:
            _, query_descriptors = extractor.extract_features(query_image_path)
            
            if query_descriptors is None or len(query_descriptors) == 0:
                print("No features found in query image")
                return []
            
            # Find similar images
            similar_images = self.find_similar_images(query_descriptors, top_k)
            
            # Add full paths
            results = []
            for image_name, similarity in similar_images:
                full_path = self.get_image_path(image_name)
                results.append((image_name, similarity, full_path))
            
            return results
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []