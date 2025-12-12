"""
ORB Feature Extraction Module
Implements ORB (Oriented FAST and Rotated BRIEF) feature extraction for images
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import pickle

class ORBFeatureExtractor:
    def __init__(self, n_features: int = 500):
        """
        Initialize ORB feature extractor
        
        Args:
            n_features: Maximum number of features to extract per image
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.feature_cache = {}
        
    def extract_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ORB features from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if descriptors is None:
            # Return empty arrays if no features found
            return np.array([]), np.array([])
        
        return keypoints, descriptors
    
    def extract_features_from_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """
        Extract features from all images in dataset
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary mapping image names to their descriptors
        """
        features_dict = {}
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(dataset_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        print(f"Processing {len(image_files)} images...")
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(dataset_path, filename)
            try:
                keypoints, descriptors = self.extract_features(image_path)
                if descriptors is not None and len(descriptors) > 0:
                    features_dict[filename] = descriptors
                    print(f"Processed {i+1}/{len(image_files)}: {filename} - {len(descriptors)} features")
                else:
                    print(f"No features found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return features_dict
    
    def save_features(self, features_dict: Dict[str, np.ndarray], save_path: str):
        """Save extracted features to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"Features saved to {save_path}")
    
    def load_features(self, load_path: str) -> Dict[str, np.ndarray]:
        """Load features from file"""
        with open(load_path, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"Features loaded from {load_path}")
        return features_dict