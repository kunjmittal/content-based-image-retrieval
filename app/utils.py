"""
Utility functions for the CBIR system
"""

import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from typing import List, Tuple, Optional
import tempfile

def get_image_files(directory: str) -> List[str]:
    """
    Get all image files from a directory
    
    Args:
        directory: Path to directory
        
    Returns:
        List of image file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)

def load_and_resize_image(image_path: str, max_size: Tuple[int, int] = (300, 300)) -> Optional[np.ndarray]:
    """
    Load and resize image for display
    
    Args:
        image_path: Path to image
        max_size: Maximum size (width, height)
        
    Returns:
        Resized image array or None if error
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def display_image_grid(images: List[Tuple[str, float, str]], title: str = "Similar Images"):
    """
    Display images in a grid layout with similarity scores
    
    Args:
        images: List of tuples (image_name, similarity_score, image_path)
        title: Title for the grid
    """
    if not images:
        st.write("No images to display")
        return
    
    st.subheader(title)
    
    # Display images in columns
    cols = st.columns(min(3, len(images)))
    
    for i, (image_name, similarity, image_path) in enumerate(images):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Load and display image
            img_array = load_and_resize_image(image_path)
            if img_array is not None:
                st.image(img_array, caption=f"{image_name}\nSimilarity: {similarity:.3f}", use_column_width=True)
            else:
                st.error(f"Could not load {image_name}")

def create_feature_cache_path(dataset_path: str) -> str:
    """Create path for feature cache file"""
    return os.path.join(os.path.dirname(dataset_path), "features_cache.pkl")

def create_clustering_cache_path(dataset_path: str) -> str:
    """Create path for clustering cache file"""
    return os.path.join(os.path.dirname(dataset_path), "clustering_cache.pkl")

def check_dataset_exists(dataset_path: str) -> bool:
    """Check if dataset directory exists and contains images"""
    if not os.path.exists(dataset_path):
        return False
    
    image_files = get_image_files(dataset_path)
    return len(image_files) > 0

def format_similarity_score(score: float) -> str:
    """Format similarity score for display"""
    return f"{score:.3f} ({score*100:.1f}%)"

def get_cluster_color(cluster_id: int) -> str:
    """Get color for cluster visualization"""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    return colors[cluster_id % len(colors)]

def display_cluster_info(cluster_id: int, total_clusters: int):
    """Display cluster information"""
    if cluster_id >= 0:
        color = get_cluster_color(cluster_id)
        st.markdown(
            f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0;">'
            f'<strong>Cluster {cluster_id}</strong> (of {total_clusters} total clusters)'
            f'</div>',
            unsafe_allow_html=True
        )