"""
Content-Based Image Retrieval System
Main Streamlit Application

This application implements image retrieval using:
- ORB (Oriented FAST and Rotated BRIEF) features
- Hamming distance similarity
- k-Means clustering
"""

import streamlit as st
import os
import numpy as np
from PIL import Image
import tempfile
import time

# Import our modules
from app.orb_features import ORBFeatureExtractor
from app.retrieval import ImageRetriever
from app.clustering import ImageClusterer
from app.utils import (
    save_uploaded_file, display_image_grid, load_and_resize_image,
    create_feature_cache_path, create_clustering_cache_path,
    check_dataset_exists, format_similarity_score, display_cluster_info
)

# Configuration
DATASET_PATH = "dataset"
CACHE_DIR = "cache"

def initialize_system():
    """Initialize the CBIR system by loading or creating features and clusters"""
    
    # Check if dataset exists
    if not check_dataset_exists(DATASET_PATH):
        st.error(f"Dataset directory '{DATASET_PATH}' not found or empty!")
        st.info("Please ensure you have images in the dataset folder.")
        return None, None, None
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    feature_cache_path = os.path.join(CACHE_DIR, "features.pkl")
    clustering_cache_path = os.path.join(CACHE_DIR, "clustering.pkl")
    
    # Initialize components
    extractor = ORBFeatureExtractor(n_features=500)
    clusterer = ImageClusterer(n_clusters=5)
    
    # Load or extract features
    if os.path.exists(feature_cache_path):
        st.info("Loading cached features...")
        features_dict = extractor.load_features(feature_cache_path)
    else:
        st.info("Extracting ORB features from dataset images...")
        with st.spinner("Processing images..."):
            features_dict = extractor.extract_features_from_dataset(DATASET_PATH)
            extractor.save_features(features_dict, feature_cache_path)
        st.success(f"Extracted features from {len(features_dict)} images!")
    
    # Load or create clustering
    if os.path.exists(clustering_cache_path):
        st.info("Loading cached clustering model...")
        clusterer.load_clustering_model(clustering_cache_path)
    else:
        st.info("Performing k-means clustering...")
        with st.spinner("Clustering images..."):
            clusterer.fit_clustering(features_dict)
            clusterer.save_clustering_model(clustering_cache_path)
        st.success("Clustering completed!")
    
    # Initialize retriever
    retriever = ImageRetriever(features_dict, DATASET_PATH)
    
    return extractor, retriever, clusterer

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Content-Based Image Retrieval",
        page_icon="🔍",
        layout="wide"
    )
    
    # Title and description
    st.title("🔍 Content-Based Image Retrieval System")
    st.markdown("### Using ORB Features & k-Means Clustering")
    
    st.markdown("""
    This system uses **ORB (Oriented FAST and Rotated BRIEF)** features to find visually similar images.
    Upload an image to find similar images from the dataset using Hamming distance similarity.
    """)
    
    # Initialize system
    with st.spinner("Initializing CBIR system..."):
        extractor, retriever, clusterer = initialize_system()
    
    if extractor is None:
        return
    
    st.success("✅ System initialized successfully!")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of similar images to retrieve", 1, 20, 5)
    show_cluster_info = st.sidebar.checkbox("Show cluster information", True)
    
    # Display dataset info
    st.sidebar.header("Dataset Info")
    dataset_images = len(retriever.features_dict)
    st.sidebar.metric("Total Images", dataset_images)
    st.sidebar.metric("Total Clusters", clusterer.n_clusters)
    
    # Main interface
    st.header("Upload Query Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to find similar images in the dataset"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Query Image")
            # Display the uploaded image
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption=uploaded_file.name, use_column_width=True)
        
        with col2:
            st.subheader("Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {query_image.size}")
            st.write(f"**Mode:** {query_image.mode}")
        
        # Search button
        if st.button("🔍 Search Similar Images", type="primary"):
            # Save uploaded file temporarily
            temp_path = save_uploaded_file(uploaded_file)
            
            try:
                with st.spinner("Searching for similar images..."):
                    # Perform retrieval
                    start_time = time.time()
                    results = retriever.retrieve_similar_images(temp_path, top_k)
                    search_time = time.time() - start_time
                
                if results:
                    st.success(f"Found {len(results)} similar images in {search_time:.2f} seconds!")
                    
                    # Show cluster information for query image if enabled
                    if show_cluster_info and clusterer:
                        try:
                            # Extract features from query image to predict cluster
                            _, query_descriptors = extractor.extract_features(temp_path)
                            if query_descriptors is not None and len(query_descriptors) > 0:
                                # Prepare feature vector for clustering prediction
                                feature_vectors, _ = clusterer.prepare_feature_vectors({
                                    'query': query_descriptors
                                })
                                if len(feature_vectors) > 0:
                                    query_cluster = clusterer.predict_cluster(feature_vectors[0])
                                    st.subheader("Query Image Cluster")
                                    display_cluster_info(query_cluster, clusterer.n_clusters)
                        except Exception as e:
                            st.warning(f"Could not determine query image cluster: {e}")
                    
                    # Display results
                    st.header("Similar Images")
                    
                    # Create columns for results
                    cols = st.columns(min(3, len(results)))
                    
                    for i, (image_name, similarity, image_path) in enumerate(results):
                        col_idx = i % len(cols)
                        
                        with cols[col_idx]:
                            # Load and display image
                            img_array = load_and_resize_image(image_path)
                            if img_array is not None:
                                st.image(img_array, use_column_width=True)
                                st.write(f"**{image_name}**")
                                st.write(f"Similarity: {format_similarity_score(similarity)}")
                                
                                # Show cluster info if enabled
                                if show_cluster_info:
                                    cluster_id = clusterer.get_cluster_id(image_name)
                                    if cluster_id >= 0:
                                        display_cluster_info(cluster_id, clusterer.n_clusters)
                            else:
                                st.error(f"Could not load {image_name}")
                    
                    # Additional statistics
                    st.subheader("Search Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Search Time", f"{search_time:.2f}s")
                    
                    with col2:
                        avg_similarity = np.mean([sim for _, sim, _ in results])
                        st.metric("Average Similarity", f"{avg_similarity:.3f}")
                    
                    with col3:
                        best_similarity = results[0][1] if results else 0
                        st.metric("Best Match", f"{best_similarity:.3f}")
                
                else:
                    st.warning("No similar images found. Try with a different image.")
            
            except Exception as e:
                st.error(f"Error during search: {e}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this system:**
    - Uses ORB (Oriented FAST and Rotated BRIEF) for feature extraction
    - Computes similarity using Hamming distance
    - Groups images using k-Means clustering
    - Built with Streamlit for interactive web interface
    """)

if __name__ == "__main__":
    main()