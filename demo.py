"""
Demo script to test the CBIR system functionality
"""

import os
import sys
from app.orb_features import ORBFeatureExtractor
from app.retrieval import ImageRetriever
from app.clustering import ImageClusterer

def demo_system():
    """Demonstrate the CBIR system functionality"""
    
    dataset_path = "dataset"
    
    print("🔍 Content-Based Image Retrieval System Demo")
    print("=" * 50)
    
    # Check dataset
    if not os.path.exists(dataset_path):
        print("❌ Dataset directory not found!")
        return
    
    # Get image files
    image_files = []
    for file in os.listdir(dataset_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    print(f"📁 Found {len(image_files)} images in dataset")
    
    if len(image_files) == 0:
        print("❌ No images found in dataset!")
        return
    
    # Initialize components
    print("\n🔧 Initializing system components...")
    extractor = ORBFeatureExtractor(n_features=500)
    
    # Extract features
    print("🔍 Extracting ORB features...")
    features_dict = extractor.extract_features_from_dataset(dataset_path)
    
    if len(features_dict) == 0:
        print("❌ No features could be extracted!")
        return
    
    print(f"✅ Extracted features from {len(features_dict)} images")
    
    # Perform clustering
    print("\n🎯 Performing k-Means clustering...")
    clusterer = ImageClusterer(n_clusters=min(5, len(features_dict)))
    image_clusters = clusterer.fit_clustering(features_dict)
    
    # Initialize retriever
    print("\n🔎 Initializing image retriever...")
    retriever = ImageRetriever(features_dict, dataset_path)
    
    # Demo retrieval with first image
    test_image = image_files[0]
    test_image_path = os.path.join(dataset_path, test_image)
    
    print(f"\n🖼️  Testing retrieval with: {test_image}")
    results = retriever.retrieve_similar_images(test_image_path, top_k=5)
    
    if results:
        print(f"✅ Found {len(results)} similar images:")
        for i, (img_name, similarity, img_path) in enumerate(results, 1):
            cluster_id = clusterer.get_cluster_id(img_name)
            print(f"  {i}. {img_name} - Similarity: {similarity:.3f} - Cluster: {cluster_id}")
    else:
        print("❌ No similar images found")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📊 System Statistics:")
    print(f"   - Total images: {len(image_files)}")
    print(f"   - Images with features: {len(features_dict)}")
    print(f"   - Number of clusters: {clusterer.n_clusters}")
    print(f"\n🚀 Ready to run the web interface!")
    print(f"   Run: streamlit run app/main.py")

if __name__ == "__main__":
    demo_system()