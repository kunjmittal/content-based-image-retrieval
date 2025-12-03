# content-based-image-retrieval
Application-level image IR project using ORB (out-of-syllabus algorithm), k-Means, and Streamlit UI.


Content-Based Image Retrieval System Using ORB Features & k-Means Clustering
🎯 Out-of-Syllabus Multimedia IR Project — Application Level

This project implements image-based Information Retrieval, using an out-of-syllabus computer vision algorithm, ORB (Oriented FAST and Rotated BRIEF), along with k-Means clustering and a Streamlit UI.

It satisfies the requirement of:

✔ Using media (Images instead of Text)

✔ Using an algorithm OUTSIDE the IR syllabus

✔ Applying IR concepts (feature extraction + similarity search)

✔ Clustering

✔ Building an application/UI

📌 1. Project Overview

The system allows a user to upload an image, and retrieves visually similar images from a dataset using:

ORB feature extraction (Out of syllabus)

Hamming distance similarity

k-Means clustering

Web UI using Streamlit

This is a complete Application-Level IR System.

📌 2. System Workflow
1. Dataset Preparation

A folder of images is placed in:

/dataset/

2. Feature Extraction (Out of Syllabus Algorithm – ORB)

Compute ORB keypoints & descriptors

Store feature vectors for each image

Build an index

3. Retrieval

When user uploads a query image:

Extract ORB features

Compare with dataset images

Compute similarity → Hamming distance

Return Top 5/10 matching images

4. Clustering (k-Means)

Convert descriptors into vector form

Apply k-Means

Group images into clusters

Show cluster ID in results

5. UI (Streamlit Application)

Upload image

Click “Search Similar Images”

Show:

Query image

Retrieved similar images

Cluster ID

📌 3. Folder Structure
cbir-orb-kmeans/
│
├── dataset/       → contains sample images
│
├── app/
│     ├── main.py          → Streamlit UI
│     ├── orb_features.py  → ORB feature extraction
│     ├── retrieval.py     → similarity search
│     ├── clustering.py    → k-Means clustering
│     └── utils.py
│
└── README.md

📌 4. Technologies Used

Python

OpenCV (ORB Features)

NumPy

scikit-learn (k-Means)

Streamlit (UI)

📌 5. Status

🚧 Code development in progress
💡 Repo created so that implementation can be added

