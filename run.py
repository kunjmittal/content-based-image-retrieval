"""
Simple script to run the CBIR Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("Error: app/main.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Warning: dataset directory not found!")
        print("Please create a 'dataset' directory and add some images.")
        return
    
    print("Starting Content-Based Image Retrieval System...")
    print("Opening in your default web browser...")
    
    try:
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")

if __name__ == "__main__":
    main()