import os
import sys
import tensorflow_datasets as tfds

""" requires gcloud auth application-default login 
   and gsutil to be installed and configured.
   If you haven't done this, run:
   gcloud auth application-default login
   gsutil config
   and then download the dataset using:
gsutil -m cp -r gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/0.1.0 /home/slurmlab/Min/openvla/ucsd_pick_place/"""

# Remove the GCS disabling since we now have auth
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Increase recursion limit for complex datasets
sys.setrecursionlimit(10000)

# Expand the path
DOWNLOAD_DIR = os.path.expanduser('~/tensorflow_datasets')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

dataset_name = "ucsd_pick_and_place_dataset_converted_externally_to_rlds"

print(f"Attempting to download {dataset_name}...")

try:
    print("Downloading dataset...")
    ds = tfds.load(
        dataset_name,
        data_dir=DOWNLOAD_DIR,
        download=True,  # This will download if not already available
        as_supervised=False,
        shuffle_files=False
    )
    print("✓ Successfully downloaded and loaded dataset")
    
    # Print some basic info about the dataset
    print(f"Dataset structure: {ds}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nIf authentication failed, make sure you've run:")
    print("1. source ~/.bashrc")
    print("2. gcloud auth application-default login")