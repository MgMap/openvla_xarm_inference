import tensorflow_datasets as tfds
import os
import sys

# Set environment variables before importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['BEAM_RUNNER'] = 'DirectRunner'

DATASET_NAME = 'ucsd_pick_and_place_dataset_converted_externally_to_rlds'
DOWNLOAD_DIR = os.path.expanduser('~/tensorflow_datasets')

print(f"Attempting to download {DATASET_NAME} to {DOWNLOAD_DIR}")

try:
    # Try with minimal beam options
    builder = tfds.builder(DATASET_NAME, data_dir=DOWNLOAD_DIR)
    
    # Use the most basic download config possible
    download_config = tfds.download.DownloadConfig(
        beam_options={'runner': 'DirectRunner'},
        try_download_gcs=True
    )
    
    builder.download_and_prepare(download_config=download_config)
    print("Download successful!")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("Trying alternative approach...")
    
    # Alternative: try to load directly without explicit download
    try:
        ds = tfds.load(
            DATASET_NAME, 
            data_dir=DOWNLOAD_DIR, 
            download=True,
            try_gcs=True,
            shuffle_files=False
        )
        print("Alternative download successful!")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")