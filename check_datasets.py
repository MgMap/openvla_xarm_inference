import tensorflow_datasets as tfds

# List all available datasets
available_datasets = tfds.list_builders()
print(f"Total available datasets: {len(available_datasets)}")

# Check if the dataset exists with a similar name
target_dataset = 'ucsd_pick_and_place_dataset_converted_externally_to_rlds'
similar_datasets = [name for name in available_datasets if 'ucsd' in name.lower() or 'pick_and_place' in name.lower()]

print(f"\nDatasets containing 'ucsd' or 'pick_and_place':")
for dataset in similar_datasets:
    print(f"  - {dataset}")

if target_dataset in available_datasets:
    print(f"\n✓ {target_dataset} is available")
else:
    print(f"\n✗ {target_dataset} is not available")