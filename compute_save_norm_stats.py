"""requires tensorflow 2.15
works with all open-x datasets and custom RLDS datasets.
if you are using new datasets, you may need to update the get_oxe_dataset_kwargs_and_weights function.

example usage:
# For Open-X datasets:
python compute_save_norm_stats.py --dataset_path open-x --output_dir norm_stats_ucsd --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds

# For custom RLDS datasets:
python compute_save_norm_stats.py --dataset_path /path/to/custom_dataset --output_dir norm_stats_custom --dataset_name my_custom_dataset --custom_dataset

if using custom datasets, ensure they are in the RLDS format.
and update the dataset_name accordingly.

Usage Example:
For Open-X datasets:
python compute_save_norm_stats.py \
  --dataset_path /path/to/open-x \
  --output_dir /norm_stats_ucsd \
  --dataset_name ucsd_pick_and_place_dataset_converted_externally_to_rlds

For custom RLDS datasets:
# Inspect dataset structure first
python compute_save_norm_stats.py \
  --dataset_path /path/to/my_custom_dataset \
  --dataset_name my_custom_dataset \
  --custom_dataset \
  --inspect_only

# Compute statistics
python compute_save_norm_stats.py \
  --dataset_path /path/to/my_custom_dataset \
  --output_dir /norm_stats_custom \
  --dataset_name my_custom_dataset \
  --custom_dataset
"""

import argparse
import json
import os
from pathlib import Path
import tensorflow as tf
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from transformers import AutoTokenizer
import numpy as np
from PIL import Image
import torch

def get_custom_dataset_config(dataset_name: str) -> dict:
    """
    Returns a default configuration for custom RLDS datasets.
    Users can modify this function to match their dataset structure.
    """
    # Default configuration - modify as needed for your dataset
    return {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": "pos_euler",  # or "pos_quat", "joint", "none"
        "action_encoding": "eef_pos",   # or "joint_pos"
    }

def inspect_dataset_structure(dataset_path: str, dataset_name: str):
    """
    Inspects the structure of a custom RLDS dataset to help users configure it properly.
    """
    import tensorflow_datasets as tfds
    
    print(f"\n=== Inspecting Dataset Structure: {dataset_name} ===")
    
    try:
        builder = tfds.builder(dataset_name, data_dir=dataset_path)
        print(f"Dataset info: {builder.info}")
        
        # Get a sample episode to inspect structure
        ds = builder.as_dataset(split='train', shuffle_files=False).take(1)
        
        for episode in ds:
            print(f"\nEpisode keys: {list(episode.keys())}")
            
            if 'steps' in episode:
                for step in episode['steps'].take(1):
                    print(f"Step keys: {list(step.keys())}")
                    
                    if 'observation' in step:
                        obs = step['observation']
                        print(f"Observation keys: {list(obs.keys())}")
                        
                        # Check image keys
                        image_keys = [k for k in obs.keys() if 'image' in k.lower() or 'rgb' in k.lower()]
                        if image_keys:
                            print(f"Image observation keys: {image_keys}")
                        
                        # Check state/proprioception keys
                        state_keys = [k for k in obs.keys() if any(x in k.lower() for x in ['state', 'pose', 'joint', 'eef', 'gripper'])]
                        if state_keys:
                            print(f"State observation keys: {state_keys}")
                    
                    if 'action' in step:
                        action = step['action']
                        print(f"Action shape: {action.shape}")
                        print(f"Action dtype: {action.dtype}")
                    
                    # Check for language instructions
                    lang_keys = [k for k in step.keys() if 'language' in k.lower() or 'instruction' in k.lower() or 'task' in k.lower()]
                    if lang_keys:
                        print(f"Language keys: {lang_keys}")
            break
            
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        print("Make sure the dataset is in proper RLDS format.")

def compute_and_save_stats(dataset_path, output_dir, dataset_name="ucsd_pick_and_place_dataset_converted_externally_to_rlds", custom_dataset=False, inspect_only=False):
    """
    Compute dataset statistics using OpenVLA's existing functions and save to specified directory.
    
    Args:
        dataset_path: Path to your RLDS dataset directory
        output_dir: Directory where to save dataset_statistics.json
        dataset_name: Name identifier for the dataset (should match OXE config key or custom dataset name)
        custom_dataset: Whether this is a custom dataset (not in OXE)
        inspect_only: If True, only inspect the dataset structure without computing stats
    """
    print(f"Starting computation with dataset_path={dataset_path}, output_dir={output_dir}, dataset_name={dataset_name}")
    print(f"Custom dataset: {custom_dataset}")
    
    # If inspect_only, just show dataset structure
    if inspect_only:
        inspect_dataset_structure(dataset_path, dataset_name)
        return
    
    # Configure TensorFlow (same as OpenVLA)
    tf.config.set_visible_devices([], "GPU")
    print("TensorFlow configured")
    
    # Check if dataset is registered in OXE configs
    if not custom_dataset and dataset_name not in OXE_DATASET_CONFIGS:
        print(f"Warning: {dataset_name} not found in OXE_DATASET_CONFIGS")
        print(f"Available OXE datasets: {list(OXE_DATASET_CONFIGS.keys())}")
        print("Consider using --custom_dataset flag for non-OXE datasets")
        
        # Auto-detect if it should be treated as custom
        custom_dataset = True
        print("Automatically treating as custom dataset...")
    
    # Create minimal components needed for RLDSDataset
    print("Loading tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="right")
    base_tokenizer.add_special_tokens({"pad_token": "<pad>"})
    action_tokenizer = ActionTokenizer(base_tokenizer)
    print("Tokenizer loaded")
    
    # Simple image transform (placeholder)
    def dummy_image_transform(image):
        return torch.zeros(3, 224, 224)
    
    # Create batch transform (needed by RLDSDataset but not used for statistics)
    print("Creating batch transform...")
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=base_tokenizer,
        image_transform=dummy_image_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    print("Batch transform created")
    
    # For custom datasets, we'll use direct TFDS approach since they're not in OXE registry
    if custom_dataset:
        print("Processing custom RLDS dataset...")
        
        # First, inspect the dataset structure to help users
        inspect_dataset_structure(dataset_path, dataset_name)
        
        # Use direct TFDS approach for custom datasets
        import tensorflow_datasets as tfds
        
        # Set TFDS data dir
        os.environ['TFDS_DATA_DIR'] = str(dataset_path)
        
        try:
            print(f"Building TFDS dataset for {dataset_name}...")
            builder = tfds.builder(dataset_name, data_dir=dataset_path)
            ds = builder.as_dataset(split='train', shuffle_files=False)
            
            # Compute basic statistics manually
            print("Computing statistics manually...")
            print("Processing all episodes in the dataset...")
            actions = []
            episode_count = 0
            total_transitions = 0
            
            # Process ALL episodes
            for episode in ds:
                episode_count += 1
                episode_transitions = 0
                
                for step in episode['steps']:
                    action = step['action'].numpy()
                    actions.append(action)
                    episode_transitions += 1
                    total_transitions += 1
                
                # Print progress every 100 episodes
                if episode_count % 100 == 0:
                    print(f"Processed {episode_count} episodes, {total_transitions} transitions")
            
            print(f"Finished processing {episode_count} episodes, {total_transitions} total transitions")
            
            actions = np.array(actions)
            action_dim = actions.shape[1]
            print(f"Action dimensions: {action_dim}")
            
            # For custom datasets, use a general approach for action mask
            # Users can modify this based on their action space
            if action_dim == 7:
                # Standard 7D: pos (3) + ori (3) + gripper (1)
                action_mask = [True, True, True, True, True, True, False]
            elif action_dim == 4:
                # Common 4D: pos (3) + gripper (1)
                action_mask = [True, True, True, False]
            else:
                # Default: assume last dimension is gripper/discrete action
                action_mask = [True] * (action_dim - 1) + [False]
            
            print(f"Using action mask: {action_mask}")
            
            # Compute statistics
            dataset_statistics = {
                dataset_name: {
                    "action": {
                        "mean": actions.mean(axis=0).tolist(),
                        "std": actions.std(axis=0).tolist(),
                        "min": actions.min(axis=0).tolist(),
                        "max": actions.max(axis=0).tolist(),
                        "q01": np.percentile(actions, 1, axis=0).tolist(),
                        "q99": np.percentile(actions, 99, axis=0).tolist(),
                        "mask": action_mask
                    },
                    "proprio": {
                        "mean": [0.0] * action_dim,  # Match action dimensionality
                        "std": [0.0] * action_dim,
                        "min": [0.0] * action_dim,
                        "max": [0.0] * action_dim,
                        "q01": [0.0] * action_dim,
                        "q99": [0.0] * action_dim,
                    },
                    "num_transitions": total_transitions,
                    "num_trajectories": episode_count,
                }
            }
            print("Successfully computed statistics for custom dataset")
            
        except Exception as e:
            print(f"Error processing custom dataset: {e}")
            raise
    
    else:
        # Use OpenVLA's standard pipeline for OXE datasets
        print("Processing Open-X dataset...")
        
        # Check if dataset exists in the expected structure
        print(f"Checking for dataset at {dataset_path}...")
        dataset_full_path = Path(dataset_path) / dataset_name
        if not dataset_full_path.exists():
            print(f"Dataset not found at {dataset_full_path}")
            print(f"Available datasets in {dataset_path}:")
            if Path(dataset_path).exists():
                for item in Path(dataset_path).iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
            
            # Try to find the dataset with a different name
            possible_names = [
                "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
            ]
            
            found_dataset = None
            for name in possible_names:
                test_path = Path(dataset_path) / name
                if test_path.exists():
                    found_dataset = name
                    dataset_name = name
                    break
            
            if not found_dataset:
                raise FileNotFoundError(f"Could not find dataset in {dataset_path}")
            else:
                print(f"Found dataset: {found_dataset}")
        else:
            print(f"Dataset found at {dataset_full_path}")
        
        # Create RLDSDataset - this will compute statistics during initialization
        try:
            print(f"Creating RLDSDataset with data_root_dir={dataset_path}, data_mix={dataset_name}")
            dataset = RLDSDataset(
                data_root_dir=Path(dataset_path),
                data_mix=dataset_name,
                batch_transform=batch_transform,
                resize_resolution=(224, 224),
                shuffle_buffer_size=1000,
                train=True,
                image_aug=False,
            )
            
            # Extract dataset statistics
            dataset_statistics = dataset.dataset_statistics
            print("Successfully created dataset and extracted statistics")
            
        except Exception as e:
            print(f"Error creating RLDSDataset: {e}")
            print("Trying alternative approach...")
            
            # Alternative: try direct dataset creation if the above fails
            from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
            from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
            from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
            
            # Create mixture spec for single dataset
            mixture_spec = [(dataset_name, 1.0)]
            
            try:
                print("Trying get_oxe_dataset_kwargs_and_weights...")
                per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
                    Path(dataset_path),
                    mixture_spec,
                    load_camera_views=("primary",),
                    load_depth=False,
                    load_proprio=False,
                    load_language=True,
                    action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
                )
                
                rlds_config = dict(
                    traj_transform_kwargs=dict(
                        window_size=1,
                        future_action_window_size=0,
                        skip_unlabeled=True,
                        goal_relabeling_strategy="uniform",
                    ),
                    frame_transform_kwargs=dict(
                        resize_size=(224, 224),
                        num_parallel_calls=1,
                    ),
                    dataset_kwargs_list=per_dataset_kwargs,
                    shuffle_buffer_size=1000,
                    sample_weights=weights,
                    balance_weights=True,
                    traj_transform_threads=1,
                    traj_read_threads=1,
                    train=True,
                )
                
                print("Creating interleaved dataset...")
                dataset, dataset_length, dataset_statistics = make_interleaved_dataset(**rlds_config)
                print("Successfully created dataset using alternative approach")
                
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                print("Falling back to direct TFDS approach...")
                
                # Fallback to the same approach as custom datasets
                import tensorflow_datasets as tfds
                
                # Set TFDS data dir
                os.environ['TFDS_DATA_DIR'] = str(dataset_path)
                
                try:
                    print(f"Building TFDS dataset for {dataset_name}...")
                    builder = tfds.builder(dataset_name, data_dir=dataset_path)
                    ds = builder.as_dataset(split='train', shuffle_files=False)
                    
                    # Compute basic statistics manually
                    print("Computing statistics manually...")
                    print("Processing all episodes in the dataset...")
                    actions = []
                    episode_count = 0
                    total_transitions = 0
                    
                    # Process ALL episodes
                    for episode in ds:
                        episode_count += 1
                        episode_transitions = 0
                        
                        for step in episode['steps']:
                            action = step['action'].numpy()
                            actions.append(action)
                            episode_transitions += 1
                            total_transitions += 1
                        
                        # Print progress every 100 episodes
                        if episode_count % 100 == 0:
                            print(f"Processed {episode_count} episodes, {total_transitions} transitions")
                    
                    print(f"Finished processing {episode_count} episodes, {total_transitions} total transitions")
                    
                    actions = np.array(actions)
                    
                    # Determine action dimensionality and create appropriate mask
                    action_dim = actions.shape[1]
                    print(f"Original action dimensions: {action_dim}")
                    
                    # For UCSD pick and place, we need to pad to 7 dimensions if it's only 4
                    if action_dim == 4:
                        print("Padding 4D actions to 7D (3D pos + 3D ori + gripper)")
                        # Current: [x, y, z, gripper]
                        # Target:  [x, y, z, rx, ry, rz, gripper]
                        
                        # Pad with zeros for orientation (dimensions 3, 4, 5)
                        padded_actions = np.zeros((actions.shape[0], 7))
                        padded_actions[:, 0:3] = actions[:, 0:3]  # position
                        padded_actions[:, 6] = actions[:, 3]      # gripper
                        # Dimensions 3, 4, 5 remain zero (orientation)
                        
                        actions = padded_actions
                        action_dim = 7
                        
                        # Standard 7D mask: pos (3) + ori (3) + gripper (1)
                        action_mask = [True, True, True, True, True, True, False]
                        
                    elif action_dim == 7:
                        # Already 7D
                        action_mask = [True, True, True, True, True, True, False]
                    else:
                        # Default: assume last dimension is gripper
                        action_mask = [True] * (action_dim - 1) + [False]
                    
                    print(f"Final action dimensions: {action_dim}")
                    
                    # Compute statistics
                    dataset_statistics = {
                        dataset_name: {
                            "action": {
                                "mean": actions.mean(axis=0).tolist(),
                                "std": actions.std(axis=0).tolist(),
                                "min": actions.min(axis=0).tolist(),
                                "max": actions.max(axis=0).tolist(),
                                "q01": np.percentile(actions, 1, axis=0).tolist(),
                                "q99": np.percentile(actions, 99, axis=0).tolist(),
                                "mask": action_mask
                            },
                            "proprio": {
                                "mean": [0.0] * action_dim,  # Match action dimensionality
                                "std": [0.0] * action_dim,
                                "min": [0.0] * action_dim,
                                "max": [0.0] * action_dim,
                                "q01": [0.0] * action_dim,
                                "q99": [0.0] * action_dim,
                            },
                            "num_transitions": total_transitions,
                            "num_trajectories": episode_count,
                        }
                    }
                    print("Successfully computed statistics manually")
                    
                except Exception as e3:
                    print(f"All approaches failed. Final error: {e3}")
                    raise
    
    # Save statistics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save using OpenVLA's save function
    save_dataset_statistics(dataset_statistics, output_path)
    
    print(f"Statistics saved to {output_path / 'dataset_statistics.json'}")
    
    # Print summary for the specific dataset
    if dataset_name in dataset_statistics:
        stats = dataset_statistics[dataset_name]
        action_stats = stats["action"]
        print(f"\nAction Statistics Summary for {dataset_name}:")
        print(f"  Dimensions: {len(action_stats['mean'])}")
        print(f"  Mean: {action_stats['mean']}")
        print(f"  Std: {action_stats['std']}")
        print(f"  Min: {action_stats['min']}")
        print(f"  Max: {action_stats['max']}")
        print(f"  Q01: {action_stats['q01']}")
        print(f"  Q99: {action_stats['q99']}")
        
        if "mask" in action_stats:
            print(f"  Mask: {action_stats['mask']}")
        
        if "num_trajectories" in stats:
            print(f"  Number of trajectories: {stats['num_trajectories']}")
        if "num_transitions" in stats:
            print(f"  Number of transitions: {stats['num_transitions']}")
    else:
        print(f"Available datasets in statistics: {list(dataset_statistics.keys())}")
    
    return dataset_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset normalization statistics for OpenVLA")
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to root directory containing datasets (for OXE) or path to custom RLDS dataset")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for dataset_statistics.json")
    parser.add_argument("--dataset_name", type=str, 
                       default="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
                       help="Dataset name (should match directory name for OXE or dataset name for custom RLDS)")
    parser.add_argument("--custom_dataset", action="store_true",
                       help="Use this flag for custom RLDS datasets (not in Open-X)")
    parser.add_argument("--inspect_only", action="store_true",
                       help="Only inspect dataset structure without computing statistics")
    
    args = parser.parse_args()
    
    print("Script started")
    compute_and_save_stats(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        custom_dataset=args.custom_dataset,
        inspect_only=args.inspect_only
    )
    print("Script completed")

