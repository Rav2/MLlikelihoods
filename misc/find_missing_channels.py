import json
import pyhf
import matplotlib.pyplot as plt
from collections import Counter

def normalize_path(path):
    return path.strip()

def clean_path(path):
    return ''.join(c for c in path if not c.isspace())

def canonicalize_path(path):
    # Example of normalization, customize as needed
    return path.lower().strip()


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_paths_from_model(model_data):
    """
    Extract paths from the background model and map them to channel names.
    """
    workspace = pyhf.Workspace(model_data)
    model_paths = {}
    for c, channel in enumerate(workspace.channels):
        # Use only the channel identifier
        model_paths[f"/channels/{c}/"] = channel
    return model_paths

def extract_channel_from_path(path):
    """
    Extract the channel part of the path.
    """
    parts = path.split('/')
    if len(parts) >= 3 and parts[1] == 'channels':
        return f"/channels/{parts[2]}/"
    return None

def extract_paths_from_patches(patch_data):
    """
    Extract paths from all patches in the patches list.
    """
    if 'patches' in patch_data and patch_data['patches']:
        all_patch_paths = []
        for patch in patch_data['patches']:
            patch_paths = [extract_channel_from_path(entry['path']) for entry in patch['patch']]
            # Remove None values and duplicates
            patch_paths = list(filter(None, patch_paths))
            all_patch_paths.append(patch_paths)
        return all_patch_paths
    else:
        return []

def extract_paths_from_first_patch(patch_data):
    """
    Extract paths from the first patch in the patches list.
    """
    if 'patches' in patch_data and patch_data['patches']:
        first_patch = patch_data['patches'][0]['patch']
        patch_paths = [extract_channel_from_path(entry['path']) for entry in first_patch]
        # Remove None values and duplicates
        patch_paths = list(filter(None, patch_paths))
        return patch_paths
    else:
        return []

def compare_paths(model_paths, patch_paths):
    """
    Compare path lists and return missing paths in the patch.
    """
    # Normalize paths
    model_path_set = set(normalize_path(path) for path in model_paths.keys())
    patch_path_set = set(normalize_path(path) for path in patch_paths)    
    missing_paths = model_path_set - patch_path_set
    # model_path_list = sorted(list(model_path_set))
    # patch_path_list = sorted(list(patch_path_set))
    # for a in model_path_list: print(a)
    # print()
    # for b in patch_path_list:print(b)
    return list(missing_paths)

def main(model_json_path, patch_json_path):
    # Load JSON files
    model_data = load_json(model_json_path)
    patch_data = load_json(patch_json_path)

    # Extract paths from the background model and map them to channel names
    model_paths = extract_paths_from_model(model_data)

    # Extract paths from all patches
    all_patch_paths = extract_paths_from_patches(patch_data)

    # Extract paths from the first patch
    first_patch_paths = extract_paths_from_first_patch(patch_data)

    # Compare and print missing paths for the first patch
    missing_paths_first_patch = compare_paths(model_paths, first_patch_paths)
    if missing_paths_first_patch:
        missing_channels_first_patch = [model_paths[path] for path in missing_paths_first_patch]
        missing_channels_names = '\n'.join(missing_channels_first_patch)
        print(f"Missing channels in the first patch:\n{missing_channels_names}")
    else:
        print("No paths are missing in the first patch.")
    # Count missing paths across all patches
    missing_path_counts = Counter()
    all_missing_channels = []
    for patch_paths in all_patch_paths:
        missing_paths = compare_paths(model_paths, patch_paths)
        if missing_paths:
            missing_channels = [model_paths[path] for path in missing_paths]
            all_missing_channels.append(set(missing_channels))
            missing_path_counts.update(missing_channels)

    # Check if all patches are missing the same channels
    unique_missing_sets = [frozenset(missing) for missing in all_missing_channels]
    all_same = all(x == unique_missing_sets[0] for x in unique_missing_sets)
    
    if all_same:
        print("All patches are missing the same channels.")
    else:
        print("Different patches are missing different channels.")

    # Plot the histogram of the top 10 missing channels
    if missing_path_counts:
        top_missing_channels = missing_path_counts.most_common(30)
        channels, counts = zip(*top_missing_channels)
        
        plt.figure(figsize=(10, 6))
        plt.barh(channels, counts, color='skyblue')
        plt.xlabel('Number of Missing Paths')
        plt.ylabel('Channel Names')
        plt.title('Top Channels with Missing Paths')
        plt.gca().invert_yaxis()  # To display the highest count at the top
        plt.tight_layout()
        plt.show()
    else:
        print("No paths are missing across all patches.")

    print(f'Number of all patches: {len(patch_data["patches"])}')

# Replace with your actual JSON file paths
model_json_path = '../data/1911.12606/EWKinos_bkgonly.json'
patch_json_path = '../data/1911.12606/EWKinos_patchset.json'

main(model_json_path, patch_json_path)
