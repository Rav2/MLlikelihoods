import pandas as pd
import glob
import argparse
import os
import json
import warnings

def merge_csv_files(input_patterns, output_file):
    all_files = []
    
    # Collect all matching CSV files
    for pattern in input_patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        print("No CSV files found matching the given patterns.")
        return None
    
    # Read and append all CSV files
    dataframes = [pd.read_csv(file) for file in all_files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Check for and remove duplicates
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    final_rows = len(merged_df)
    duplicates_removed = initial_rows - final_rows
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate row(s)")
    else:
        print("No duplicates found")
    
    # Save merged DataFrame to disk
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file} ({final_rows} rows)")
    
    return all_files

def merge_json_metadata(csv_files, output_json):
    metadata_files = [file.replace('.csv', '.json') for file in csv_files]
    
    merged_metadata = {}
    starting_points = []
    # Initialize as None; they will be replaced by lists from the first metadata file
    global_extrema = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    min_values = {"nLL_exp_max": [None, float('inf')], "nLLA_exp_max": [None, float('inf')],
                  "nLL_obs_max": [None, float('inf')], "nLLA_obs_max": [None, float('inf')]}
    reference_metadata = None
    
    for meta_file in metadata_files:
        if not os.path.exists(meta_file):
            warnings.warn(f"Metadata file {meta_file} not found.")
            continue
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Concatenate "starting_points"
        starting_points.extend(metadata.get("starting_points", []))
        
        # Update global extrema elementwise for x and y keys
        for key in ["x_min", "x_max", "y_min", "y_max"]:
            values = metadata.get(key)
            if values and isinstance(values, list):
                if global_extrema[key] is None:
                    # Use a copy of the list from the first metadata file
                    global_extrema[key] = values.copy()
                else:
                    if key.endswith("_min"):
                        # Compute elementwise minimum
                        global_extrema[key] = [min(old, new) for old, new in zip(global_extrema[key], values)]
                    else:
                        # Compute elementwise maximum
                        global_extrema[key] = [max(old, new) for old, new in zip(global_extrema[key], values)]
        
        # Find global min values (since stored values are negative log likelihood)
        for key in min_values.keys():
            if key in metadata and isinstance(metadata[key], list) and len(metadata[key]) == 2:
                arg, val = metadata[key]
                if val < min_values[key][1]:
                    min_values[key] = [arg, val]
        
        # Check for consistency in other parameters
        if reference_metadata is None:
            reference_metadata = metadata.copy()
        else:
            for key, value in metadata.items():
                if key not in ["starting_points", "merged"] + list(global_extrema.keys()) + list(min_values.keys()):
                    if key != 'seed' and reference_metadata.get(key) != value:
                        warnings.warn(f"Inconsistent value for {key}: {reference_metadata.get(key)} vs {value}")
    
    # Construct merged metadata using the first metadata as reference
    merged_metadata.update(reference_metadata)
    merged_metadata["starting_points"] = starting_points
    merged_metadata.update(global_extrema)
    merged_metadata.update({key: val for key, val in min_values.items()})
    merged_metadata["merged"] = True
    
    with open(output_json, 'w') as f:
        json.dump(merged_metadata, f, indent=4)
    print(f"Merged metadata saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple CSV and JSON metadata files into one.")
    parser.add_argument("input_patterns", nargs="+", help="Wildcard patterns to locate CSV files (e.g., 'data/*.csv')")
    parser.add_argument("-o", "--output", required=True, help="Base name for output files (CSV and JSON)")
    
    args = parser.parse_args()
    merged_csv_file = args.output + ".csv"
    merged_json_file = args.output + ".json"
    
    csv_files = merge_csv_files(args.input_patterns, merged_csv_file)
    if csv_files:
        merge_json_metadata(csv_files, merged_json_file)