import json
import numpy as np 

def calculate_total_background(pyhf_data_file):
    """
    Calculate and print the total background for each channel in pyhf data.

    Args:
        pyhf_data_file (str): Path to the JSON file containing pyhf data.
    """
    # Load the JSON data
    with open(pyhf_data_file, 'r') as file:
        pyhf_data = json.load(file)

    # Debug: Print the keys of the JSON data
    print("Top-level keys in JSON:", pyhf_data.keys())

    # Extract the necessary fields from the JSON
    channels = pyhf_data['channels']  # Channel definitions

    # Create a dictionary to hold total background per channel
    total_background = {}

    # Loop through each channel
    for channel in channels:
        channel_name = channel['name']
        total_background[channel_name] = 0

        # Loop through the samples within this channel
        for sample in channel['samples']:
            # Assuming 'background' samples are identified by their modifiers or type
            total_background[channel_name] += sum(sample['data'])

    # Print the total background for each channel
    for channel_name, background_sum in total_background.items():
        print(f"Channel: {channel_name}, Total Background: {np.round(background_sum,1)}")
    print('*'*60)
    print('-', end='')
    for channel_name, background_sum in total_background.items():
        # if 'CR' in channel_name:
            print(f'- {np.round(background_sum,1)}')
    print('uncertainties')
    print('-', end='')
    for channel_name, background_sum in total_background.items():
        # if 'CR' in channel_name:
            print(f'- {np.round(np.sqrt(background_sum),1)}')


if __name__ == "__main__":
    # Specify the path to your pyhf JSON file
    json_file_path = "../data/1909.09226/BkgOnly.json"

    # Run the calculation
    calculate_total_background(json_file_path)
