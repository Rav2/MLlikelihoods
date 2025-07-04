import json
import sys

def extract_channels_and_bins(json_path):
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check if the structure has "channels" key
        if "channels" not in data:
            print("Error: JSON file does not contain 'channels' key.")
            return

        total_bins = 0
        total_channels = 0

        # Extract and print channel names and number of data entries
        print("Channels and number of bins:")
        for channel in data["channels"]:
            channel_name = channel.get("name", "Unnamed Channel")
            samples = channel.get("samples", [])

            # Count the total number of bins (data entries) in the channel
            bin_count = len(samples[0].get("data", []))
            total_bins += bin_count
            total_channels += 1

            print(f"- Channel: {channel_name}, Number of bins: {bin_count}")

        print(f"\nTotal number of channels: {total_channels}")
        print(f"Total number of bins: {total_bins}")

    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
    except json.JSONDecodeError:
        print(f"Error: File is not a valid JSON: {json_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
    else:
        json_path = sys.argv[1]
        extract_channels_and_bins(json_path)
