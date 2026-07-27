import onnx
import sys

def remove_metadata_entry(model_path, output_path, key_to_remove="starting_points"):
    """
    Load an ONNX model, remove a specific metadata entry, and save it.
    
    Args:
        model_path: Path to the input ONNX model file
        output_path: Path where the cleaned model will be saved
        key_to_remove: The metadata key to remove (default: "starting_points")
    """
    try:
        # Load the ONNX model
        print(f"Loading model from {model_path}...")
        model = onnx.load(model_path)
        
        # Check current metadata
        metadata_props = model.metadata_props
        print(f"Found {len(metadata_props)} metadata entries")
        
        # List current metadata keys
        current_keys = [prop.key for prop in metadata_props]
        print(f"Current metadata keys: {current_keys}")
        
        # Remove the specified key if it exists
        if key_to_remove in current_keys:
            new_metadata = [prop for prop in metadata_props if prop.key != key_to_remove]
            del model.metadata_props[:]
            model.metadata_props.extend(new_metadata)
            print(f"Removed '{key_to_remove}' from metadata")
        else:
            print(f"Key '{key_to_remove}' not found in metadata")
        
        # Save the cleaned model
        print(f"Saving cleaned model to {output_path}...")
        onnx.save(model, output_path)
        print(f"Successfully saved cleaned model")
        
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_model.onnx> [output_model.onnx]")
        print("Example: python script.py model.onnx model_cleaned.onnx")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else input_model.replace(".onnx", "_cleaned.onnx")
    
    remove_metadata_entry(input_model, output_model)