import argparse
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
from pathlib import Path


class DataReader(CalibrationDataReader):
    """Custom calibration data reader for static quantization."""
    
    def __init__(self, input_data, input_name, batch_size=32):
        self.input_data = input_data
        self.input_name = input_name
        self.batch_size = batch_size
        self.current_index = 0
    
    def get_next(self):
        if self.current_index >= len(self.input_data):
            return None
        
        end_index = min(self.current_index + self.batch_size, len(self.input_data))
        batch = self.input_data[self.current_index:end_index].astype(np.float32)
        self.current_index = end_index
        
        return {self.input_name: batch}


def load_and_prepare_data(csv_path, onnx_model_path):
    """Load CSV data, process it according to spec, and return calibration input."""
    
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    
    print("Processing negative log likelihood columns...")
    # Subtract mu=0 from mu=1
    df.iloc[:, -7] = df.iloc[:, -7] - df.iloc[:, -8]
    df.iloc[:, -5] = df.iloc[:, -5] - df.iloc[:, -6]
    df.iloc[:, -3] = df.iloc[:, -3] - df.iloc[:, -4]
    df.iloc[:, -1] = df.iloc[:, -1] - df.iloc[:, -2]
    
    cols_to_drop = [df.columns[col] for col in [-8, -6, -4, -2]]
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    df.rename(columns={
        'nLL_exp_mu1': 'Delta_nLL_exp',
        'nLL_obs_mu1': 'Delta_nLL_obs',
        'nLLA_exp_mu1': 'Delta_nLLA_exp',
        'nLLA_obs_mu1': 'Delta_nLLA_obs',
    }, inplace=True)
    
    print(f"Processed data shape: {df.shape}")
    
    # Load standardization parameters from ONNX metadata
    print("Loading standardization parameters from ONNX model metadata...")
    model = onnx.load(onnx_model_path)
    
    metadata = {item.key: item.value for item in model.metadata_props}
    
    if 'standardization_mean' not in metadata or 'standardization_std' not in metadata:
        raise ValueError("standardization_mean and/or standardization_std not found in ONNX metadata")
    
    means = np.array(eval(metadata['standardization_mean']), dtype=np.float32)
    stds = np.array(eval(metadata['standardization_std']), dtype=np.float32)
    
    # Identify input and target columns
    delta_cols = ['Delta_nLL_exp', 'Delta_nLL_obs', 'Delta_nLLA_exp', 'Delta_nLLA_obs']
    input_cols = [col for col in df.columns if col not in delta_cols]
    
    print(f"Input columns: {len(input_cols)}, Target columns: {len(delta_cols)}")
    
    # Standardize input features
    X = df[input_cols].values.astype(np.float32)
    X_standardized = (X - means[:len(input_cols)]) / stds[:len(input_cols)]
    
    # Standardize targets
    y = df[delta_cols].values.astype(np.float32)
    y_standardized = (y - means[len(input_cols):]) / stds[len(input_cols):]
    
    return X_standardized, y_standardized, means, stds


def remove_metadata_entry(onnx_model_path, output_path):
    """Remove 'starting_points' from top-level metadata."""
    print("Removing 'starting_points' metadata entry...")
    model = onnx.load(onnx_model_path)
    
    metadata_to_keep = [item for item in model.metadata_props if item.key != 'starting_points']
    del model.metadata_props[:]
    for item in metadata_to_keep:
        model.metadata_props.append(item)
    
    onnx.save(model, output_path)
    print(f"Metadata cleaned. Saved to: {output_path}")


def convert_to_float16(onnx_model_path, output_path):
    """Convert entire model to float16 for 2x size reduction with automatic type conversion."""
    
    print("\nConverting model to float16...")
    
    model = onnx.load(onnx_model_path)
    
    from onnx import TensorProto, numpy_helper
    from onnx.helper import make_node
    
    # Convert all initializers to float16
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            weights = numpy_helper.to_array(initializer)
            weights_fp16 = weights.astype(np.float16)
            initializer.CopyFrom(numpy_helper.from_array(weights_fp16, initializer.name))
    
    # Add Cast nodes at graph inputs to convert float32 -> float16
    print("Adding Cast nodes for input type conversion...")
    cast_nodes = []
    graph_input_names = [inp.name for inp in model.graph.input]
    
    for input_val in model.graph.input:
        if input_val.type.tensor_type.elem_type == TensorProto.FLOAT:
            # Create cast node: float32 -> float16
            cast_output_name = f"{input_val.name}_cast"
            cast_node = make_node('Cast', inputs=[input_val.name], outputs=[cast_output_name], to=TensorProto.FLOAT16)
            cast_nodes.append((cast_node, input_val.name, cast_output_name))
            
            # Change graph input type to float16
            # input_val.type.tensor_type.elem_type = TensorProto.FLOAT16
    
    # Add Cast nodes at graph outputs to convert float16 -> float32
    output_cast_nodes = []
    for output_val in model.graph.output:
        if output_val.type.tensor_type.elem_type == TensorProto.FLOAT:
            # Create cast node: float16 -> float32
            cast_input_name = f"{output_val.name}_fp16"
            cast_node = make_node('Cast', inputs=[cast_input_name], outputs=[output_val.name], to=TensorProto.FLOAT)
            output_cast_nodes.append((cast_node, output_val.name, cast_input_name))
            
            # Change graph output type to float32
            output_val.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Update first nodes' inputs to use cast outputs
    for cast_node, orig_input, cast_output in cast_nodes:
        for node in model.graph.node:
            for i, node_input in enumerate(node.input):
                if node_input == orig_input:
                    node.input[i] = cast_output
            break  # Only update first occurrence
    
    # Update last nodes' outputs for output casts
    # Find nodes that produce outputs
    output_producer_map = {}
    for node in model.graph.node:
        for output in node.output:
            output_producer_map[output] = node
    
    for cast_node, orig_output, cast_input in output_cast_nodes:
        if orig_output in output_producer_map:
            producer_node = output_producer_map[orig_output]
            for i, node_output in enumerate(producer_node.output):
                if node_output == orig_output:
                    producer_node.output[i] = cast_input
    
    # Insert cast nodes at the beginning
    for cast_node, _, _ in cast_nodes:
        model.graph.node.insert(0, cast_node)
    
    # Append output cast nodes at the end
    for cast_node, _, _ in output_cast_nodes:
        model.graph.node.append(cast_node)
    
    # Convert all value_info to float16
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
    
    onnx.save(model, output_path)
    print(f"Float16 model with automatic type conversion saved to: {output_path}")


def quantize_model(onnx_model_path, output_path, calibration_data, input_name, quant_type='dynamic'):
    """Quantize model using dynamic or static quantization."""
    
    if quant_type == 'dynamic':
        print("\nStarting dynamic quantization (int8)...")
        print("(Weights quantized to int8, activations remain float32)")
        
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"Dynamic quantization complete. Model saved to: {output_path}")
    
    elif quant_type == 'static':
        print("\nStarting static quantization (int8)...")
        print("(Using calibration data for optimal quantization)")
        
        data_reader = DataReader(calibration_data, input_name)
        
        quantize_static(
            model_input=onnx_model_path,
            model_output=output_path,
            calibration_data_reader=data_reader,
            weight_type=QuantType.QInt8,
        )
        print(f"Static quantization complete. Model saved to: {output_path}")


def validate_model(original_model_path, compressed_model_path, X_test, y_test, compression_type='float16', input_name='input_1', output_name='output_1'):
    """Validate compressed model by comparing predictions with original."""
    
    print("\n" + "="*60)
    print(f"VALIDATION: Comparing original vs {compression_type} model")
    print("="*60)
    
    try:
        # Load original model
        print("Loading original model for validation...")
        sess_original = ort.InferenceSession(original_model_path, providers=['CPUExecutionProvider'])
        
        # Load compressed model
        print("Loading compressed model...")
        sess_compressed = ort.InferenceSession(compressed_model_path, providers=['CPUExecutionProvider'])
        
        # Get predictions from original model
        print("Running inference with original model...")
        pred_original = sess_original.run([output_name], {input_name: X_test})[0]
        
        # Get predictions from compressed model
        print("Running inference with compressed model...")
        if compression_type == 'float16':
            # Float16 model now has automatic type conversion, use float32 input
            pred_compressed = sess_compressed.run([output_name], {input_name: X_test})[0]
        else:
            # Dynamic/static quantization uses float32
            pred_compressed = sess_compressed.run([output_name], {input_name: X_test})[0]
        
        # Calculate validation metrics
        mse_original = np.mean((pred_original - y_test) ** 2)
        mse_compressed = np.mean((pred_compressed - y_test) ** 2)
        
        mae_original = np.mean(np.abs(pred_original - y_test))
        mae_compressed = np.mean(np.abs(pred_compressed - y_test))
        
        # Difference between original and compressed predictions
        pred_diff = np.mean(np.abs(pred_original - pred_compressed))
        max_pred_diff = np.max(np.abs(pred_original - pred_compressed))
        
        print("\nValidation Results:")
        print(f"  Original MSE:   {mse_original:.6f}")
        print(f"  Compressed MSE: {mse_compressed:.6f}")
        print(f"  MSE increase:   {((mse_compressed / mse_original - 1) * 100):.2f}%")
        print(f"\n  Original MAE:   {mae_original:.6f}")
        print(f"  Compressed MAE: {mae_compressed:.6f}")
        print(f"  MAE increase:   {((mae_compressed / mae_original - 1) * 100):.2f}%")
        print(f"\n  Mean pred diff: {pred_diff:.6f}")
        print(f"  Max pred diff:  {max_pred_diff:.6f}")
        
        # Shape validation
        print("\nShape Validation:")
        print(f"  Input shape:    {X_test.shape}")
        print(f"  Original output shape: {pred_original.shape}")
        print(f"  Compressed output shape: {pred_compressed.shape}")
        
        # File size comparison
        original_size = Path(original_model_path).stat().st_size / (1024 * 1024)
        compressed_size = Path(compressed_model_path).stat().st_size / (1024 * 1024)
        
        print("\nModel Size:")
        print(f"  Original:   {original_size:.2f} MB")
        print(f"  Compressed: {compressed_size:.2f} MB")
        print(f"  Reduction:  {((1 - compressed_size / original_size) * 100):.1f}%")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Validation error: {e}")
        print("Checking file sizes...")
        
        original_size = Path(original_model_path).stat().st_size / (1024 * 1024)
        compressed_size = Path(compressed_model_path).stat().st_size / (1024 * 1024)
        
        print(f"  Original:   {original_size:.2f} MB")
        print(f"  Compressed: {compressed_size:.2f} MB")
        print(f"  Reduction:  {((1 - compressed_size / original_size) * 100):.1f}%")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compress ONNX model using float16 conversion or int8 quantization."
    )
    parser.add_argument('--input_model', type=str, required=True,
                        help='Path to input ONNX model')
    parser.add_argument('--calibration_data', type=str, required=True,
                        help='Path to CSV calibration data file')
    parser.add_argument('--output_model', type=str, required=True,
                        help='Path to output compressed ONNX model')
    parser.add_argument('--compression', type=str, default='float16',
                        choices=['float16', 'dynamic_int8', 'static_int8'],
                        help='Compression method: float16 (50%% reduction), dynamic_int8 (75%% reduction, lower accuracy), static_int8 (75%% reduction, better accuracy)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.input_model).exists():
        raise FileNotFoundError(f"Input model not found: {args.input_model}")
    if not Path(args.calibration_data).exists():
        raise FileNotFoundError(f"Calibration data not found: {args.calibration_data}")
    
    print(f"Input model: {args.input_model}")
    print(f"Calibration data: {args.calibration_data}")
    print(f"Output model: {args.output_model}")
    print(f"Compression method: {args.compression}\n")
    
    # Load and prepare data
    X_calib, y_calib, means, stds = load_and_prepare_data(
        args.calibration_data, args.input_model
    )
    
    # Remove metadata
    temp_model_path = Path(args.output_model).parent / "temp_cleaned.onnx"
    remove_metadata_entry(args.input_model, str(temp_model_path))
    
    # Apply compression
    if args.compression == 'float16':
        convert_to_float16(str(temp_model_path), args.output_model)
        compression_type = 'float16'
    elif args.compression == 'dynamic_int8':
        quantize_model(str(temp_model_path), args.output_model, X_calib, 'input_1', quant_type='dynamic')
        compression_type = 'dynamic_int8'
    elif args.compression == 'static_int8':
        quantize_model(str(temp_model_path), args.output_model, X_calib, 'input_1', quant_type='static')
        compression_type = 'static_int8'
    
    # Validate
    validate_model(
        str(temp_model_path),
        args.output_model,
        X_calib,
        y_calib,
        compression_type=compression_type,
        input_name='input_1',
        output_name='output_1'
    )
    
    # Clean up temporary file
    temp_model_path.unlink()
    
    print("✓ Compression complete!")


if __name__ == '__main__':
    main()