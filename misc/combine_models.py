#!/usr/bin/env python3

import sys
import argparse
import logging
import json
import onnx
import numpy as np
from onnx import helper, TensorProto

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model(path):
    """Load ONNX model from file."""
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        sys.exit(1)


def get_metadata_dict(model):
    """Extract metadata from model as dictionary."""
    return {prop.key: prop.value for prop in model.metadata_props}


def check_consistency(models):
    """Check consistency across all models and log warnings on mismatch."""
    for i in range(1, len(models)):
        model0_inputs = [inp.name for inp in models[0].graph.input]
        modeli_inputs = [inp.name for inp in models[i].graph.input]
        
        if len(model0_inputs) != len(modeli_inputs):
            logger.warning(
                f"Model 0 has {len(model0_inputs)} inputs, "
                f"Model {i} has {len(modeli_inputs)} inputs"
            )
        
        if len(models[i].graph.output) != 1:
            logger.warning(
                f"Model {i} has {len(models[i].graph.output)} outputs, expected 1"
            )
    
    metadata_list = [get_metadata_dict(m) for m in models]
    keys_0 = set(metadata_list[0].keys())
    
    for i in range(1, len(models)):
        keys_i = set(metadata_list[i].keys())
        
        if keys_0 != keys_i:
            missing = keys_0 - keys_i
            extra = keys_i - keys_0
            if missing:
                logger.warning(f"Model {i} missing metadata keys: {missing}")
            if extra:
                logger.warning(f"Model {i} has extra metadata keys: {extra}")
        
        for key in keys_0 & keys_i:
            if key not in ['standardization_mean', 'standardization_std']:
                if metadata_list[0][key] != metadata_list[i][key]:
                    logger.warning(
                        f"Metadata key '{key}' differs between Model 0 and Model {i}: "
                        f"'{metadata_list[0][key]}' vs '{metadata_list[i][key]}'"
                    )


def parse_metadata_array(value_str):
    """Parse metadata array string (JSON list or comma/space separated) to numpy array."""
    try:
        return np.array(json.loads(value_str))
    except (json.JSONDecodeError, ValueError):
        pass
    
    if ',' in value_str:
        values = value_str.split(',')
    else:
        values = value_str.split()
    
    return np.array([float(v.strip()) for v in values if v.strip()])


def combine_standardization_params(models):
    """Combine standardization mean and std from all models."""
    metadata_list = [get_metadata_dict(m) for m in models]
    
    combined_mean = None
    combined_std = None
    
    if 'standardization_mean' in metadata_list[0]:
        mean_0 = parse_metadata_array(metadata_list[0]['standardization_mean'])
        input_means = mean_0[:-1]
        
        output_means = []
        for i, metadata in enumerate(metadata_list):
            if 'standardization_mean' in metadata:
                mean_i = parse_metadata_array(metadata['standardization_mean'])
                output_means.append(mean_i[-1])
            else:
                logger.warning(f"Model {i} missing standardization_mean")
                output_means.append(0.0)
        
        combined_mean = np.concatenate([input_means, output_means])
    
    if 'standardization_std' in metadata_list[0]:
        std_0 = parse_metadata_array(metadata_list[0]['standardization_std'])
        input_stds = std_0[:-1]
        
        output_stds = []
        for i, metadata in enumerate(metadata_list):
            if 'standardization_std' in metadata:
                std_i = parse_metadata_array(metadata['standardization_std'])
                output_stds.append(std_i[-1])
            else:
                logger.warning(f"Model {i} missing standardization_std")
                output_stds.append(1.0)
        
        combined_std = np.concatenate([input_stds, output_stds])
    
    return combined_mean, combined_std


def create_combined_model(models, output_path):
    """Create a combined ONNX model from 4 separate models."""
    check_consistency(models)
    
    graph_inputs = list(models[0].graph.input)
    all_nodes = []
    all_initializers = []
    graph_outputs = []
    
    for model_idx, model in enumerate(models):
        init_renames = {}
        for initializer in model.graph.initializer:
            old_name = initializer.name
            new_name = f"{old_name}_m{model_idx}"
            init_renames[old_name] = new_name
            
            new_init = onnx.helper.make_tensor(
                name=new_name,
                data_type=initializer.data_type,
                dims=initializer.dims,
                vals=initializer.raw_data,
                raw=True
            )
            all_initializers.append(new_init)
        
        orig_output_name = model.graph.output[0].name
        new_output_name = f"output_{model_idx}"
        
        inter_renames = {}
        if model_idx > 0:
            for node in model.graph.node:
                for out in node.output:
                    if out != orig_output_name:
                        inter_renames[out] = f"{out}_m{model_idx}"
        
        for node in model.graph.node:
            new_inputs = []
            for inp in node.input:
                if inp in init_renames:
                    new_inputs.append(init_renames[inp])
                elif inp in inter_renames:
                    new_inputs.append(inter_renames[inp])
                else:
                    new_inputs.append(inp)
            
            new_node_outputs = []
            for out in node.output:
                if out == orig_output_name:
                    new_node_outputs.append(new_output_name)
                elif out in inter_renames:
                    new_node_outputs.append(inter_renames[out])
                else:
                    new_node_outputs.append(out)
            
            new_node = onnx.helper.make_node(
                node.op_type,
                inputs=new_inputs,
                outputs=new_node_outputs,
                name=f"{node.name}_m{model_idx}" if node.name else f"node_m{model_idx}",
                **{attr.name: helper.get_attribute_value(attr) 
                   for attr in node.attribute}
            )
            all_nodes.append(new_node)
        
        output_tensor = models[model_idx].graph.output[0]
        try:
            shape_dims = []
            if hasattr(output_tensor.type, 'tensor_type'):
                for dim in output_tensor.type.tensor_type.shape.dim:
                    # Use None for dynamic dimensions, actual value for fixed dimensions
                    if dim.dim_value > 0:
                        shape_dims.append(dim.dim_value)
                    else:
                        shape_dims.append(None)  # Dynamic dimension
            
            output_info = helper.make_tensor_value_info(
                new_output_name,
                output_tensor.type.tensor_type.elem_type,
                shape_dims
            )
            graph_outputs.append(output_info)
        except Exception as e:
            logger.warning(f"Could not extract output shape for model {model_idx}: {e}")
            # Use dynamic batch size and single output value
            output_info = helper.make_tensor_value_info(
                new_output_name,
                TensorProto.FLOAT,
                [None, 1]
            )
            graph_outputs.append(output_info)
    
    # Create a concatenation node to merge all 4 outputs into one
    concat_node = helper.make_node(
        'Concat',
        inputs=[f'output_{i}' for i in range(4)],
        outputs=['output'],
        axis=1
    )
    all_nodes.append(concat_node)
    
    # Create the final concatenated output
    concat_output = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [None, 4]
    )
    
    new_graph = helper.make_graph(
        all_nodes,
        models[0].graph.name,
        graph_inputs,
        [concat_output],  # Only the concatenated output
        all_initializers,
    )
    
    new_model = helper.make_model(new_graph, producer_name="ONNX_Combiner")
    
    metadata = get_metadata_dict(models[0])
    metadata['model_type'] = 'Composite Regressor'
    
    combined_mean, combined_std = combine_standardization_params(models)
    
    if combined_mean is not None:
        metadata['standardization_mean'] = json.dumps(combined_mean.tolist())
    
    if combined_std is not None:
        metadata['standardization_std'] = json.dumps(combined_std.tolist())
    
    for key, value in metadata.items():
        prop = onnx.StringStringEntryProto()
        prop.key = key
        prop.value = str(value)
        new_model.metadata_props.append(prop)
    
    try:
        onnx.checker.check_model(new_model)
        onnx.save(new_model, output_path)
        logger.info(f"Combined model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to create/save model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Combine 4 ONNX regression models into a single model with 4 outputs'
    )
    parser.add_argument('input1', help='Path to first ONNX model')
    parser.add_argument('input2', help='Path to second ONNX model')
    parser.add_argument('input3', help='Path to third ONNX model')
    parser.add_argument('input4', help='Path to fourth ONNX model')
    parser.add_argument('output', help='Path to save combined ONNX model')
    
    args = parser.parse_args()
    
    input_paths = [args.input1, args.input2, args.input3, args.input4]
    
    logger.info("Loading models...")
    models = [load_model(path) for path in input_paths]
    
    logger.info("Creating combined model...")
    create_combined_model(models, args.output)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()