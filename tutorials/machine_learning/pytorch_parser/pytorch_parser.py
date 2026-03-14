import torch
from torch.onnx.utils import _model_to_graph

def _node_get(node, key):
    """
    Helper to extract node attributes safely.
    See https://github.com/pytorch/pytorch/pull/82628 for why we do this.
    """
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

def extract_node_attributes(node):
    """
    Extracts attributes for a given ONNX node.
    """
    attributes = {}
    for attr_name in node.attributeNames():
        attributes[attr_name] = _node_get(node, attr_name)
    return attributes

def parse_onnx_node(node):
    """
    Parses an ONNX node and returns its type, attributes, inputs, and outputs.
    """
    def safe_scalar_type(x):
        try:
            return x.type().scalarType()
        except RuntimeError:
            return None

    node_data = {
        "nodeType": node.kind(),
        "nodeAttributes": extract_node_attributes(node),
        "nodeInputs": [x.debugName() for x in node.inputs()],
        "nodeOutputs": [x.debugName() for x in node.outputs()],
        "nodeDType": [safe_scalar_type(x) for x in node.outputs()]
    }
    return node_data

def process_elu(node_data):
    """Extracts information for an ELU layer."""
    return {
        "layer_type": "ELU",
        "alpha": node_data["nodeAttributes"].get("alpha", 1.0),
        "input_tensor": node_data["nodeInputs"][0],
        "output_tensor": node_data["nodeOutputs"][0]
    }

def process_maxpool2d(node_data):
    """Extracts information for a MaxPool2D layer."""
    attrs = node_data["nodeAttributes"]
    return {
        "layer_type": "MaxPool2D",
        "kernel_shape": attrs.get("kernel_shape", []),
        "strides": attrs.get("strides", []),
        "pads": attrs.get("pads", []),
        "dilations": attrs.get("dilations", []),
        "ceil_mode": attrs.get("ceil_mode", 0),
        "input_tensor": node_data["nodeInputs"][0],
        "output_tensor": node_data["nodeOutputs"][0]
    }

def process_batchnorm2d(node_data):
    """Extracts information for a BatchNorm2D layer."""
    attrs = node_data["nodeAttributes"]
    return {
        "layer_type": "BatchNorm2D",
        "epsilon": attrs.get("epsilon", 1e-5),
        "momentum": attrs.get("momentum", 0.9),
        "input_tensor": node_data["nodeInputs"][0],
        "weight": node_data["nodeInputs"][1] if len(node_data["nodeInputs"]) > 1 else None,
        "bias": node_data["nodeInputs"][2] if len(node_data["nodeInputs"]) > 2 else None,
        "running_mean": node_data["nodeInputs"][3] if len(node_data["nodeInputs"]) > 3 else None,
        "running_var": node_data["nodeInputs"][4] if len(node_data["nodeInputs"]) > 4 else None,
        "output_tensor": node_data["nodeOutputs"][0]
    }

def process_rnn_lstm_gru(node_data, layer_type):
    """
    Extracts information for an RNN, LSTM, or GRU layer from ONNX representation. 
    ONNX represents these via `onnx::RNN`, `onnx::LSTM`, or `onnx::GRU`.
    """
    attrs = node_data["nodeAttributes"]
    inputs = node_data["nodeInputs"]
    
    layer_info = {
        "layer_type": layer_type,
        "hidden_size": attrs.get("hidden_size", None),
        "direction": attrs.get("direction", "forward"),
        "activations": attrs.get("activations", []),
        "input_tensor": inputs[0] if len(inputs) > 0 else None,
        "weight_ih": inputs[1] if len(inputs) > 1 else None,
        "weight_hh": inputs[2] if len(inputs) > 2 else None,
        "bias": inputs[3] if len(inputs) > 3 else None,
        "sequence_lens": inputs[4] if len(inputs) > 4 else None,
        "initial_h": inputs[5] if len(inputs) > 5 else None,
    }
    
    # LSTM specifically has an initial cell state (initial_c)
    if layer_type == "LSTM" and len(inputs) > 6:
        layer_info["initial_c"] = inputs[6]
        
    return layer_info

# Mapping ONNX nodes to our specific processors
PROCESSOR_MAP = {
    "onnx::Elu": process_elu,
    "onnx::MaxPool": process_maxpool2d,
    "onnx::BatchNormalization": process_batchnorm2d,
    "onnx::RNN": lambda d: process_rnn_lstm_gru(d, "RNN"),
    "onnx::LSTM": lambda d: process_rnn_lstm_gru(d, "LSTM"),
    "onnx::GRU": lambda d: process_rnn_lstm_gru(d, "GRU"),
}

def parse_pytorch_model(model, dummy_inputs):
    """
    Main entry point to parse a loaded PyTorch model.
    Returns:
    - parsed_layers: A list of dicts with extracted structural information for each supported layer.
    - initializers: A dictionary of {tensor_name: numpy_array} containing model weights.
    """
    model.eval()
    
    # NOTE: _model_to_graph traces the model and may fold BatchNorm2d into
    # adjacent Conv layers during optimisation. If BatchNorm2d is not appearing
    # in parsed_layers, use ParseTorchLayers.py (nn.Module inspection) instead,
    # which always extracts BatchNorm2d regardless of graph optimisations.
    
    # Trace the ONNX graph
    graph, _, _ = _model_to_graph(model, dummy_inputs)
    
    parsed_layers = []
    
    # Iterate through operators
    for node in graph.nodes():
        node_data = parse_onnx_node(node)
        node_type = node_data["nodeType"]
        
        # Process node if we recognize it
        if node_type in PROCESSOR_MAP:
            parsed_layer_info = PROCESSOR_MAP[node_type](node_data)
            parsed_layers.append(parsed_layer_info)
            
    # Extract weights and initializers (similar to C++ _model_to_graph tuple access)
    # The output of _model_to_graph is (graph, init_params_dict, ?)
    # We can fetch the raw numpy arrays out of the parameters dictionary
    initializers = {}
    params_dict = _model_to_graph(model, dummy_inputs)[1]
    for param_name, tensor_val in params_dict.items():
        if isinstance(tensor_val, torch.Tensor):
            initializers[param_name] = tensor_val.detach().cpu().numpy()
            
    return parsed_layers, initializers
