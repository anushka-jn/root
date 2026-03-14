"""
ParseTorchLayers.py
-------------------
GSoC 2026 — SOFIE: Improving Keras and PyTorch Parsers
Exercise 4: Python parsing functions for 6 PyTorch layer types.

Supported layers:
  1. ELU
  2. MaxPool2D
  3. BatchNorm2D
  4. RNN
  5. LSTM
  6. GRU

Usage:
    import torch.nn as nn
    from ParseTorchLayers import parse_model

    model = nn.Sequential(
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.GRU(input_size=10, hidden_size=20),
    )
    layers = parse_model(model)
    for l in layers:
        print(l["name"], l["type"])
"""


# ---------------------------------------------------------------------------
# 1. ELU
# ---------------------------------------------------------------------------

def parse_elu(layer):
    """
    Parse nn.ELU layer.

    ELU (Exponential Linear Unit):
        f(x) = x            if x > 0
        f(x) = alpha*(e^x-1) if x <= 0

    Attributes:
        alpha (float): Controls the value to which ELU saturates for negative inputs.
                       Default in PyTorch is 1.0.
    """
    return {
        "type": "Elu",
        "attributes": {
            "alpha": float(layer.alpha),
        },
        "weights": {},
    }


# ---------------------------------------------------------------------------
# 2. MaxPool2D
# ---------------------------------------------------------------------------

def parse_maxpool2d(layer):
    """
    Parse nn.MaxPool2d layer.

    Performs max pooling over a 2D input (H x W).

    Attributes:
        kernel_shape (list[int]): Height and width of the pooling window.
        strides      (list[int]): Step size of the window in H and W.
        pads         (list[int]): Padding [top, left, bottom, right].
                                  ONNX/SOFIE use 4-element pads, not 2-element.
        dilations    (list[int]): Dilation factor for pooling window.
        ceil_mode    (int):       0 = floor (default), 1 = ceil for output size.
    """
    def to2(v):
        return [v, v] if isinstance(v, int) else list(v)

    ks = to2(layer.kernel_size)
    st = to2(layer.stride) if layer.stride is not None else ks
    pd = to2(layer.padding)
    dl = to2(layer.dilation)

    return {
        "type": "MaxPool",
        "attributes": {
            "kernel_shape": ks,
            "strides":      st,
            "pads":         pd + pd,          # [top, left, bottom, right]
            "dilations":    dl,
            "ceil_mode":    int(layer.ceil_mode),
        },
        "weights": {},
    }


# ---------------------------------------------------------------------------
# 3. BatchNorm2D
# ---------------------------------------------------------------------------

def parse_batchnorm2d(layer):
    """
    Parse nn.BatchNorm2d layer.

    Normalises each channel across the batch during training.
    Uses running statistics (mean, var) during inference.

    Attributes:
        epsilon  (float): Small value added to denominator for numerical stability.
        momentum (float): ONNX/SOFIE convention is INVERTED vs PyTorch:
                          momentum_onnx = 1.0 - momentum_pytorch

    Weights (all stored as 1D arrays of length num_features):
        scale        : learnable gamma (layer.weight)
        bias         : learnable beta  (layer.bias)
        running_mean : tracked mean used at inference
        running_var  : tracked variance used at inference
    """
    import numpy as np

    return {
        "type": "BatchNormalization",
        "attributes": {
            "epsilon":  float(layer.eps),
            # PyTorch momentum is complementary to ONNX/Keras convention
            # Handle PyTorch None momentum
            "momentum": float(1.0 - layer.momentum) if layer.momentum is not None else 1.0, 
        },
        "weights": {
            "scale":        layer.weight.detach().numpy() if layer.weight is not None else None,
            "bias":         layer.bias.detach().numpy() if layer.bias is not None else None,
            "running_mean": layer.running_mean.detach().numpy() if layer.running_mean is not None else None,
            "running_var":  layer.running_var.detach().numpy() if layer.running_var is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Helper: extract RNN/LSTM/GRU weights (shared structure)
# ---------------------------------------------------------------------------

def _extract_recurrent_weights(layer):
    """
    Extract weight matrices for RNN, LSTM, or GRU.

    All three recurrent types store weights the same way in PyTorch:
      weight_ih_l{i}  : input-hidden weights for layer i
      weight_hh_l{i}  : hidden-hidden weights for layer i
      bias_ih_l{i}    : input-hidden bias for layer i
      bias_hh_l{i}    : hidden-hidden bias for layer i

    For bidirectional models, the same set exists with a '_reverse' suffix.

    Gate layouts (rows of weight matrices, each block = hidden_size rows):
      RNN:  [1 gate]        tanh or relu
      GRU:  [r, z, n]       reset, update, new       (3 * hidden_size rows)
      LSTM: [i, f, g, o]    input, forget, cell, out  (4 * hidden_size rows)
    """
    import numpy as np

    weights = {}
    for i in range(layer.num_layers):
        for name in [
            f"weight_ih_l{i}", f"weight_hh_l{i}",
            f"bias_ih_l{i}",   f"bias_hh_l{i}",
        ]:
            if hasattr(layer, name) and getattr(layer, name) is not None:
                weights[name] = getattr(layer, name).detach().numpy()

        if layer.bidirectional:
            for name in [
                f"weight_ih_l{i}_reverse", f"weight_hh_l{i}_reverse",
                f"bias_ih_l{i}_reverse",   f"bias_hh_l{i}_reverse",
            ]:
                if hasattr(layer, name) and getattr(layer, name) is not None:
                    weights[name] = getattr(layer, name).detach().numpy()

    # Re-order the matrices for LSTM and GRU when exporting them to external formats
    # as mentioned in your module
    
    return weights


# ---------------------------------------------------------------------------
# 4. RNN
# ---------------------------------------------------------------------------

def parse_rnn(layer):
    """
    Parse nn.RNN layer.

    Simple recurrent network: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    (or ReLU if nonlinearity='relu')

    Attributes:
        hidden_size   : Number of features in hidden state.
        num_layers    : Number of stacked RNN layers.
        bidirectional : Whether the RNN processes input in both directions.
        batch_first   : If True, input shape is (batch, seq, feature),
                        otherwise (seq, batch, feature).
        nonlinearity  : 'tanh' (default) or 'relu'.

    Weights: see _extract_recurrent_weights()
    """
    return {
        "type": "RNN",
        "attributes": {
            "hidden_size":   layer.hidden_size,
            "num_layers":    layer.num_layers,
            "bidirectional": int(layer.bidirectional),
            "batch_first":   int(layer.batch_first),
            "nonlinearity":  layer.nonlinearity,    # "tanh" or "relu"
        },
        "weights": _extract_recurrent_weights(layer),
    }


# ---------------------------------------------------------------------------
# 5. LSTM
# ---------------------------------------------------------------------------

def parse_lstm(layer):
    """
    Parse nn.LSTM layer.

    LSTM (Long Short-Term Memory) maintains both hidden state h_t AND cell state c_t.
    Uses 4 gates: input (i), forget (f), cell (g), output (o).

    Weight matrix shape: weight_ih_l0 → (4 * hidden_size, input_size)
    Gate block order (rows): [i, f, g, o]  ← PyTorch/Keras order
    ONNX gate order:          [i, o, f, c]  ← different! reorder when using ONNX ops.

    Attributes:
        hidden_size   : Size of hidden/cell state.
        num_layers    : Number of stacked LSTM layers.
        bidirectional : Bidirectional flag.
        batch_first   : Input shape convention.
        proj_size     : If > 0, adds a projection layer after LSTM output.
                        Useful for reducing output dimensionality.

    Weights: see _extract_recurrent_weights()
    """
    return {
        "type": "LSTM",
        "attributes": {
            "hidden_size":   layer.hidden_size,
            "num_layers":    layer.num_layers,
            "bidirectional": int(layer.bidirectional),
            "batch_first":   int(layer.batch_first),
            "proj_size":     layer.proj_size,         # 0 means no projection
        },
        "weights": _extract_recurrent_weights(layer),
    }


# ---------------------------------------------------------------------------
# 6. GRU
# ---------------------------------------------------------------------------

def parse_gru(layer):
    """
    Parse nn.GRU layer.

    GRU (Gated Recurrent Unit) — simpler than LSTM, no separate cell state.
    Uses 3 gates: reset (r), update (z), new (n).

    Weight matrix shape: weight_ih_l0 → (3 * hidden_size, input_size)
    Gate block order (rows): [r, z, n]  ← PyTorch order
    Keras order:              [z, r, h]  ← swap first two blocks when converting
    ONNX order:               [r, z, h]  ← same as PyTorch

    Attributes:
        hidden_size   : Size of hidden state.
        num_layers    : Number of stacked GRU layers.
        bidirectional : Bidirectional flag.
        batch_first   : Input shape convention.

    Weights: see _extract_recurrent_weights()
    """
    return {
        "type": "GRU",
        "attributes": {
            "hidden_size":   layer.hidden_size,
            "num_layers":    layer.num_layers,
            "bidirectional": int(layer.bidirectional),
            "batch_first":   int(layer.batch_first),
        },
        "weights": _extract_recurrent_weights(layer),
    }


# ---------------------------------------------------------------------------
# 7. Linear
# ---------------------------------------------------------------------------

def parse_linear(layer):
    """
    Parse nn.Linear layer.

    Performs a linear transformation: y = xA^T + b
    
    Attributes:
        in_features  : Size of input feature vector.
        out_features : Size of output feature vector.
    
    Weights:
        weight : Transposed learnable weights (out_features, in_features)
        bias   : Learnable bias (out_features)
    """
    return {
        "type": "Linear",
        "attributes": {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        },
        "weights": {
            "weight": layer.weight.detach().numpy() if layer.weight is not None else None,
            "bias": layer.bias.detach().numpy() if layer.bias is not None else None,
        },
    }

# ---------------------------------------------------------------------------
# 8. Dropout
# ---------------------------------------------------------------------------

def parse_dropout(layer):
    """
    Parse nn.Dropout layer.

    Randomly zeroes elements with probability p during training.
    Acts as an identity function during inference, so SOFIE often ignores them
    or maps them to Identity.

    Attributes:
        p      : Probability of zeroing an element.
        inplace: Whether the operation is performed in-place.
    """
    return {
        "type": "Dropout",
        "attributes": {
            "p": float(layer.p),
            "inplace": int(layer.inplace),
        },
        "weights": {},
    }

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Maps torch.nn layer types to their parse functions
_LAYER_PARSERS = None   # lazy init to avoid importing torch at module level


def _get_parsers():
    """Lazy-load layer parsers to avoid importing torch at module level."""
    global _LAYER_PARSERS
    if _LAYER_PARSERS is None:
        import torch.nn as nn
        _LAYER_PARSERS = {
            nn.ELU:         parse_elu,
            nn.MaxPool2d:   parse_maxpool2d,
            nn.BatchNorm2d: parse_batchnorm2d,
            nn.RNN:         parse_rnn,
            nn.LSTM:        parse_lstm,
            nn.GRU:         parse_gru,
            nn.Linear:      parse_linear,
            nn.Dropout:     parse_dropout,
        }
    return _LAYER_PARSERS


def parse_layer(layer):
    """
    Parse a single PyTorch layer into a SOFIE-compatible dict.

    Returns None if the layer type is not supported.
    """
    parsers = _get_parsers()
    fn = parsers.get(type(layer))
    return fn(layer) if fn else None


def parse_model(model):
    """
    Walk all modules in a PyTorch model and parse supported layers.

    Returns a list of dicts, each containing:
        name       (str)  : module path within the model (e.g. "encoder.0")
        type       (str)  : SOFIE operator name (e.g. "LSTM")
        attributes (dict) : layer hyperparameters
        weights    (dict) : numpy arrays of learnable/tracked parameters

    Only layers whose types are in the supported set are included.
    Container modules like nn.Sequential are skipped automatically.

    Example:
        model = nn.Sequential(nn.BatchNorm2d(16), nn.GRU(16, 32))
        layers = parse_model(model)
        # → [{"name": "0", "type": "BatchNormalization", ...},
        #    {"name": "1", "type": "GRU", ...}]
    """
    results = []
    
    # Do not parse the entire Sequential container module as a whole if it is the root
    # nn.Module.named_modules() will iterate over the root itself first and then its children.
    for name, layer in model.named_modules():
        parsed = parse_layer(layer)
        if parsed is not None:
            parsed["name"] = name if name else getattr(layer, "__class__").__name__
            results.append(parsed)
    return results
