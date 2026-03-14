import time
import numpy as np
import ROOT
import ParseHLS4ML


class RealLayerAdapter:
    """
    Wraps a real HLS4ML layer (VivadoDense, VivadoActivation, etc.)
    to match the interface expected by ParseHLS4ML's handler functions.
    """
    def __init__(self, real_layer):
        self._layer = real_layer
        attrs = real_layer.attributes

        # Use the Keras class_name stored inside attributes
        self.class_name = attrs.get('class_name', real_layer.__class__.__name__)

        self.name = real_layer.name

        # inputs/outputs are direct attributes on the layer
        self.inputs  = list(real_layer.inputs)
        self.outputs = [real_layer.name]   # HLS4ML output tensor == layer name

        # Build a clean attribute dict ParseHLS4ML can use
        self.attributes = {}

        # Input shape — HLS4ML stores [[n_in]] or [n_in]
	# Input shape — prefer n_in for accurate shape, fallback to input_shape
        if 'n_in' in attrs:
            raw_shape = [attrs['n_in']]
        else:
            raw_shape = attrs.get('input_shape', [1])
            if raw_shape and isinstance(raw_shape[0], list):
                raw_shape = raw_shape[0]
        self.attributes['input_shape'] = raw_shape
       

        # Dense specifics
        if 'n_in' in attrs:
            self.attributes['n_in'] = attrs['n_in']
        if 'n_out' in attrs:
            self.attributes['n_out'] = attrs['n_out']

        # Activation — read from class_name or activation attribute
        act = attrs.get('activation', 'linear')
        if self.class_name == 'Activation':
            # HLS4ML names activation layers like "dense_relu", "dense_softmax"
            for a in ['relu', 'elu', 'softmax', 'sigmoid', 'tanh']:
                if a in real_layer.name.lower():
                    act = a
                    break
        self.attributes['activation'] = act
        self.attributes['alpha'] = attrs.get('alpha', 1.0)

    def get_weights(self):
        """
        Return weight objects with .name and .data matching ParseHLS4ML expectations.
        We read from layer.attributes['weight_data'] and ['bias_data'] directly,
        which are plain numpy arrays — more reliable than get_weights() name guessing.
        """
        weights = []
        attrs = self._layer.attributes

        if 'weight_data' in attrs:
            weights.append(_WeightAdapter('kernel', np.array(attrs['weight_data'])))
        if 'bias_data' in attrs:
            weights.append(_WeightAdapter('bias', np.array(attrs['bias_data'])))

        return weights


class _WeightAdapter:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class RealModelGraphAdapter:
    """
    Wraps hls_model.graph (OrderedDict) to provide get_layers()
    returning RealLayerAdapter objects.
    """
    def __init__(self, hls_model):
        self._graph = hls_model.graph

    def get_layers(self):
        return [RealLayerAdapter(layer) for layer in self._graph.values()]


def Parse(hls_model, model_name="HLS4ML_SOFIE_Model", batch_size=1):
    """
    Entry point: takes a real hls4ml model object and returns a SOFIE RModel.
    """
    adapted = RealModelGraphAdapter(hls_model)
    return ParseHLS4ML.Parse(adapted, model_name, batch_size)
