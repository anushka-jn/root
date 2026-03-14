import numpy as np
import ROOT
import ParseHLS4ML

class MockWeightVariable:
    def __init__(self, name, data):
        self.name = name
        self.data = np.array(data)

class MockLayer:
    def __init__(self, name, class_name, inputs, outputs, attributes=None, weights=None):
        self.name = name
        self.class_name = class_name
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}
        self.weights = weights or []
        
    def get_weights(self):
        return self.weights

class MockModelGraph:
    def __init__(self, layers):
        self._layers = layers
        
    def get_layers(self):
        return self._layers

def test_hls4ml_parser():
    print("Building Mock HLS4ML ModelGraph...")
    
    layers = []
    
    # 1. InputLayer
    layers.append(MockLayer("input_1", "InputLayer", [], ["input_tensor"], {"input_shape": [None, 12]}))
    
    # 2. Dense Layer mapping 12 -> 8
    w_data = np.random.rand(12, 8).astype(np.float32)
    b_data = np.random.rand(8).astype(np.float32)
    weights = [
        MockWeightVariable("kernel", w_data),
        MockWeightVariable("bias", b_data)
    ]
    layers.append(MockLayer("dense_1", "Dense", ["input_tensor"], ["dense_tensor"], {}, weights))
    
    # 3. ReLU layer
    layers.append(MockLayer("relu_1", "ReLU", ["dense_tensor"], ["relu_tensor"], {"activation": "relu"}))
    
    # 4. Elu layer (parallel to ReLU, drawing from same Dense output)
    layers.append(MockLayer("elu_1", "Activation", ["dense_tensor"], ["elu_tensor"], {"activation": "elu", "alpha": 1.5}))
    
    # 5. Concatenate layer
    layers.append(MockLayer("concat_1", "Concatenate", ["relu_tensor", "elu_tensor"], ["concat_tensor"], {"axis": 1}))

    # 6. Reshape layer (8 + 8 = 16) -> (4, 4)
    layers.append(MockLayer("reshape_1", "Reshape", ["concat_tensor"], ["reshape_tensor"], {"target_shape": [4, 4]}))
    
    # 7. Add dummy spatial dimension to Reshape so Conv2D can work: (batch, height, width, channels) -> (1, 4, 4, 1)
    # SOFIE convolution generally expects NCHW (batch, channel, height, width) format based on root parser mapping
    layers.append(MockLayer("reshape_to_img", "Reshape", ["reshape_tensor"], ["img_tensor"], {"target_shape": [1, 1, 4, 4]}))

    # 8. Conv2D Layer
    conv_w = np.random.rand(1, 1, 3, 3).astype(np.float32) # (out_channels, in_channels, k_h, k_w)
    conv_b = np.random.rand(1).astype(np.float32)
    conv_weights = [MockWeightVariable("kernel", conv_w), MockWeightVariable("bias", conv_b)]
    layers.append(MockLayer("conv_1", "Conv2D", ["img_tensor"], ["conv_tensor"], 
                           {"kernel_size": [3, 3], "strides": [1, 1], "padding": "same"}, conv_weights))
                           
    # 9. BatchNormalization Layer
    bn_w = [
        MockWeightVariable("gamma", np.random.rand(1).astype(np.float32)),
        MockWeightVariable("beta", np.random.rand(1).astype(np.float32)),
        MockWeightVariable("moving_mean", np.random.rand(1).astype(np.float32)),
        MockWeightVariable("moving_variance", np.random.rand(1).astype(np.float32))
    ]
    layers.append(MockLayer("bn_1", "BatchNormalization", ["conv_tensor"], ["bn_tensor"], {}, bn_w))

    # 10. MaxPooling2D Layer
    layers.append(MockLayer("maxpool_1", "MaxPooling2D", ["bn_tensor"], ["pool_tensor"], 
                           {"pool_size": [2, 2], "strides": [2, 2]}))
                           
    mock_model = MockModelGraph(layers)
    
    print("Invoking ParseHLS4ML.Parse()...")
    rmodel = ParseHLS4ML.Parse(mock_model, "TestHLS4MLModel")
    
    print("Generating C++ inference header...")
    rmodel.Generate()
    rmodel.OutputGenerated()
    print("Successfully generated TestHLS4MLModel.hxx! Checking if standard compile works...")
    
    ROOT.gInterpreter.Declare('#include "TestHLS4MLModel.hxx"')
    print("Successfully parsed, generated, and compiled SOFIE model from HLS4ML mock graph.")

if __name__ == "__main__":
    test_hls4ml_parser()

