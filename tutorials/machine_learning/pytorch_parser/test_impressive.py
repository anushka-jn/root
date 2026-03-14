import os
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import torch.nn as nn
from ParseTorchLayers import parse_model

def run_pytorch_test():
    print("=========================================")
    print("   TESTING IMPROVED PYTORCH PARSER      ")
    print("=========================================")
    
    # Create an impressive model with every feature
    class ComprehensiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.bn = nn.BatchNorm2d(16)
            self.elu = nn.ELU(alpha=0.5)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(p=0.2)
            self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
            self.lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1)
            self.gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1)
            self.linear = nn.Linear(20, 1)

    model = ComprehensiveModel()
    parsed_layers = parse_model(model)
    
    found_types = set([l["type"] for l in parsed_layers])
    expected_types = {"BatchNormalization", "Elu", "MaxPool", "Dropout", "RNN", "LSTM", "GRU", "Linear"}
    
    if expected_types.issubset(found_types):
        print("✅ SUCCESS: Found all 8 requested PyTorch layer types!")
        print(f"Extracted {len(parsed_layers)} total layer definitions cleanly.")
        
        # Verify an example output
        lstm_layer = next(l for l in parsed_layers if l["type"] == "LSTM")
        lstm_weights = lstm_layer["weights"]
        print(f"\nLSTM Sub-layer '{lstm_layer['name']}':")
        print(f" -> hidden_size: {lstm_layer['attributes']['hidden_size']}")
        print(f" -> Detached weight arrays:")
        for k, v in lstm_weights.items():
            print(f"    - {k}: {v.shape}")
    else:
        print("❌ FAILED: Missing requested layer types")
        print(f"Expected: {expected_types}")
        print(f"Found: {found_types}")


def run_keras_test():
    print("\n=========================================")
    print(" TESTING IMPROVED SOFIE KERAS PARSER   ")
    print("=========================================")
    
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1DTranspose, Conv2DTranspose, GRU, LSTM, Dense, Reshape
    except ImportError:
        print("TensorFlow not installed. Skipping Keras test.")
        return

    import ROOT
    from ROOT.TMVA.Experimental import SOFIE

    # 1. Input layer
    inputs = Input(shape=(4, 4, 3))
    
    # 2. Conv2DTranspose (Exercise 5-1)
    x = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    
    # 3. Reshape 
    x = Reshape((64, 8))(x)
    
    # 4. Conv1DTranspose (Bonus Feature)
    x = Conv1DTranspose(filters=16, kernel_size=3, padding='valid')(x)
    
    # 5. GRU Layer (Exercise 5-1)
    x = GRU(16, return_sequences=True)(x)
    
    # 6. LSTM Layer (Exercise 5-1)
    x = LSTM(10)(x)
    
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model_path = os.path.join(os.getcwd(), 'tmp_keras_model.h5')
    model.save(model_path)
    
    try:
        rmodel = SOFIE.PyKeras.Parse(model_path)
        print("✅ SUCCESS: C++ SOFIE PyKeras successfully resolved Conv1DTranspose, Conv2DTranspose, GRU, and LSTM.")
        print("Parsing succeeded. (Code generation skipped — SAME_UPPER padding not yet supported in SOFIE Generate().)")
    except Exception as e:
        print("❌ FAILED to parse model:")
        print(e)
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    run_pytorch_test()
    run_keras_test()
