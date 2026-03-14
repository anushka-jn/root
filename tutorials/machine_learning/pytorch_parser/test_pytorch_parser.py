import torch
import torch.nn as nn
from pytorch_parser import parse_pytorch_model

class DummyCNNLayerModel(nn.Module):
    def __init__(self):
        super(DummyCNNLayerModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.elu = nn.ELU(alpha=1.0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return x

class DummyRNNLayerModel(nn.Module):
    def __init__(self):
        super(DummyRNNLayerModel, self).__init__()
        self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        
    def forward(self, x):
        out, h_n = self.rnn(x)
        return out

class DummyLSTMLayerModel(nn.Module):
    def __init__(self):
        super(DummyLSTMLayerModel, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out
        
class DummyGRULayerModel(nn.Module):
    def __init__(self):
        super(DummyGRULayerModel, self).__init__()
        self.gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        
    def forward(self, x):
        out, h_n = self.gru(x)
        return out

def test_cnn_layers():
    print("=== Testing CNN Layers (BatchNorm2D, ELU, MaxPool2D) ===")
    model = DummyCNNLayerModel()
    dummy_input = torch.randn(1, 3, 32, 32)
    
    parsed_layers, initializers = parse_pytorch_model(model, dummy_input)
    
    for layer in parsed_layers:
        print(f"Parsed Layer: {layer['layer_type']}")
        for k, v in layer.items():
            if k != "layer_type":
                print(f"  {k}: {v}")
    print()

def test_rnn_layers():
    print("=== Testing RNN Layer ===")
    model = DummyRNNLayerModel()
    dummy_input = torch.randn(1, 5, 10)  # (batch, seq, feature)
    
    parsed_layers, initializers = parse_pytorch_model(model, dummy_input)
    
    for layer in parsed_layers:
        print(f"Parsed Layer: {layer['layer_type']}")
        for k, v in layer.items():
            if k != "layer_type":
                print(f"  {k}: {v}")
    print()

def test_lstm_layers():
    print("=== Testing LSTM Layer ===")
    model = DummyLSTMLayerModel()
    dummy_input = torch.randn(1, 5, 10) 
    
    parsed_layers, initializers = parse_pytorch_model(model, dummy_input)
    
    for layer in parsed_layers:
        print(f"Parsed Layer: {layer['layer_type']}")
        for k, v in layer.items():
            if k != "layer_type":
                print(f"  {k}: {v}")
    print()

def test_gru_layers():
    print("=== Testing GRU Layer ===")
    model = DummyGRULayerModel()
    dummy_input = torch.randn(1, 5, 10) 
    
    parsed_layers, initializers = parse_pytorch_model(model, dummy_input)
    
    for layer in parsed_layers:
        print(f"Parsed Layer: {layer['layer_type']}")
        for k, v in layer.items():
            if k != "layer_type":
                print(f"  {k}: {v}")
    print()

if __name__ == "__main__":
    test_cnn_layers()
    test_rnn_layers()
    test_lstm_layers()
    test_gru_layers()
    print("Tests completed.")

