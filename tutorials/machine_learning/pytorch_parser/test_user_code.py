import torch
import torch.nn as nn

from ParseTorchLayers import parse_model

def test_user_code():
    print("Testing ELU")
    model = nn.ELU(alpha=2.0)
    print(parse_model(model))

    print("\nTesting MaxPool2D")
    model = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(parse_model(model))

    print("\nTesting BatchNorm2D")
    model = nn.BatchNorm2d(16)
    print(parse_model(model))

    print("\nTesting RNN")
    model = nn.RNN(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
    res = parse_model(model)
    print(res)
    for k, v in res[0]["weights"].items():
        print(f"Weight {k} shape: {v.shape}")

    print("\nTesting LSTM")
    model = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, proj_size=5)
    res = parse_model(model)
    print(res)
    for k, v in res[0]["weights"].items():
        print(f"Weight {k} shape: {v.shape}")

    print("\nTesting GRU")
    model = nn.GRU(input_size=10, hidden_size=20)
    res = parse_model(model)
    print(res)
    for k, v in res[0]["weights"].items():
        print(f"Weight {k} shape: {v.shape}")

if __name__ == "__main__":
    test_user_code()
