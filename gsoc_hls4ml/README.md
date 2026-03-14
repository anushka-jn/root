# GSoC 2026 — SOFIE HLS4ML Integration

**Candidate:** Anushka  
**Mentors:** Lorenzo Moneta, Sanjiban Sengupta  
**Branch:** gsoc-2026-hls4ml

---

## Exercise 1 — Building ROOT from Source

- Forked and cloned ROOT from https://github.com/root-project/root
- Built ROOT from source with the following configuration:
```bash
  cmake -Dtmva-sofie=On -Dtmva=On ..
  make -j$(nproc)
```
- Confirmed SOFIE is enabled with protobuf support
- Python environment includes: numpy, tensorflow, hls4ml 1.2.0
- Development branch created: `gsoc-2026-hls4ml`

---

## Exercise 2 — Familiarity with ROOT TMVA Deep Learning

- Ran `tutorials/machine_learning/TMVA_Higgs_Classification` — 
  classification using deep learning network in TMVA
- Ran `tutorials/machine_learning/TMVA_CNN_Classification.C` — 
  2D convolutional network classification
- Explored the pymva module for Keras/PyTorch Python interfaces in TMVA
- Ran SOFIE tutorials:
  - `TMVA_SOFIE_ONNX.C`
  - `TMVA_SOFIE_Keras.C`
  - `TMVA_SOFIE_PyTorch.C`
- Studied the existing Keras parser (`ParseKeras.py`) as reference 
  for implementing the HLS4ML parser

---

## Exercise 3 — Exploring HLS4ML

- Installed HLS4ML 1.2.0 and explored its architecture
- Key findings:
  - `hls_model.graph` is an `OrderedDict` of layer objects
  - Layer class names are backend-prefixed: `VivadoDense`, `VivadoActivation` etc.
  - Keras class name is stored inside `layer.attributes['class_name']`
  - Weights are stored as `weight_data` and `bias_data` in attributes (numpy arrays)
  - Input/output tensor names follow layer name conventions
  - HLS4ML explicitly separates fused activations into standalone nodes
    (e.g. `dense_relu`, `dense_1_relu`, `dense_2_softmax`)
- This exploration directly informed the adapter design in `ParseHLS4ML_real.py`

---

## Exercise 4 — Parsing Function

Implemented in `ParseHLS4ML.py` — the `Parse()` function takes an HLS4ML 
ModelGraph and returns a configured SOFIE `RModel`.

**Parsing Strategy (3 passes):**
1. **Input pass** — scans for InputLayer nodes, registers input tensors with shape
2. **Operator pass** — topology-preserving traversal, translates each layer to a SOFIE ROperator
3. **Output resolution pass** — set difference of all inputs vs outputs to find terminal tensors

---

## Exercise 5 (Bonus) — Basic Operators

All 5 required operators implemented:

| Operator | Handler | SOFIE Class |
|----------|---------|-------------|
| ReLU | `MakeHLS4MLActivation` | `ROperator_Relu` |
| Elu | `MakeHLS4MLActivation` | `ROperator_Elu` |
| Gemm (Dense) | `MakeHLS4MLDense` | `ROperator_Gemm` |
| Reshape | `MakeHLS4MLReshape` | `ROperator_Reshape` |
| Concat | `MakeHLS4MLConcat` | `ROperator_Concat` |

**Bonus operators also implemented:** Conv1D/2D, BatchNormalization, 
MaxPooling1D/2D, AveragePooling1D/2D

---

## Real Integration Test

`ParseHLS4ML_real.py` bridges the real HLS4ML 1.2.0 API to the parser.

**Test result:**
```
SUCCESS: Real HLS4ML model parsed, generated, and compiled!
Generated file: RealHLS4MLModel.hxx
```

Full output in `real_test_output.txt`.

---

## Files

| File | Description |
|------|-------------|
| `ParseHLS4ML.py` | Core SOFIE parser for HLS4ML ModelGraph |
| `ParseHLS4ML_real.py` | Adapter for real HLS4ML 1.2.0 API |
| `test_hls4ml.py` | Mock-based unit test |
| `test_real_final.py` | Real integration test with Keras → HLS4ML → SOFIE |
| `real_test_output.txt` | Proof of successful real test |
| `RealHLS4MLModel.hxx` | Generated C++ inference header |
