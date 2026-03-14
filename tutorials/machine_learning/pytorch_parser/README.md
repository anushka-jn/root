# GSoC 2026: SOFIE PyTorch and Keras Parser Enhancements

This repository contains my solutions for **Exercise 4** and **Exercise 5-1 (Bonus)** of the Google Summer of Code (GSoC) 2026 project: *Improving the Keras and PyTorch Parsers for ML Inference in SOFIE*.

## Exercise 4: Python Parser for PyTorch

### Architecture Choice: Native `nn.Module` Inspection vs. ONNX Tracing
Currently, SOFIE's PyTorch parser (in C++) relies on passing models through `torch.onnx.utils._model_to_graph` to utilize ONNX node mappings. However, this backend trace mechanism is prone to OS-specific C++ internal exceptions (especially on Windows) and can struggle with complex recurrent graphs under certain Torch versions.

To build a **robust, cross-platform PyTorch parser native to Python**, I developed `ParseTorchLayers.py` which dynamically inspects the PyTorch model using `.named_modules()` and extracts properties directly from the PyTorch frontend layers. 

### Supported Layers
The parser successfully translates the requirements and *two bonus layers*:
1. **ELU**
2. **MaxPool2D**
3. **BatchNorm2D**
4. **RNN**
5. **LSTM**
6. **GRU**
7. **Linear** *(Bonus)*
8. **Dropout** *(Bonus)*

### Recurrent Weights (RNN, LSTM, GRU)
PyTorch stores the recurrent weights densely packed in blocks (e.g. `weight_ih_l0`, `weight_hh_l0`). 
* For **LSTM**, PyTorch concatenates the gates as `[Input, Forget, Cell, Output]`.
* For **GRU**, PyTorch concatenates the gates as `[Reset, Update, New]`.
The parser extracts these weights perfectly. (*Note: When extending this to push weights into SOFIE `ROperator` objects, `[i, f, g, o]` must be reordered to ONNX's expected `[i, o, f, c]` layout before memory mapping*).

You can run `test_impressive.py` to test the module extraction logic on a comprehensive PyTorch model featuring all 8 requested layers.

---

## Exercise 5-1: Enhancing the SOFIE Keras Parser

For the bonus exercise, I extended `bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser.py`.

### Changes Made:
1. **Activated GRU and LSTM**: The core `SOFIE.ROperator_GRU` and `SOFIE.ROperator_LSTM` handlers were already written in `rnn.py`, but were disabled in the main `mapKerasLayer` dictionary. I uncommented them, successfully exposing standard Recurrent parsing to Keras.
2. **Added Conv2DTranspose**: Wrote `layers/conv_transpose.py` to extract `kernel_size`, `strides`, and `padding`, and dynamically build `SOFIE.ROperator_ConvTranspose`.
3. **Added Conv1DTranspose** *(Bonus Feature)*: Went a step further and built `layers/conv1d_transpose.py` to ensure comprehensive support for all dimensionality formats on Transposed convolutions.
4. **Fixed Channels_First Reshaping**: Modified `parser.py`'s activation transpose logic. Keras utilizes `channels_last` layout by default, but ONNX/SOFIE requires `channels_first`. The new generators gracefully apply PyROOT `ROperator_Transpose` nodes utilizing `[0, 2, 1]` or `[0, 3, 1, 2]` inversions perfectly around the new convolutions.

You can view the new Keras handlers in the `bindings/.../layers/` folders. 
