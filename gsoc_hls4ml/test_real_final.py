import tensorflow as tf
import hls4ml
import ROOT
import ParseHLS4ML_real

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Step 1: Building Keras model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()

print("\nStep 2: Converting to HLS4ML...")
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=hls_config,
    output_dir='real_hls4ml_model',
    backend='Vivado'
)

print("\nStep 3: Inspecting adapted layers...")
adapter = ParseHLS4ML_real.RealModelGraphAdapter(hls_model)
for layer in adapter.get_layers():
    print(f"  Layer: {layer.name} | class: {layer.class_name}")
    print(f"    inputs:  {layer.inputs}")
    print(f"    outputs: {layer.outputs}")
    print(f"    attributes: {layer.attributes}")
    for w in layer.get_weights():
        print(f"    weight: {w.name}, shape: {w.data.shape}")

print("\nStep 4: Parsing into SOFIE RModel...")
rmodel = ParseHLS4ML_real.Parse(hls_model, "RealHLS4MLModel")

print("\nStep 5: Generating C++ inference header...")
rmodel.Generate()
rmodel.OutputGenerated()

print("\nStep 6: Compiling generated header...")
ROOT.gInterpreter.Declare('#include "RealHLS4MLModel.hxx"')

print("\n SUCCESS: Real HLS4ML model parsed, generated, and compiled!")
print("Generated file: RealHLS4MLModel.hxx")
