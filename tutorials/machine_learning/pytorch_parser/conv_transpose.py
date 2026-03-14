from .. import get_keras_version

def MakeKerasConv2DTranspose(layer):
    """
    Create a Keras-compatible transposed convolutional layer operation using SOFIE.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1] if len(fWeightNames) > 1 else ""

    fAttrDilations = attributes["dilation_rate"]
    fAttrGroup = 1 # Keras Conv2DTranspose usually has group=1
    fAttrKernelShape = attributes["kernel_size"]
    fKerasPadding = str(attributes["padding"])
    fAttrStrides = attributes["strides"]
    
    # Optional output padding
    fAttrOutputPadding = attributes.get("output_padding")
    if fAttrOutputPadding is None:
        fAttrOutputPadding = [0, 0]

    # outputShape is empty — SOFIE infers it from autopad + strides
    fAttrOutputShape = []

    fAttrPads = []

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "SAME_UPPER" 
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet supports Conv2DTranspose padding " + fKerasPadding
        )

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrOutputPadding,
            fAttrOutputShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported type for Conv2DTranspose: " + fLayerDType)
