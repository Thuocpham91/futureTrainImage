from tflite_support.metadata import schema_py_generated as _metadata_fb
from tflite_support.metadata import metadata_schema_py_generated as _metadata_schema_fb
from tflite_support.metadata import metadata as _metadata
from tflite_support import flatbuffers

def get_model_metadata(model_path):
    with open(model_path, 'rb') as f:
        model_buf = f.read()

    model = _metadata_fb.Model.GetRootAsModel(model_buf, 0)
    metadata_buf = model.MetadataAsNumpy()
    
    if not metadata_buf:
        print("The model does not have metadata.")
        return
    
    metadata = _metadata_fb.ModelMetadata.GetRootAsModelMetadata(metadata_buf, 0)
    print("Metadata name: ", metadata.Name())
    
    # Displaying inputs and outputs
    for i in range(metadata.SubgraphMetadataLength()):
        subgraph = metadata.SubgraphMetadata(i)
        print("\nSubgraph name: ", subgraph.Name())
        
        for j in range(subgraph.InputTensorMetadataLength()):
            input_tensor = subgraph.InputTensorMetadata(j)
            print(f"Input tensor {j} name: ", input_tensor.Name())
            print(f"Input tensor {j} description: ", input_tensor.Description())
        
        for j in range(subgraph.OutputTensorMetadataLength()):
            output_tensor = subgraph.OutputTensorMetadata(j)
            print(f"Output tensor {j} name: ", output_tensor.Name())
            print(f"Output tensor {j} description: ", output_tensor.Description())
    
    # Displaying signatures
    if hasattr(model, 'SignatureDefsLength') and model.SignatureDefsLength() > 0:
        for i in range(model.SignatureDefsLength()):
            signature = model.SignatureDefs(i)
            print(f"\nSignature {i} key: ", signature.Key().decode('utf-8'))
            print(f"Signature {i} name: ", signature.Name())
            if signature.HasInputs():
                print("Inputs:")
                for input_key in signature.InputsKeysAsNumpy():
                    print(f"  {signature.Inputs(input_key).decode('utf-8')}")
            if signature.HasOutputs():
                print("Outputs:")
                for output_key in signature.OutputsKeysAsNumpy():
                    print(f"  {signature.Outputs(output_key).decode('utf-8')}")
    else:
        print("No signatures found in the model.")

# Path to your TFLite model
model_path = "model.tflite"

get_model_metadata(model_path)