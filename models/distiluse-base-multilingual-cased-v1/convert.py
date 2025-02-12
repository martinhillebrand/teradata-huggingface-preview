import os
import json
import shutil

from optimum.exporters.onnx import main_export
import onnx
from onnxconverter_common import float16
import onnxruntime as rt
from onnxruntime.tools.onnx_model_utils import *
from onnxruntime.quantization import quantize_dynamic, QuantType

with open('conversion_config.json') as json_file:
    conversion_config = json.load(json_file)


    model_id = conversion_config["model_id"]
    number_of_generated_embeddings = conversion_config["number_of_generated_embeddings"]
    precision_to_filename_map = conversion_config["precision_to_filename_map"]
    opset = conversion_config["opset"]
    IR = conversion_config["IR"]

    
    op = onnx.OperatorSetIdProto()
    op.version = opset
    
    
    if not os.path.exists("onnx"):
        os.makedirs("onnx")

    print("Exporting the main model version")

    main_export(model_name_or_path=model_id, output="./",  opset=opset, trust_remote_code=True, task="feature-extraction", dtype="fp32")
    
    if "fp32" in precision_to_filename_map:
        print("Exporting the fp32 onnx file...")
        
        shutil.copyfile('model.onnx', precision_to_filename_map["fp32"])
        
        print("Done\n\n")
    if "fp16" in precision_to_filename_map:
        print("Exporting the fp16 onnx file...")
        model_fp16 = float16.convert_float_to_float16(onnx.load('model.onnx'),\
                                                      min_positive_val=1e-7, \
                                                      max_finite_val=1e4, \
                                                      keep_io_types=True, \
                                                      disable_shape_infer=False, \
                                                      op_block_list=None, \
                                                      node_block_list=None)
        
        model_fp16 = onnx.helper.make_model(model_fp16.graph, ir_version = IR, opset_imports = [op]) #to be sure that we have compatible opset and IR version
        
        onnx.save(model_fp16, precision_to_filename_map["fp16"])
        print("Done\n\n")

    if "int8" in precision_to_filename_map:
        print("Quantizing fp32 model to int8...")
        quantize_dynamic("model.onnx",  precision_to_filename_map["int8"], weight_type=QuantType.QInt8)
        print("Done\n\n")
        
    if "uint8" in precision_to_filename_map:
        print("Quantizing fp32 model to uint8...")
        quantize_dynamic("model.onnx", precision_to_filename_map["uint8"], weight_type=QuantType.QUInt8)
        print("Done\n\n")
        
    os.remove("model.onnx")
