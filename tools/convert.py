import onnx

model = onnx.load('../model/letnet-gray.onnx')
inputs = model.graph.input  # inputs是一个列表，可以操作多输入~
inputs[0].type.tensor_type.shape.dim[0].dim_value = 1
inputs[0].type.tensor_type.shape.dim[1].dim_value = 1
inputs[0].type.tensor_type.shape.dim[2].dim_value = 640
inputs[0].type.tensor_type.shape.dim[3].dim_value = 480
onnx.save(model, '../model/model640*480*1.onnx')
