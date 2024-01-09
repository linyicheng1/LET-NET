import os.path as osp

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

# your test input array
test_arr = np.random.randn(1, 3, 224, 224).astype(np.float32)

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torchvision.models.resnet50(pretrained=True).cuda().eval()
print('pytorch result:', model(torch.from_numpy(test_arr).cuda()))

input_names = ["input"]
output_names = ["output"]

if not osp.exists('resnet50.onnx'):
    # translate your pytorch model to onnx
    torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True, input_names=input_names, output_names=output_names)

# model = onnx.load("resnet50.onnx")
# ort_session = ort.InferenceSession('resnet50.onnx')
# outputs = ort_session.run(None, {'input': test_arr})
#
# print('onnx result:', outputs[0])