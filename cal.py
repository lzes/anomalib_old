import torch
import torch.onnx
import onnx

# 加载导出的 ONNX 模型
onnx_model = onnx.load("results/efficient_ad/anomaly_detection/run/weights/onnx/model.onnx")

# 检查模型是否有有效的结构
onnx.checker.check_model(onnx_model)

# 使用 ONNX Python API 查看模型中的算子
ls = []
for node in onnx_model.graph.node:
    ls.append(node.op_type)
    print(node.op_type)
print(len(ls), len(set(ls)))
print('List of different words:')
for word in sorted(set(ls)):
    print(word)