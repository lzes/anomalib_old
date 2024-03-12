import onnxruntime as ort
import numpy as np
import time

# 创建 ONNX Runtime 会话
ort_session = ort.InferenceSession("results/efficient_ad/anomaly_detection/run/weights/onnx/model.onnx")

# 准备输入数据，这里使用随机数据作为示例
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
input_shape = ort_session.get_inputs()[0].shape

# 替换动态维度（例如：使用批大小为 1）
input_shape = [1 if dim is None or isinstance(dim, str) else dim for dim in input_shape]

# 生成随机输入数据
input_data = np.random.randn(*input_shape).astype(np.float32)

# 热身运行，确保不受初始缓存加载的影响
for _ in range(10):
    ort_session.run([output_name], {input_name: input_data})

# 实际测量
start_time = time.time()
for _ in range(100):
    ort_session.run([output_name], {input_name: input_data})
end_time = time.time()

# 计算平均推理时间
avg_inference_time = (end_time - start_time) / 100
print(f"Average inference time: {avg_inference_time * 1000} ms")
