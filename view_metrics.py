import numpy as np

# Đọc file metrics
metrics = np.load('metrics/final_training_metrics.npy', allow_pickle=True).item()

# Hiển thị toàn bộ nội dung
for key, value in metrics.items():
    print(f"{key}: {value}\n")
    