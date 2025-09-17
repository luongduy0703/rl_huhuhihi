import pickle

# Đường dẫn tới file lưu tham số mạng
model_path = 'models/robot_arm_params.pkl'

with open(model_path, 'rb') as f:
    data = pickle.load(f)

print('Loại dữ liệu lưu:', type(data))

if isinstance(data, dict):
    print('Các key trong file:')
    for key in data:
        print(f'- {key}: {type(data[key])}')
        # Nếu là numpy array hoặc tensor, in kích thước
        try:
            shape = getattr(data[key], 'shape', None)
            if shape:
                print(f'  Kích thước: {shape}')
        except Exception:
            pass
else:
    print(data)
