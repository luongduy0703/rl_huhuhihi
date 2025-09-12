# Thông Số Kỹ Thuật Hệ Thống Robot Arm - Tiếng Việt

## 🔧 Phân Tích Bộ Điều Khiển Phần Cứng

### Tổng Quan Bộ Điều Khiển PWM PCA9685
PCA9685 là bộ điều khiển PWM (Điều Chế Độ Rộng Xung) 16 kênh giao tiếp với Raspberry Pi qua giao thức I2C.

### Giải Thích Các Thông Số Chính

#### Thông Số Độ Rộng Xung
```python
min_pulse: int = 500    # Độ rộng xung tối thiểu (micro giây)
max_pulse: int = 2500   # Độ rộng xung tối đa (micro giây)
```

**Ý nghĩa của chúng:**
- **min_pulse (500μs)**: Độ rộng xung tương ứng với vị trí 0° của servo
- **max_pulse (2500μs)**: Độ rộng xung tương ứng với vị trí 180° của servo
- **Điều khiển servo tiêu chuẩn**: Hầu hết servo thông thường mong đợi 1000-2000μs, nhưng 500-2500μs cho phạm vi tốt hơn

#### Cách Độ Rộng Xung Điều Khiển Servo:
1. **Tín Hiệu PWM**: Tần số 50Hz (chu kỳ 20ms)
2. **Độ Rộng Xung**: Thời gian tín hiệu HIGH trong mỗi chu kỳ 20ms
3. **Phản Ứng Servo**: Bộ điều khiển servo bên trong chuyển độ rộng xung thành vị trí

```
Tính Toán Góc Servo:
góc_theo_độ = (độ_rộng_xung - min_pulse) / (max_pulse - min_pulse) * 180

Ví dụ:
- 500μs  → 0°
- 1500μs → 90° (trung tính)
- 2500μs → 180°
```

### Quy Trình Xử Lý Hoàn Chỉnh

#### 1. Quá Trình Khởi Tạo
```python
def __init__(self, num_servos=4, min_pulse=500, max_pulse=2500):
```

**Khởi tạo từng bước:**
1. **Phát Hiện Phần Cứng**: Kiểm tra thư viện CircuitPython có sẵn
2. **Thiết Lập I2C**: Khởi tạo bus I2C (chân SCL/SDA)
3. **Cấu Hình PCA9685**: Đặt tần số 50Hz cho điều khiển servo
4. **Theo Dõi Vị Trí**: Khởi tạo mảng current_positions
5. **Giới Hạn An Toàn**: Đặt giới hạn góc (0-180°) cho mỗi servo
6. **Vị Trí Trung Tính**: Di chuyển tất cả servo về 90° vị trí bắt đầu

#### 2. Chuyển Đổi Góc-sang-Xung
```python
def angle_to_pulse(self, angle: float) -> int:
    pulse = self.min_pulse + (angle / 180.0) * (self.max_pulse - self.min_pulse)
    return int(pulse)
```

**Quá Trình Toán Học:**
- Đầu vào: Góc (0-180 độ)
- Nội suy tuyến tính giữa min_pulse và max_pulse
- Đầu ra: Độ rộng xung tính bằng micro giây

**Ví Dụ Tính Toán:**
- 0°   → 500 + (0/180) × (2500-500) = 500μs
- 90°  → 500 + (90/180) × (2500-500) = 1500μs  
- 180° → 500 + (180/180) × (2500-500) = 2500μs

#### 3. Quá Trình Điều Khiển Servo
```python
def set_servo_angle(self, servo_id: int, angle: float) -> bool:
```

**Các Bước Xử Lý:**
1. **Xác Thực**: Kiểm tra servo_id (0-15) và giới hạn góc
2. **Giới Hạn**: Đảm bảo góc trong giới hạn servo
3. **Chuyển Đổi**: góc → độ rộng xung → duty cycle 16-bit
4. **Ghi Phần Cứng**: Gửi tín hiệu PWM tới kênh PCA9685
5. **Cập Nhật Trạng Thái**: Cập nhật theo dõi current_positions
6. **Xử Lý Lỗi**: Trả về trạng thái thành công/thất bại

**Tính Toán Duty Cycle 16-bit:**
```python
duty_cycle = int(pulse * 65535 / 20000)
```
- 65535: Giá trị 16-bit tối đa
- 20000: Chu kỳ 20ms tính bằng micro giây
- Kết quả: Tỷ lệ thời gian HIGH trong tín hiệu PWM

### Thông Số Đầu Vào/Đầu Ra

#### Đầu Vào
1. **ID Servo**: Số nguyên (0-15, thường 0-3 cho cánh tay 4 khớp)
2. **Góc Mục Tiêu**: Số thực (0.0-180.0 độ)
3. **Thông Số Chuyển Động**: Bước, độ trễ cho chuyển động mượt
4. **Giới Hạn An Toàn**: Góc min/max cho mỗi khớp

#### Đầu Ra
1. **Tín Hiệu PWM**: PWM 50Hz tới servo motor
2. **Phản Hồi Vị Trí**: Mảng góc khớp hiện tại
3. **Mã Trạng Thái**: Trả về boolean thành công/thất bại
4. **Thông Tin Debug**: Đầu ra console để giám sát

#### Kết Nối Phần Cứng
```
Raspberry Pi 4 → PCA9685 → Servo
GPIO 2 (SDA) ──┐
GPIO 3 (SCL) ──┤ Bus I2C → Board PCA9685
Nguồn 5V   ────┤
Ground     ────┘

Kênh PCA9685:
Kênh 0 → Servo Đế (Khớp 1)
Kênh 1 → Servo Vai (Khớp 2)  
Kênh 2 → Servo Khuỷu (Khớp 3)
Kênh 3 → Servo Cổ Tay (Khớp 4)
```

### Tính Năng An Toàn

#### 1. Giới Hạn Góc
```python
self.angle_limits = [(0, 180) for _ in range(self.num_servos)]
```
- Ngăn chặn hỏng servo do xoay quá mức
- Có thể tùy chỉnh theo từng khớp
- Bảo vệ phần cứng

#### 2. Xác Thực Đầu Vào
- Kiểm tra phạm vi ID servo
- Giới hạn góc trong phạm vi hợp lệ
- Xử lý ngoại lệ cho lỗi phần cứng

#### 3. Chuyển Động Mượt
```python
def move_servo_smoothly(self, servo_id, target_angle, steps=20, delay=0.05):
```
- Ngăn chặn chuyển động giật đột ngột
- Giảm ứng suất cơ học
- Điều khiển tốc độ có thể cấu hình

### Đặc Tính Hiệu Suất

#### Thông Số Thời Gian
- **Tần Số PWM**: 50Hz (chu kỳ 20ms)
- **Tốc Độ Cập Nhật**: tối đa ~50 cập nhật/giây
- **Chuyển Động Mượt**: Có thể cấu hình (mặc định: 20 bước, độ trễ 0.05s)
- **Tốc Độ I2C**: Tiêu chuẩn 100kHz

#### Độ Chính Xác
- **Độ Phân Giải Góc**: ~0.35° (500 bước trên 180°)
- **Độ Phân Giải Xung**: ~4μs (phạm vi 2000μs / 500 bước)
- **Độ Lặp Lại**: ±1° điển hình cho servo thông thường

### Xử Lý Lỗi

#### Các Tình Huống Lỗi Thông Thường
1. **Phần Cứng Không Kết Nối**: Chuyển về chế độ mô phỏng
2. **ID Servo Không Hợp Lệ**: Trả về False, ghi log lỗi
3. **Lỗi Giao Tiếp I2C**: Bắt ngoại lệ, logic thử lại
4. **Góc Ngoài Phạm Vi**: Tự động giới hạn về giới hạn

#### Ví Dụ Đầu Ra Debug
```
PCA9685 khởi tạo thành công
Thư viện phần cứng không có sẵn - chạy ở chế độ mô phỏng
Lỗi đặt servo 2 về góc 200: Góc được giới hạn về 180
ID servo không hợp lệ: 8
```

---

## 🎯 Thông Số Hệ Thống Chính

### **Thông Số Phần Cứng**
```python
# Bộ Điều Khiển PWM PCA9685
frequency = 50          # Hz - Tần số servo tiêu chuẩn
min_pulse = 500         # μs - Vị trí 0°
max_pulse = 2500        # μs - Vị trí 180° (THÔNG SỐ BẠN ĐÃ CHỌN)
num_servos = 4          # Số lượng khớp

# Giao Tiếp I2C
i2c_address = 0x40      # Địa chỉ PCA9685 mặc định
scl_pin = GPIO3         # Đường clock I2C
sda_pin = GPIO2         # Đường dữ liệu I2C
```

### **Thông Số Môi Trường RL**
```python
# Không Gian Trạng Thái
state_size = 10         # 4 khớp + 3 mục tiêu + 3 đầu_cánh_tay
joint_limits = (0, 180) # Độ mỗi khớp
workspace = cube(0.5m)  # Thể tích có thể với tới

# Không Gian Hành Động  
action_size = 4         # Một cho mỗi khớp
action_range = (-1, +1) # Điều khiển liên tục
max_angle_change = 15   # Độ mỗi bước (an toàn)

# Hàm Phần Thưởng
distance_weight = 1.0   # Hệ số cải thiện khoảng cách
target_bonus = 10.0     # Thưởng đạt mục tiêu (<5cm)
movement_penalty = 0.01 # Phạt di chuyển lớn
```

### **Thông Số Học Tập**
```python
# DDPG Agent
actor_lr = 0.001        # Tốc độ học mạng Actor
critic_lr = 0.002       # Tốc độ học mạng Critic
memory_size = 100000    # Bộ đệm experience replay
batch_size = 32         # Kích thước batch huấn luyện
tau = 0.005            # Tốc độ cập nhật soft target network

# Huấn Luyện
episodes = 1000         # Episode huấn luyện
max_steps = 200         # Bước mỗi episode
exploration_noise = 0.1 # Nhiễu hành động để khám phá
```

---

## 🚀 Hướng Dẫn Bắt Đầu - Quy Trình Hoàn Chỉnh

### **1. Demo Nhanh (Không Phần Cứng)**
```bash
# Chạy phiên bản đơn giản đang hoạt động
python3 simple_train.py --mode train --episodes 10
python3 simple_train.py --mode demo

# Xem pipeline xử lý hoàn chỉnh
python3 system_flow_demo.py
```

### **2. Thiết Lập Phần Cứng (Raspberry Pi)**
```bash
# Cài đặt phụ thuộc
pip3 install adafruit-circuitpython-pca9685

# Kết nối phần cứng:
# GPIO2 (SDA) → PCA9685 SDA
# GPIO3 (SCL) → PCA9685 SCL  
# 5V → PCA9685 VCC
# GND → PCA9685 GND

# Kiểm tra phần cứng
python3 robot_arm_controller.py
```

### **3. Huấn Luyện RL Đầy Đủ**
```bash
# Huấn luyện mô phỏng (không phần cứng)
python3 main.py --mode train --no-robot --episodes 100

# Huấn luyện phần cứng (robot thật)
python3 main.py --mode train --hardware --episodes 50

# Điều khiển thủ công
python3 main.py --mode manual --hardware
```

---

Đây là bản dịch tiếng Việt hoàn chỉnh của tài liệu kỹ thuật hệ thống robot arm sử dụng deep reinforcement learning!
