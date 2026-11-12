import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog, local_binary_pattern

# ==========================================
# 1. CẤU HÌNH THAM SỐ
# ==========================================
DATA_DIR = 'dataset'        # Thư mục ảnh đã làm sạch (tạo bởi tai_du_lieu.py)
# 8 lớp phương tiện. Vị trí trong list chính là nhãn số: xe_dap = 0, xe_may = 1, ..., tau_hoa = 7
CLASSES = ['xe_dap', 'xe_may', 'o_to', 'xe_buyt', 'xe_tai', 'may_bay', 'tau_thuy', 'tau_hoa']
# Tên tiếng Việt có dấu, chỉ dùng để ghi lên biểu đồ cho dễ đọc
TEN_LOP = ['Xe đạp', 'Xe máy', 'Ô tô', 'Xe buýt', 'Xe tải', 'Máy bay', 'Tàu thủy', 'Tàu hỏa']
IMG_SIZE = 64               # Mọi ảnh được đưa về 64x64 trước khi trích đặc trưng
TRAIN_RATIO = 0.8           # 80% ảnh mỗi lớp để train, 20% để test
EPOCHS = 30                 # Thử 40 epoch không tốt hơn: độ chính xác test đã chững lại từ khoảng epoch 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHTS_FILE = 'weight_ann_phuong_tien.pth'    # File lưu trọng số mạng sau khi train
USE_SAVED_WEIGHTS = True   # True: Dùng weights đã lưu, False: Train từ đầu

# Cố định seed để mỗi lần chạy cho ra cùng một kết quả
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# ==========================================
# 2. ĐỌC DANH SÁCH ẢNH & CHIA TRAIN/TEST THEO TỪNG LỚP
# ==========================================
# Tạo bảng gồm 2 cột: đường dẫn ảnh và nhãn số của ảnh đó
rows = []
for label, class_name in enumerate(CLASSES):
    for path in sorted(glob.glob(f'{DATA_DIR}/{class_name}/*.jpg')):
        rows.append({'path': path, 'label': label})
df = pd.DataFrame(rows)
print(f"Tổng số ảnh: {len(df)}")

# Chia train/test RIÊNG cho từng lớp (giống trichdactrung.py):
# lớp nào cũng có đúng 80% train và 20% test, không lớp nào bị dồn hết vào một bên.
# Chia theo ĐƯỜNG DẪN ảnh (chưa trích đặc trưng) để bước sau chỉ thêm ảnh lật vào tập train.
train_list = []
test_list = []
for label, group in df.groupby("label"):
    # Xáo trộn ngẫu nhiên các ảnh trong lớp (random_state=42 để lần nào chạy cũng chia giống nhau)
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(TRAIN_RATIO * len(group))
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)
train_paths = train_df['path'].tolist()
test_paths = test_df['path'].tolist()
print(f"Số ảnh train: {len(train_paths)}, số ảnh test: {len(test_paths)}")

# Trực quan hóa dữ liệu: biểu đồ cột số ảnh train/test của mỗi lớp
train_counts = train_df['label'].value_counts().sort_index().values
test_counts = test_df['label'].value_counts().sort_index().values
x_pos = np.arange(len(CLASSES))
plt.figure(figsize=(10, 5))
plt.bar(x_pos - 0.2, train_counts, width=0.4, label='Train', color='steelblue')
plt.bar(x_pos + 0.2, test_counts, width=0.4, label='Test', color='orange')
plt.xticks(x_pos, TEN_LOP)
plt.title('Phân bố số ảnh train/test của mỗi lớp')
plt.xlabel('Lớp phương tiện')
plt.ylabel('Số ảnh')
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('phan_bo_du_lieu.png')
plt.close()
print("Đã lưu biểu đồ phân bố dữ liệu vào 'phan_bo_du_lieu.png'")

# ==========================================
# 3. HÀM TRÍCH ĐẶC TRƯNG (HOG + MÀU + KẾT CẤU)
# ==========================================
# Mạng ANN (MLP) không tự nhìn được ảnh như CNN, nên ta tự biến mỗi ảnh thành 1 vector số.
# Dùng 5 loại đặc trưng, mỗi loại bổ sung một "góc nhìn" (đã thử bỏ từng loại: độ chính xác đều giảm):
# - HOG (Histogram of Oriented Gradients): mô tả HÌNH DẠNG qua hướng các cạnh trong ảnh
#   (bánh xe tròn, thân xe dài, cánh máy bay, đường ray...). Tính ở 2 cỡ ảnh: 64x64 thấy chi tiết,
#   32x32 thấy dáng tổng thể.
# - Histogram màu HSV theo 3 dải ngang (trên/giữa/dưới): mô tả BỐI CẢNH và VỊ TRÍ màu
#   (trời xanh ở trên -> máy bay, nước xanh ở dưới -> tàu thủy, đường nhựa xám ở dưới -> ô tô)
# - LBP (Local Binary Pattern): mô tả KẾT CẤU bề mặt (mặt nước gợn sóng khác bầu trời trơn)
# - Ảnh thu nhỏ 8x8: bố cục màu thô của cả bức ảnh

def color_histogram(hsv_part):
    # Histogram màu của 1 vùng ảnh HSV (Hue = màu gì, Saturation = độ đậm, Value = độ sáng):
    # mỗi kênh chia 16 khoảng giá trị, đếm số pixel rơi vào từng khoảng => 3 * 16 = 48 số.
    # Chia cho số pixel của vùng để thành tỉ lệ (0..1), không phụ thuộc kích thước vùng.
    so_pixel = hsv_part.shape[0] * hsv_part.shape[1]
    hist_list = []
    for c in range(3):
        hist, _ = np.histogram(hsv_part[:, :, c], bins=16, range=(0, 256))
        hist_list.append(hist / so_pixel)
    return np.concatenate(hist_list)

def extract_features(img):
    # Đưa ảnh về RGB và cùng kích thước 64x64 để vector đặc trưng của mọi ảnh dài bằng nhau
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img)

    # (1) HOG trên ảnh MÀU 64x64 (channel_axis=-1: mỗi pixel lấy kênh màu có cạnh rõ nhất,
    #     nên thấy được cả cạnh giữa 2 vùng khác màu nhưng cùng độ sáng, ảnh xám sẽ bỏ sót).
    #     Chia ô 8x8 pixel -> 8x8 ô; gom khối 2x2 ô -> 7x7 khối; mỗi khối 4 ô x 9 hướng = 36 số
    #     => 7 * 7 * 36 = 1764 số
    hog_64 = hog(rgb, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)

    # (2) HOG trên ảnh thu nhỏ 32x32: 4x4 ô -> 3x3 khối x 36 = 324 số (dáng tổng thể của xe)
    rgb_32 = np.array(img.resize((32, 32)))
    hog_32 = hog(rgb_32, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)

    # (3) Histogram màu HSV cho 3 dải ngang: hàng 0-20 (trên), 21-41 (giữa), 42-63 (dưới)
    #     => 3 dải x 48 = 144 số
    hsv = np.array(img.convert('HSV'))
    color_features = []
    for top, bottom in [(0, 21), (21, 42), (42, 64)]:
        color_features.append(color_histogram(hsv[top:bottom]))

    # (4) LBP trên ảnh xám: so mỗi pixel với 8 pixel xung quanh -> mã kết cấu từ 0 đến 9
    #     (method='uniform'). Đếm tỉ lệ từng mã trên cả ảnh và trên 4 góc ảnh (2x2)
    #     => 5 vùng x 10 mã = 50 số
    gray = np.array(img.convert('L'))
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_features = []
    for vung in [lbp, lbp[:32, :32], lbp[:32, 32:], lbp[32:, :32], lbp[32:, 32:]]:
        hist, _ = np.histogram(vung, bins=10, range=(0, 10))
        lbp_features.append(hist / vung.size)

    # (5) Ảnh thu nhỏ 8x8 pixel x 3 kênh màu, chia 255 về khoảng 0..1 => 192 số
    tiny = np.array(img.resize((8, 8))).reshape(-1) / 255.0

    # Nối lại thành 1 vector: 1764 + 324 + 144 + 50 + 192 = 2474 số
    return np.concatenate([hog_64, hog_32] + color_features + lbp_features + [tiny]).astype(np.float32)

# Minh họa đầu vào của mạng: mỗi lớp lấy 1 ảnh, hàng trên là ảnh gốc, hàng dưới là ảnh HOG
plt.figure(figsize=(16, 5))
for label, class_name in enumerate(CLASSES):
    # Lấy ảnh train đầu tiên của lớp này
    sample_path = train_df[train_df['label'] == label]['path'].iloc[0]
    sample_img = Image.open(sample_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    # visualize=True: hàm hog trả thêm 1 ảnh vẽ các hướng cạnh để ta nhìn được
    _, hog_image = hog(np.array(sample_img), orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), channel_axis=-1, visualize=True)

    plt.subplot(2, len(CLASSES), label + 1)
    plt.imshow(sample_img)
    plt.title(TEN_LOP[label])
    plt.axis('off')

    plt.subplot(2, len(CLASSES), label + 1 + len(CLASSES))
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG')
    plt.axis('off')
plt.suptitle('Ảnh gốc (64x64) và đặc trưng HOG tương ứng')
plt.tight_layout()
plt.savefig('anh_mau_va_hog.png')
plt.close()
print("Đã lưu ảnh minh họa HOG vào 'anh_mau_va_hog.png'")

# ==========================================
# 4. TRÍCH ĐẶC TRƯNG CHO TOÀN BỘ ẢNH
# ==========================================
print("Đang trích đặc trưng ảnh train...")
X_train = []
y_train = []
for i, (path, label) in enumerate(zip(train_df['path'], train_df['label'])):
    img = Image.open(path).convert('RGB')
    # Ảnh gốc
    X_train.append(extract_features(img))
    y_train.append(label)
    # Tăng cường dữ liệu: thêm ảnh lật ngang. Xe quay đầu trái hay phải vẫn là cùng một loại xe,
    # nên ta có gấp đôi dữ liệu train mà không phải tải thêm ảnh.
    X_train.append(extract_features(img.transpose(Image.FLIP_LEFT_RIGHT)))
    y_train.append(label)
    if (i + 1) % 500 == 0:
        print(f"  Đã xử lý {i + 1}/{len(train_df)} ảnh train")

print("Đang trích đặc trưng ảnh test...")
X_test = []
y_test = []
for i, (path, label) in enumerate(zip(test_df['path'], test_df['label'])):
    # Tập test chỉ dùng ảnh gốc, không lật, để đánh giá đúng với ảnh thật
    img = Image.open(path).convert('RGB')
    X_test.append(extract_features(img))
    y_test.append(label)
    if (i + 1) % 500 == 0:
        print(f"  Đã xử lý {i + 1}/{len(test_df)} ảnh test")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ==========================================
# 5. CHUẨN HÓA Z-SCORE
# ==========================================
# TỰ CODE CLASS CHUẨN HÓA Z-SCORE (THAY THẾ SKLEARN) - chép theo btap1.py
class ManualZScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        # Chuyển sang numpy array để tính toán nhanh
        data_np = np.array(data)
        # Tính trung bình (mean) theo từng cột (axis=0)
        self.mean_ = np.mean(data_np, axis=0)
        # Tính độ lệch chuẩn (std) theo từng cột
        self.std_ = np.std(data_np, axis=0)

        # Tránh lỗi chia cho 0 nếu một cột có giá trị cố định không đổi.
        # Ở đây dùng 1.0 (không dùng 1e-8 như btap1.py): một số ô HOG luôn bằng 0 ở tập train,
        # nếu ảnh test có giá trị khác 0 ở ô đó mà chia cho 1e-8 thì số sẽ nổ lên hàng triệu.
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

    def transform(self, data):
        data_np = np.array(data)
        # Áp dụng công thức Z = (x - mean) / std
        return (data_np - self.mean_) / self.std_

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data_np = np.array(data)
        # Áp dụng công thức đảo ngược: x = Z * std + mean
        return data_np * self.std_ + self.mean_

# Đưa mọi đặc trưng về cùng thang đo (trung bình 0, độ lệch chuẩn 1) để mạng học ổn định hơn.
# Chỉ fit (tính mean/std) trên tập train, rồi dùng đúng mean/std đó cho tập test,
# vì khi dự đoán thật ta không được biết trước dữ liệu test.
scaler = ManualZScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. CHUYỂN SANG TENSOR & DATALOADER
# ==========================================
# Đặc trưng là số thực -> float32; nhãn lớp là số nguyên -> long (CrossEntropyLoss yêu cầu kiểu long)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Dùng mini-batch (mỗi lần 64 mẫu) thay vì "Siêu Batch" (cả tập 1 lần) như btap1.py:
# mỗi epoch trọng số được cập nhật hàng trăm lần thay vì 1 lần, nên học nhanh hơn,
# và nhiễu nhỏ giữa các batch giúp mạng tổng quát hóa tốt hơn (bớt học thuộc).
# shuffle=True: mỗi epoch xáo trộn lại thứ tự để các batch không lặp lại y hệt nhau.
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
print(f"Số batch mỗi epoch: {len(train_loader)}")

# ==========================================
# 7. XÂY DỰNG MẠNG NƠ-RON ANN (MLP)
# ==========================================
# Số nơ-ron lớp vào = số đặc trưng của 1 ảnh (2474, xem khối 3).
# Lấy trực tiếp từ dữ liệu giống cách btap1.py tính input_nodes, nên đổi đặc trưng thì mạng tự khớp.
input_nodes = X_train_tensor.shape[1]

model = nn.Sequential(
    # Dropout ở đầu vào: mỗi bước train che ngẫu nhiên 30% đặc trưng, buộc mạng nhìn nhiều đặc trưng
    # cùng lúc thay vì học thuộc vài ô HOG -> chống học thuộc (overfitting), tăng độ chính xác test rõ nhất
    nn.Dropout(0.3),
    # Lớp ẩn 1: 2474 -> 512 nơ-ron, nén vector đặc trưng dài thành các "mẫu" hình dạng/màu sắc
    nn.Linear(in_features=input_nodes, out_features=512),
    # ReLU thay cho Tanh: không bị bão hòa (đạo hàm không tiến về 0 khi giá trị lớn) nên gradient truyền tốt hơn
    nn.ReLU(),
    # Dropout: mỗi bước train tắt ngẫu nhiên 50% nơ-ron, buộc mạng không dựa dẫm vào vài nơ-ron -> chống học thuộc
    nn.Dropout(0.5),
    # Lớp ẩn 2: 512 -> 128 nơ-ron, tiếp tục nén thông tin lại, chỉ giữ phần quan trọng để phân biệt các lớp
    nn.Linear(in_features=512, out_features=128),
    nn.ReLU(),
    nn.Dropout(0.5),
    # Lớp ra: 8 nơ-ron ứng với 8 lớp phương tiện. Không cần Softmax ở đây vì CrossEntropyLoss đã tự làm
    nn.Linear(in_features=128, out_features=len(CLASSES))
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số của mạng: {total_params:,}")

# CrossEntropyLoss: hàm mất mát chuẩn cho bài toán phân loại nhiều lớp
criterion = nn.CrossEntropyLoss()
# Adam: thuật toán tối ưu tự điều chỉnh bước học cho từng trọng số.
# weight_decay=1e-4: phạt nhẹ các trọng số quá lớn, thêm một cách chống học thuộc
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ==========================================
# 8. QUÁ TRÌNH HUẤN LUYỆN (hoặc Load weights từ file)
# ==========================================
if USE_SAVED_WEIGHTS and os.path.exists(WEIGHTS_FILE):
    print(f"Đang load weights từ file '{WEIGHTS_FILE}'...")
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    print("Đã load weights thành công! Bỏ qua quá trình huấn luyện.")
else:
    print("Bắt đầu huấn luyện mạng ANN...")
    losses = []         # Loss trung bình của mỗi epoch
    train_accs = []     # Độ chính xác trên tập train sau mỗi epoch
    test_accs = []      # Độ chính xác trên tập test sau mỗi epoch
    for epoch in range(EPOCHS):
        # model.train(): bật chế độ huấn luyện (Dropout hoạt động)
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass: đưa 1 batch qua mạng, tính loss so với nhãn thật
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass: xóa gradient cũ, tính gradient mới, cập nhật trọng số
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cộng dồn loss (nhân số mẫu trong batch để lấy trung bình đúng ở cuối epoch)
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        # model.eval(): tắt Dropout để đánh giá; no_grad(): không cần tính gradient nên nhanh hơn
        model.eval()
        with torch.no_grad():
            train_pred = torch.argmax(model(X_train_tensor), dim=1)
            train_acc = (train_pred == y_train_tensor).sum().item() / len(y_train_tensor)
            test_pred = torch.argmax(model(X_test_tensor), dim=1)
            test_acc = (test_pred == y_test_tensor).sum().item() / len(y_test_tensor)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, '
                  f'Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%')

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"Đã lưu trọng số mô hình vào '{WEIGHTS_FILE}'")

    # Vẽ đồ thị: bên trái là loss, bên phải là độ chính xác train/test theo epoch
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), losses, linewidth=2, color='blue')
    plt.title('Training Loss theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in train_accs], linewidth=2, color='blue', label='Train')
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in test_accs], linewidth=2, color='orange', label='Test')
    plt.title('Độ chính xác theo Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print("Đã lưu đồ thị huấn luyện vào 'training_curves.png'")

# ==========================================
# 9. ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    # Lớp được dự đoán là lớp có điểm (output) cao nhất trong 8 nơ-ron đầu ra
    predicted = torch.argmax(outputs, dim=1)
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Ma trận nhầm lẫn (tự tính, không dùng sklearn):
# cm[i][j] = số ảnh có nhãn thật là lớp i nhưng mạng đoán là lớp j.
# Đường chéo (i == j) là số ảnh đoán đúng.
cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
for true_label, pred_label in zip(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()):
    cm[true_label][pred_label] += 1

# Độ chính xác từng lớp = số ảnh đoán đúng / tổng số ảnh của lớp đó
print("Độ chính xác từng lớp:")
for i, class_name in enumerate(CLASSES):
    print(f"  {class_name:10s}: {cm[i, i] / cm[i].sum() * 100:.2f}%  ({cm[i, i]}/{cm[i].sum()})")

# Vẽ ma trận nhầm lẫn: ô càng đậm thì càng nhiều ảnh, ghi số lượng vào từng ô
plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        # Chữ trắng trên ô đậm, chữ đen trên ô nhạt cho dễ đọc
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=color)
plt.xticks(range(len(CLASSES)), TEN_LOP, rotation=45)
plt.yticks(range(len(CLASSES)), TEN_LOP)
plt.xlabel('Lớp dự đoán')
plt.ylabel('Lớp thật')
plt.title(f'Ma trận nhầm lẫn trên tập test (Accuracy = {accuracy * 100:.2f}%)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Đã lưu ma trận nhầm lẫn vào 'confusion_matrix.png'")

# ==========================================
# 10. DỰ ĐOÁN ẢNH MỚI
# ==========================================
def predict_image(path):
    # Ảnh mới phải đi qua đúng các bước như lúc train: trích đặc trưng -> chuẩn hóa bằng scaler của train
    img = Image.open(path).convert('RGB')
    features = extract_features(img)
    # scaler.transform nhận bảng 2 chiều, nên bọc vector thành 1 hàng: [features]
    features_scaled = scaler.transform([features])
    x = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        idx = torch.argmax(model(x), dim=1).item()
    return CLASSES[idx]

# Thử dự đoán 16 ảnh test ngẫu nhiên và vẽ kết quả: đoán đúng chữ xanh, đoán sai chữ đỏ
sample_idx = np.random.choice(len(test_paths), 16, replace=False)
plt.figure(figsize=(14, 14))
for k, idx in enumerate(sample_idx):
    path = test_paths[idx]
    true_name = CLASSES[y_test[idx]]
    pred_name = predict_image(path)
    plt.subplot(4, 4, k + 1)
    plt.imshow(Image.open(path).convert('RGB'))
    color = 'green' if pred_name == true_name else 'red'
    plt.title(f"Thật: {TEN_LOP[CLASSES.index(true_name)]}\nĐoán: {TEN_LOP[CLASSES.index(pred_name)]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig('du_doan_mau.png')
plt.close()
print("Đã lưu kết quả dự đoán mẫu vào 'du_doan_mau.png'")
