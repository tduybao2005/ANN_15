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
DATA_DIR = 'dataset'
CLASSES = ['xe_dap', 'xe_may', 'o_to', 'xe_buyt', 'xe_tai', 'may_bay', 'tau_thuy', 'tau_hoa']
TEN_LOP = ['Xe đạp', 'Xe máy', 'Ô tô', 'Xe buýt', 'Xe tải', 'Máy bay', 'Tàu thủy', 'Tàu hỏa']
IMG_SIZE = 64
TRAIN_RATIO = 0.8
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHTS_FILE = 'weight_ann_phuong_tien.pth'
USE_SAVED_WEIGHTS = True

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# ==========================================
# 2. ĐỌC DANH SÁCH ẢNH & CHIA TRAIN/TEST THEO TỪNG LỚP
# ==========================================
rows = []
for label, class_name in enumerate(CLASSES):
    for path in sorted(glob.glob(f'{DATA_DIR}/{class_name}/*.jpg')):
        rows.append({'path': path, 'label': label})
df = pd.DataFrame(rows)
print(f"Tổng số ảnh: {len(df)}")

train_list = []
test_list = []
for label, group in df.groupby("label"):
    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(TRAIN_RATIO * len(group))
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)
train_paths = train_df['path'].tolist()
test_paths = test_df['path'].tolist()
print(f"Số ảnh train: {len(train_paths)}, số ảnh test: {len(test_paths)}")

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

def color_histogram(hsv_part):
    so_pixel = hsv_part.shape[0] * hsv_part.shape[1]
    hist_list = []
    for c in range(3):
        hist, _ = np.histogram(hsv_part[:, :, c], bins=16, range=(0, 256))
        hist_list.append(hist / so_pixel)
    return np.concatenate(hist_list)

def extract_features(img):
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img)

    hog_64 = hog(rgb, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)

    rgb_32 = np.array(img.resize((32, 32)))
    hog_32 = hog(rgb_32, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)

    hsv = np.array(img.convert('HSV'))
    color_features = []
    for top, bottom in [(0, 21), (21, 42), (42, 64)]:
        color_features.append(color_histogram(hsv[top:bottom]))

    gray = np.array(img.convert('L'))
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_features = []
    for vung in [lbp, lbp[:32, :32], lbp[:32, 32:], lbp[32:, :32], lbp[32:, 32:]]:
        hist, _ = np.histogram(vung, bins=10, range=(0, 10))
        lbp_features.append(hist / vung.size)

    tiny = np.array(img.resize((8, 8))).reshape(-1) / 255.0

    return np.concatenate([hog_64, hog_32] + color_features + lbp_features + [tiny]).astype(np.float32)

plt.figure(figsize=(16, 5))
for label, class_name in enumerate(CLASSES):
    sample_path = train_df[train_df['label'] == label]['path'].iloc[0]
    sample_img = Image.open(sample_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
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
    X_train.append(extract_features(img))
    y_train.append(label)
    X_train.append(extract_features(img.transpose(Image.FLIP_LEFT_RIGHT)))
    y_train.append(label)
    if (i + 1) % 500 == 0:
        print(f"  Đã xử lý {i + 1}/{len(train_df)} ảnh train")

print("Đang trích đặc trưng ảnh test...")
X_test = []
y_test = []
for i, (path, label) in enumerate(zip(test_df['path'], test_df['label'])):
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
class ManualZScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        data_np = np.array(data)
        self.mean_ = np.mean(data_np, axis=0)
        self.std_ = np.std(data_np, axis=0)

        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

    def transform(self, data):
        data_np = np.array(data)
        return (data_np - self.mean_) / self.std_

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data_np = np.array(data)
        return data_np * self.std_ + self.mean_

scaler = ManualZScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. CHUYỂN SANG TENSOR & DATALOADER
# ==========================================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
print(f"Số batch mỗi epoch: {len(train_loader)}")

# ==========================================
# 7. XÂY DỰNG MẠNG NƠ-RON ANN (MLP)
# ==========================================
input_nodes = X_train_tensor.shape[1]

model = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features=input_nodes, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=512, out_features=128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=128, out_features=len(CLASSES))
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số của mạng: {total_params:,}")

criterion = nn.CrossEntropyLoss()
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
    losses = []
    train_accs = []
    test_accs = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

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
    predicted = torch.argmax(outputs, dim=1)
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
for true_label, pred_label in zip(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()):
    cm[true_label][pred_label] += 1

print("Độ chính xác từng lớp:")
for i, class_name in enumerate(CLASSES):
    print(f"  {class_name:10s}: {cm[i, i] / cm[i].sum() * 100:.2f}%  ({cm[i, i]}/{cm[i].sum()})")

plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
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
    img = Image.open(path).convert('RGB')
    features = extract_features(img)
    features_scaled = scaler.transform([features])
    x = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        idx = torch.argmax(model(x), dim=1).item()
    return CLASSES[idx]

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
