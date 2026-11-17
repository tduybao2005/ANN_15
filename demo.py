import os
import sys
import warnings
warnings.filterwarnings('ignore')
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use(os.environ.get('DEMO_BACKEND', 'TkAgg'))
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog, local_binary_pattern

CLASSES = ['xe_dap', 'xe_may', 'o_to', 'xe_buyt', 'xe_tai', 'may_bay', 'tau_thuy', 'tau_hoa']
TEN_LOP = ['Xe đạp', 'Xe máy', 'Ô tô', 'Xe buýt', 'Xe tải', 'Máy bay', 'Tàu thủy', 'Tàu hỏa']
IMG_SIZE = 64
WEIGHTS_FILE = 'weight_ann_phuong_tien.pth'
SCALER_FILE = 'scaler_stats.npz'
IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else 'demo_images'


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
    color_features = [color_histogram(hsv[t:b]) for t, b in [(0, 21), (21, 42), (42, 64)]]
    gray = np.array(img.convert('L'))
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_features = []
    for vung in [lbp, lbp[:32, :32], lbp[:32, 32:], lbp[32:, :32], lbp[32:, 32:]]:
        hist, _ = np.histogram(vung, bins=10, range=(0, 10))
        lbp_features.append(hist / vung.size)
    tiny = np.array(img.resize((8, 8))).reshape(-1) / 255.0
    return np.concatenate([hog_64, hog_32] + color_features + lbp_features + [tiny]).astype(np.float32)


device = torch.device('cpu')
stats = np.load(SCALER_FILE)
mean, std = stats['mean'], stats['std']
model = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(len(mean), 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, len(CLASSES)),
).to(device)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model.eval()


def recognize(img):
    t0 = time.perf_counter()
    x = torch.tensor(((extract_features(img) - mean) / std)[None, :], dtype=torch.float32).to(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    t2 = time.perf_counter()
    return probs, (t1 - t0) * 1000, (t2 - t1) * 1000


paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
if not paths:
    sys.exit(f"Không có ảnh .jpg trong '{IMG_DIR}'")
recognize(Image.open(paths[0]).convert('RGB'))   # chạy khởi động 1 lần, không tính giờ

# Bỏ phím tắt mặc định của matplotlib trùng với phím mũi tên
plt.rcParams['keymap.back'] = []
plt.rcParams['keymap.forward'] = []

fig = plt.figure(figsize=(12.8, 6.8))
state = {'i': 0}
HINT = '←  →  hoặc Space: đổi ảnh      q: thoát'


def draw_input(n, img):
    # Bước 1: chỉ hiện ảnh mới (đầu vào của model), chưa có kết quả
    fig.clf()
    fig.suptitle('Demo phân loại phương tiện bằng ANN (HOG + màu + LBP → MLP)', fontsize=16, y=0.97)
    ax = fig.add_axes([0.04, 0.12, 0.45, 0.70])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Ảnh {n + 1}/{len(paths)}', fontsize=12)
    fig.text(0.62, 0.50, 'Đang nhận dạng...', fontsize=18, color='gray')
    fig.text(0.04, 0.04, HINT, fontsize=11, color='gray')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.4)


def show(n):
    # Mỗi lần đổi ảnh: hiện ảnh -> đưa qua model -> đo thời gian -> hiện kết quả
    path = paths[n]
    true_name = os.path.splitext(os.path.basename(path))[0].split('_', 1)[1]
    img = Image.open(path).convert('RGB')
    draw_input(n, img)

    probs, t_feat, t_net = recognize(img)
    pred = int(probs.argmax())
    true = CLASSES.index(true_name)
    ok = pred == true
    col = '#1a9850' if ok else '#d73027'

    fig.clf()
    fig.suptitle('Demo phân loại phương tiện bằng ANN (HOG + màu + LBP → MLP)', fontsize=16, y=0.97)
    ax = fig.add_axes([0.04, 0.12, 0.45, 0.70])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Ảnh {n + 1}/{len(paths)}', fontsize=12)
    bx = fig.add_axes([0.62, 0.20, 0.34, 0.50])
    bars = bx.barh(TEN_LOP[::-1], (probs * 100)[::-1], color='lightgray')
    bars[len(CLASSES) - 1 - pred].set_color(col)
    bx.set_xlim(0, 100)
    bx.set_xlabel('Xác suất (%)')
    bx.set_title('Xác suất từng lớp', fontsize=12)
    for b, p in zip(bars, (probs * 100)[::-1]):
        bx.text(min(p + 1, 85), b.get_y() + b.get_height() / 2, f'{p:.1f}%', va='center', fontsize=9)
    fig.text(0.62, 0.86, f'Dự đoán: {TEN_LOP[pred]}', fontsize=20, color=col, weight='bold')
    fig.text(0.62, 0.81, f'Nhãn thật: {TEN_LOP[true]}   →   {"ĐÚNG" if ok else "SAI"}', fontsize=13, color=col)
    fig.text(0.62, 0.085, f'Thời gian nhận dạng: {t_feat + t_net:.1f} ms', fontsize=14, weight='bold')
    fig.text(0.62, 0.045, f'(trích đặc trưng {t_feat:.1f} ms + chạy mạng {t_net:.1f} ms)', fontsize=9, color='gray')
    fig.text(0.04, 0.04, HINT, fontsize=11, color='gray')
    fig.canvas.draw_idle()


def on_key(event):
    if event.key in ('right', ' ', 'enter', 'n'):
        state['i'] = (state['i'] + 1) % len(paths)
    elif event.key in ('left', 'backspace', 'b'):
        state['i'] = (state['i'] - 1) % len(paths)
    else:
        return
    show(state['i'])


fig.canvas.mpl_connect('key_press_event', on_key)
show(0)
plt.show()
