# ==========================================
# TẢI ẢNH PHƯƠNG TIỆN TỪ KAGGLE (bổ sung cho ảnh crawl từ Bing)
# - Cần file token Kaggle: ~/.kaggle/access_token
#   (kaggle.com -> Settings -> API -> Create New Token)
# - Kết quả: data_kaggle/<lớp>/*.jpg (ảnh RGB, cạnh dài <= 256px)
# - Sau đó chạy tai_du_lieu.py: bước làm sạch sẽ gộp data_kaggle/ với ảnh Bing
# ==========================================
import os
import io
import glob
import zipfile
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# 1. CẤU HÌNH THAM SỐ
# ==========================================
ZIP_DIR = 'data_kaggle_zip'   # Thư mục chứa file zip tải về (xóa được sau khi chạy xong)
OUT_DIR = 'data_kaggle'       # Thư mục ảnh đã giải nén + thu nhỏ, chia theo lớp
MAX_SIDE = 256                # Thu nhỏ ảnh để tiết kiệm ổ đĩa (giống tai_du_lieu.py)

# Mỗi dòng: (tên dataset trên Kaggle, thư mục ảnh bên trong file zip, lớp của mình)
# Tên lớp phải trùng với 8 lớp trong phan_loai_phuong_tien.py
SOURCES = [
    # Đã xem ảnh mẫu từng thư mục trước khi chọn. Các thư mục bị loại vì lẫn lớp khác:
    #   mohamedmaher5 Vehicles/Cars (lẫn nhiều xe tải), marquis03 minibus (toàn xe van),
    #   kaggleashwin Dataset/Truck (nhiều xe bán tải, ảnh game), marquis03 fire engine (toàn màu đỏ)
    ('mohamedmaher5/vehicle-classification', 'Vehicles/Bikes', 'xe_dap'),
    ('mohamedmaher5/vehicle-classification', 'Vehicles/Motorcycles', 'xe_may'),
    ('maciejgronczynski/vehicle-classification-dataset', 'dataset/cars', 'o_to'),
    ('marquis03/vehicle-classification', 'train/family sedan', 'o_to'),
    ('marquis03/vehicle-classification', 'val/family sedan', 'o_to'),
    ('marquis03/vehicle-classification', 'train/SUV', 'o_to'),
    ('marquis03/vehicle-classification', 'val/SUV', 'o_to'),
    ('marquis03/vehicle-classification', 'train/taxi', 'o_to'),
    ('marquis03/vehicle-classification', 'val/taxi', 'o_to'),
    ('marquis03/vehicle-classification', 'train/bus', 'xe_buyt'),
    ('marquis03/vehicle-classification', 'val/bus', 'xe_buyt'),
    ('kaggleashwin/vehicle-type-recognition', 'Dataset/Bus', 'xe_buyt'),
    ('marquis03/vehicle-classification', 'train/truck', 'xe_tai'),
    ('marquis03/vehicle-classification', 'val/truck', 'xe_tai'),
    ('marquis03/vehicle-classification', 'train/heavy truck', 'xe_tai'),
    ('marquis03/vehicle-classification', 'val/heavy truck', 'xe_tai'),
    ('mohamedmaher5/vehicle-classification', 'Vehicles/Planes', 'may_bay'),
    ('maciejgronczynski/vehicle-classification-dataset', 'dataset/planes', 'may_bay'),
    ('mohamedmaher5/vehicle-classification', 'Vehicles/Ships', 'tau_thuy'),
    ('mohamedmaher5/vehicle-classification', 'Vehicles/Trains', 'tau_hoa'),
    ('maciejgronczynski/vehicle-classification-dataset', 'dataset/trains', 'tau_hoa'),
]

# ==========================================
# 2. TẢI FILE ZIP CỦA TỪNG DATASET
# ==========================================
api = KaggleApi()
api.authenticate()   # Đọc token trong ~/.kaggle/

# Lấy danh sách dataset không trùng lặp (một dataset có thể cho nhiều lớp)
datasets = []
for dataset, folder, class_name in SOURCES:
    if dataset not in datasets:
        datasets.append(dataset)

for dataset in datasets:
    owner = dataset.split('/')[0]
    # Mỗi dataset tải vào 1 thư mục riêng theo tên tác giả,
    # vì 2 dataset khác nhau có thể trùng tên file zip (vd: vehicle-classification.zip)
    save_dir = os.path.join(ZIP_DIR, owner)
    if glob.glob(os.path.join(save_dir, '*.zip')):
        print(f"Đã có file zip của {dataset}, bỏ qua bước tải")
        continue
    print(f"Đang tải {dataset} ...")
    api.dataset_download_files(dataset, path=save_dir, quiet=False)

# ==========================================
# 3. GIẢI NÉN ẢNH CẦN DÙNG, ĐỔI SANG RGB & THU NHỎ
# ==========================================
count = {}   # Đếm số ảnh đã lưu cho mỗi lớp
for dataset, folder, class_name in SOURCES:
    owner = dataset.split('/')[0]
    zip_path = glob.glob(os.path.join(ZIP_DIR, owner, '*.zip'))[0]
    os.makedirs(os.path.join(OUT_DIR, class_name), exist_ok=True)
    count.setdefault(class_name, 0)

    with zipfile.ZipFile(zip_path) as zf:
        # Chỉ lấy các file nằm trong đúng thư mục của lớp này
        names = [n for n in zf.namelist() if n.startswith(folder + '/') and not n.endswith('/')]
        saved = 0
        for name in names:
            try:
                # Đọc ảnh từ zip vào bộ nhớ, đổi sang RGB (bỏ kênh alpha / ảnh xám / CMYK)
                img = Image.open(io.BytesIO(zf.read(name))).convert('RGB')
            except Exception:
                continue   # File hỏng hoặc không phải ảnh -> bỏ qua
            img.thumbnail((MAX_SIDE, MAX_SIDE))   # Giữ tỉ lệ, cạnh dài tối đa 256px
            out_name = f"{owner}_{count[class_name]:05d}.jpg"
            img.save(os.path.join(OUT_DIR, class_name, out_name), quality=90)
            count[class_name] += 1
            saved += 1
    print(f"{dataset} | {folder} -> {class_name}: {saved} ảnh")

# ==========================================
# 4. IN BẢNG TỔNG KẾT
# ==========================================
print("\nSố ảnh Kaggle mỗi lớp:")
for class_name, n in count.items():
    print(f"  {class_name}: {n}")
