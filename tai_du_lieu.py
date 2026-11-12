# Tải ảnh phương tiện từ Bing, gộp với ảnh Kaggle, làm sạch rồi lưu vào thư mục dataset/<lớp>/.
# Chạy tai_du_lieu_kaggle.py TRƯỚC (để có thư mục data_kaggle/), rồi mới chạy file này.

import os
import glob
import shutil
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

# ==========================================
# 1. CẤU HÌNH THAM SỐ
# ==========================================
RAW_DIR = 'data_raw'        # Thư mục chứa ảnh thô vừa tải về từ Bing (sẽ xóa sau khi làm sạch)
KAGGLE_DIR = 'data_kaggle'  # Ảnh lấy từ các bộ dữ liệu Kaggle (đã gán nhãn sẵn, tạo bởi tai_du_lieu_kaggle.py)
OUT_DIR = 'dataset'         # Thư mục chứa ảnh sạch, dùng cho file huấn luyện
MAX_PER_KEYWORD = 300       # Số ảnh tối đa tải cho mỗi từ khóa
MAX_PER_CLASS = 1500        # Số ảnh sạch tối đa giữ lại cho mỗi lớp (để các lớp cân bằng)
MIN_SIDE = 80               # Ảnh có cạnh ngắn nhỏ hơn 80px thì quá mờ, bỏ đi
MAX_SIDE = 256              # Thu nhỏ ảnh về cạnh dài tối đa 256px để tiết kiệm ổ đĩa

# Tên thư mục của 8 lớp. Thứ tự này cũng là thứ tự nhãn 0..7 trong file huấn luyện
CLASSES = ['xe_dap', 'xe_may', 'o_to', 'xe_buyt', 'xe_tai', 'may_bay', 'tau_thuy', 'tau_hoa']

# Mỗi lớp dùng nhiều từ khóa (trộn tiếng Anh + tiếng Việt) để ảnh đa dạng hơn:
# nhiều góc chụp, nhiều kiểu xe, nhiều bối cảnh (Việt Nam và nước ngoài).
# Bing hiện chỉ trả khoảng 60-80 ảnh khác nhau cho mỗi từ khóa (dù xin tới 300).
# Lớp nào có ít ảnh Kaggle (nhất là ô tô, xe buýt, xe tải) thì dùng thêm nhiều từ khóa Bing để các lớp cân bằng.
# Từ khóa mới luôn được THÊM VÀO CUỐI danh sách, vì thư mục ảnh thô đặt tên theo số thứ tự từ khóa.
#
# Sau khi tải, ảnh thô trong data_raw được xem lại TỪNG ẢNH bằng mắt và xóa tay ảnh rác
# (quảng cáo có chữ, hình vẽ / ảnh 3D, nội thất xe, ảnh cận cảnh linh kiện, xe khác lớp...).
# Từ khóa nào cho hơn 50% ảnh rác thì xóa hết ảnh của từ khóa đó (giữ thư mục rỗng để không bị tải lại).
# Các từ khóa "hỏng" như vậy vẫn để nguyên trong danh sách (để số thứ tự thư mục không bị lệch), gồm:
#   o_to:     "car photo street", "xe ô tô con"
#   xe_buyt:  "city bus", "electric bus" (toàn xe van), "bus side view", "city bus front view", "bus depot"
#   xe_tai:   "cargo truck", "xe tải", "xe tải hino", "delivery truck", "truck side view", "logging truck"
#   may_bay:  "airplane", "airplane runway"
#   tau_thuy: "boat sea", "sailing ship", "yacht sea"
#   tau_hoa:  "tàu hỏa", "passenger train", "locomotive", "commuter train"
# Kinh nghiệm: từ khóa tiếng Anh cụ thể (tên hãng/dòng xe) cho ảnh sạch hơn hẳn từ khóa chung chung hoặc tiếng Việt.
KEYWORDS = {
    'xe_dap':   ["bicycle photo", "xe đạp", "road bike", "city bicycle street", "mountain bike", "xe đạp thể thao"],
    'xe_may':   ["motorbike vietnam street", "xe máy", "scooter honda", "motorcycle photo", "xe máy honda wave", "xe tay ga"],
    'o_to':     ["car photo street", "xe ô tô con", "sedan car", "toyota vios", "hatchback car", "suv car road",
                 "honda civic", "toyota camry", "hyundai accent", "kia morning", "mazda 3 sedan", "honda city car"],
    'xe_buyt':  ["city bus", "xe buýt", "xe buýt hà nội", "coach bus", "school bus", "double decker bus",
                 "public bus street", "xe buýt sài gòn", "electric bus", "articulated bus", "bus side view",
                 "tourist coach", "xe khách", "xe buýt vinbus", "intercity bus", "transit bus",
                 "city bus front view", "xe buýt điện", "bus depot", "xe buýt đà nẵng",
                 "xe buýt tphcm", "mercedes citaro bus", "volvo bus", "scania bus", "london red bus",
                 "greyhound bus", "thaco bus", "hyundai universe bus", "bus brt hà nội"],
    'xe_tai':   ["cargo truck", "xe tải", "dump truck", "lorry truck road", "xe tải hino", "container truck",
                 "semi truck", "box truck", "tanker truck", "xe tải isuzu", "xe ben",
                 "delivery truck", "truck side view", "xe tải thùng", "heavy truck highway", "xe đầu kéo",
                 "garbage truck", "flatbed truck", "xe tải hyundai", "logging truck",
                 "volvo fh truck", "scania truck", "mercedes actros", "man tgx truck", "kenworth truck",
                 "freightliner truck", "isuzu truck", "cement mixer truck"],
    'may_bay':  ["airplane", "máy bay", "passenger jet", "aircraft flying", "vietnam airlines plane", "airplane runway"],
    'tau_thuy': ["cargo ship", "tàu thủy", "boat sea", "cruise ship", "ferry boat", "container ship",
                 "fishing boat", "sailing ship", "speedboat", "yacht sea"],
    'tau_hoa':  ["train railway", "tàu hỏa", "passenger train", "locomotive", "tàu hỏa việt nam", "high speed train",
                 "freight train", "steam locomotive", "commuter train", "shinkansen"],
}

# ==========================================
# 2. HÀM TẢI ẢNH TỪ BING
# ==========================================
def crawl_class(class_name, keywords):
    # Mỗi từ khóa lưu vào một thư mục con riêng (data_raw/<lớp>/<số thứ tự từ khóa>)
    # vì icrawler luôn đặt tên file 000001.jpg, 000002.jpg... nên dùng chung thư mục sẽ bị ghi đè.
    # Tách thư mục còn giúp truy ra từ khóa nào cho nhiều ảnh rác.
    for i, kw in enumerate(keywords):
        # Từ khóa nào đã tải rồi (đã có thư mục) thì bỏ qua.
        # Nhờ vậy khi thêm từ khóa mới rồi chạy lại file, chỉ các từ khóa mới được tải.
        if os.path.exists(f'{RAW_DIR}/{class_name}/{i}'):
            print(f"[{class_name}] Từ khóa {i}: '{kw}' đã tải rồi, bỏ qua")
            continue
        print(f"[{class_name}] Đang tải từ khóa {i}: '{kw}'")
        crawler = BingImageCrawler(storage={'root_dir': f'{RAW_DIR}/{class_name}/{i}'},
                                   downloader_threads=8,   # 8 luồng tải song song cho nhanh
                                   log_level=50)           # Tắt thông báo lỗi tải lẻ tẻ (link hỏng, web chậm), đỡ rối màn hình
        # filters type='photo': chỉ lấy ảnh chụp thật, bỏ bớt clipart / hình vẽ / logo
        crawler.crawl(keyword=kw, filters={'type': 'photo'}, max_num=MAX_PER_KEYWORD)

# ==========================================
# 3. HÀM TÍNH "VÂN TAY" CỦA ẢNH (AVERAGE HASH) ĐỂ KHỬ TRÙNG
# ==========================================
def average_hash(img):
    # Thu ảnh về 8x8 ô xám: chỉ giữ bố cục sáng/tối tổng thể, bỏ chi tiết nhỏ.
    # Nhờ vậy cùng một ảnh nhưng khác kích thước / khác chất lượng nén vẫn cho cùng hash.
    small = np.array(img.convert('L').resize((8, 8)), dtype=np.float32)
    # Ô nào sáng hơn trung bình thì là bit 1, ngược lại là bit 0 -> được 64 bit
    bits = (small > small.mean()).flatten()
    # Ghép 64 bit thành một số nguyên để dễ cho vào set và so sánh
    value = 0
    for b in bits:
        value = value * 2 + int(b)
    return value

# ==========================================
# 4. HÀM LÀM SẠCH ẢNH CỦA MỘT LỚP
# ==========================================
def clean_class(class_name, seen_hashes):
    os.makedirs(f'{OUT_DIR}/{class_name}', exist_ok=True)
    # Gom ảnh của lớp này từ 2 nguồn, theo thứ tự: Kaggle TRƯỚC (ảnh đã gán nhãn sẵn nên sạch hơn),
    # rồi tới ảnh Bing (ở tất cả thư mục con của các từ khóa). Nguồn nào chưa có thì glob trả về rỗng, tự bỏ qua.
    # Sắp xếp tên file để lần nào chạy lại cũng ra cùng kết quả.
    files = []
    for source_dir in [KAGGLE_DIR, RAW_DIR]:
        files += sorted(glob.glob(f'{source_dir}/{class_name}/**/*.*', recursive=True))
    kept = 0
    from_kaggle = 0     # Đếm riêng số ảnh giữ lại lấy từ Kaggle, để in thống kê nguồn ảnh
    for path in files:
        # Mở ảnh bằng PIL và đổi về RGB: xử lý luôn ảnh PNG có kênh alpha, ảnh xám, ảnh CMYK.
        # File hỏng / không phải ảnh sẽ báo lỗi -> bỏ qua file đó.
        try:
            img = Image.open(path)
            img = img.convert('RGB')
        except Exception:
            continue

        # Bỏ ảnh quá nhỏ (mờ, thường là icon) hoặc quá dài/quá bẹt (banner, ảnh ghép)
        w, h = img.size
        if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > 3:
            continue

        # Thu nhỏ về cạnh dài tối đa MAX_SIDE (giữ nguyên tỉ lệ).
        # Thu nhỏ TRƯỚC khi tính hash, để cùng một ảnh ở 2 độ phân giải khác nhau cũng ra cùng hash.
        img.thumbnail((MAX_SIDE, MAX_SIDE))

        # Bỏ ảnh trùng: seen_hashes dùng CHUNG cho cả 8 lớp,
        # nên một ảnh xuất hiện ở nhiều từ khóa / nhiều lớp chỉ được giữ đúng 1 lần.
        # Điều này tránh việc cùng một ảnh nằm cả trong tập train lẫn tập test (accuracy ảo).
        h_value = average_hash(img)
        if h_value in seen_hashes:
            continue
        seen_hashes.add(h_value)

        # Lưu thành JPEG
        img.save(f'{OUT_DIR}/{class_name}/{class_name}_{kept:05d}.jpg', quality=90)
        kept += 1
        if path.startswith(KAGGLE_DIR):
            from_kaggle += 1

        # Đủ số ảnh tối đa thì dừng, để các lớp không chênh lệch quá nhiều
        if kept >= MAX_PER_CLASS:
            break
    print(f"  {class_name}: {kept} ảnh ({from_kaggle} từ Kaggle, {kept - from_kaggle} từ Bing)")
    return kept

# ==========================================
# 5. CHẠY: TẢI ẢNH CHO 8 LỚP
# ==========================================
for class_name in CLASSES:
    crawl_class(class_name, KEYWORDS[class_name])

# ==========================================
# 6. CHẠY: LÀM SẠCH 8 LỚP VÀ IN THỐNG KÊ
# ==========================================
# Xóa dataset cũ (nếu có) để làm sạch lại từ đầu cho cả 8 lớp,
# tránh sót lại ảnh cũ khi chạy lại sau khi đổi từ khóa
shutil.rmtree(OUT_DIR, ignore_errors=True)
seen_hashes = set()     # Tập hash dùng chung cho cả 8 lớp
counts = {}
print("Số ảnh sạch mỗi lớp:")
for class_name in CLASSES:
    counts[class_name] = clean_class(class_name, seen_hashes)

print(f"Tổng cộng: {sum(counts.values())} ảnh")
print(f"Tỉ lệ lớp nhiều nhất / lớp ít nhất: {max(counts.values()) / max(min(counts.values()), 1):.2f}")
