# Phân loại phương tiện giao thông bằng ANN: Kế hoạch thực hiện

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Context

Bạn có bài tập: "thiết kế và training model học máy cho bài toán phân loại phương tiện giao thông dùng mạng ANN".
Bạn muốn: (1) mình tự tìm thật nhiều ảnh phương tiện, (2) viết **một file .py** gồm xử lý dữ liệu + dựng mạng nơ-ron bằng PyTorch, code **đơn giản, comment từng bước**, giống style repo `tduybao2005/machine_learning`.

Các quyết định bạn đã chốt:
- **8 lớp:** xe đạp, xe máy, ô tô con, xe buýt, xe tải, máy bay, tàu thủy, tàu hỏa
- **Nguồn ảnh:** crawl Bing (lọc kiểu "photo"), nhiều từ khóa Anh + Việt
- **Mô hình:** MLP trên đặc trưng **HOG + histogram màu**, không dùng CNN

**Goal:** Có thư mục `dataset/` với khoảng 1000–1500 ảnh sạch cho mỗi lớp, và file `phan_loai_phuong_tien.py` chạy từ đầu đến cuối, in accuracy và lưu biểu đồ cùng trọng số.

**Architecture:** Có 2 script top-level, giống style repo (không chia module).
- `tai_du_lieu.py`: crawl Bing rồi làm sạch, ghi ảnh vào `dataset/<lớp>/`.
- `phan_loai_phuong_tien.py` (file nộp bài): đọc ảnh, chia train/test theo từng lớp, trích đặc trưng HOG + màu, chuẩn hóa Z-score, cho qua `nn.Sequential` MLP, train bằng `CrossEntropyLoss` + Adam, rồi đánh giá và vẽ biểu đồ.

**Tech Stack:** Python 3.12 (pyenv), torch 2.10 CPU, numpy, Pillow, matplotlib, **scikit-image** (hàm `hog`) và **icrawler** (crawl Bing). Hai thư viện cuối cần cài thêm.

**Spec:** Không có file spec riêng. Spec gồm yêu cầu của bạn trong hội thoại và 3 lựa chọn ở trên.

## Style tham chiếu từ repo `machine_learning` (bắt buộc bám theo)

- `BTVN_HocMay_HK2_2025_2026 /BT1/btap1.py`:
  - Chia code thành các khối `# ==========================================` + `# N. TÊN BƯỚC`.
  - Khối "CẤU HÌNH THAM SỐ" viết HOA ở đầu file.
  - Có `device`, cờ `USE_SAVED_WEIGHTS` + `WEIGHTS_FILE`.
  - Có class `ManualZScaler` (fit/transform/inverse_transform).
  - Vẽ loss bằng matplotlib + `savefig`.
- `example_code/trích đặc trưng/trichdactrung.py`:
  - Chia train/test **theo từng lớp** bằng `groupby` + `sample(frac=1, random_state=42)` + `split_idx = int(0.7*len)`.
  - Có `model.train()`/`model.eval()` và `torch.no_grad()`.
  - Tính accuracy bằng `(predicted == y).sum().item() / len(y)`.
- `example_code/classification/classification.ipynb`: `nn.Sequential(Linear, Tanh, Linear)`, `CrossEntropyLoss`, `optim.Adam`, `torch.argmax(outputs, dim=1)`.

## Global Constraints

- Code **đơn giản như repo**:
  - Script chạy tuần tự từ trên xuống, chỉ dùng hàm khi cần tái sử dụng (trích đặc trưng, hash ảnh).
  - Không viết class phức tạp. Ngoại lệ duy nhất là `ManualZScaler`, chép theo đúng mẫu `btap1.py`.
- **Comment tiếng Việt** ở mọi bước. Mỗi khối có tiêu đề `# N. ...`, và mỗi dòng không hiển nhiên có comment giải thích "làm gì / tại sao".
- PyTorch cho phần mạng nơ-ron. Không dùng sklearn trong file nộp bài, vì repo cũ tự code để thay sklearn.
- Tên thư mục lớp (cố định, cũng là thứ tự nhãn 0..7): `['xe_dap', 'xe_may', 'o_to', 'xe_buyt', 'xe_tai', 'may_bay', 'tau_thuy', 'tau_hoa']`.
- Seed cố định: `np.random.seed(42)`, `torch.manual_seed(42)`, `random_state=42`.
- Máy chỉ có CPU, 12 core, RAM 23GB, ổ đĩa **chỉ còn ~12GB trống**:
  - Ảnh sạch lưu tối đa 256px.
  - Thư mục ảnh thô `data_raw/` bị xóa sau khi làm sạch xong.
- Thư mục làm việc `/home/ncd/learnspaces/ANN_15` **không phải git repo**, nên bỏ các bước commit (trừ khi bạn muốn `git init`).
- **Kiểm thử:** file nộp bài là script dạy học top-level theo style bạn yêu cầu, nên không viết pytest. Mỗi task kiểm bằng lệnh chạy thật cùng kết quả mong đợi (xem từng task).

## Review Focus

1. **Ảnh trùng giữa train và test:** nhiều từ khóa trả về cùng một ảnh, làm accuracy ảo.
   - Cách xử lý: khử trùng bằng average-hash trên **toàn bộ** dataset, xuyên các lớp.
   - Task 1 kiểm: số hash trùng = 0.
2. **Ảnh PNG có alpha / ảnh xám / CMYK / file hỏng làm script crash khi trích đặc trưng.**
   - Cách xử lý: khi làm sạch, mọi ảnh được `convert('RGB')` và lưu lại thành JPEG. File chính cũng `convert('RGB')` lại.
   - Task 2 kiểm: trích đặc trưng toàn bộ dataset không lỗi.
3. **Cột đặc trưng có std = 0** (ô HOG luôn bằng 0): chia cho `1e-8` thì test bị nổ giá trị.
   - Cách xử lý: `ManualZScaler` đặt `std = 1.0` khi std == 0.
   - Task 2 kiểm: `np.isfinite(X_test).all()` và `abs(X_test).max() < 1e3`.
4. **Ảnh nhiễu** (clipart, logo, ảnh sai lớp) làm mạng học sai.
   - Cách xử lý: tạo contact sheet 48 ảnh ngẫu nhiên mỗi lớp rồi xem bằng mắt. Từ khóa nào cho nhiều rác thì bỏ hoặc thay, rồi crawl lại (Task 1).
5. **Mất cân bằng lớp** (Bing trả ít ảnh cho một lớp) làm mạng thiên lệch.
   - Task 1 kiểm: mỗi lớp ≥ 800 ảnh và max/min ≤ 1.5.
   - Task 3 in accuracy **theo từng lớp**.

---

### Task 1: Tìm & tải ảnh, làm sạch dataset (`tai_du_lieu.py`)

**Files:**
- Create: `/home/ncd/learnspaces/ANN_15/tai_du_lieu.py`
- Output: `dataset/<lớp>/<lớp>_00000.jpg ...`. Thư mục tạm `data_raw/<lớp>/<idx_từ_khóa>/` bị xóa ở cuối task.

**Interfaces:**
- Produces: `dataset/` gồm 8 thư mục con đúng tên lớp. Ảnh là JPEG RGB, cạnh dài ≤ 256px, cạnh ngắn ≥ 80px, không trùng hash.

- [ ] **Step 1: Cài thư viện**

  Run: `pip install icrawler scikit-image`
  Expected: `python3 -c "import icrawler, skimage; print('ok')"` in ra `ok`.

- [ ] **Step 2: Viết `tai_du_lieu.py` (top-level, comment tiếng Việt)**

  Cấu hình đầu file: `RAW_DIR='data_raw'`, `OUT_DIR='dataset'`, `MAX_PER_KEYWORD=300`, `MAX_PER_CLASS=1500`, `MIN_SIDE=80`, `MAX_SIDE=256`.

  Dict `KEYWORDS` có 6 từ khóa mỗi lớp, trộn Anh + Việt:
  - `xe_dap`: "bicycle photo", "xe đạp", "road bike", "city bicycle street", "mountain bike", "xe đạp thể thao"
  - `xe_may`: "motorbike vietnam street", "xe máy", "scooter honda", "motorcycle photo", "xe máy honda wave", "xe tay ga"
  - `o_to`: "car photo street", "xe ô tô con", "sedan car", "toyota vios", "hatchback car", "suv car road"
  - `xe_buyt`: "city bus", "xe buýt", "xe buýt hà nội", "coach bus", "school bus", "double decker bus"
  - `xe_tai`: "cargo truck", "xe tải", "dump truck", "lorry truck road", "xe tải hino", "container truck"
  - `may_bay`: "airplane", "máy bay", "passenger jet", "aircraft flying", "vietnam airlines plane", "airplane runway"
  - `tau_thuy`: "cargo ship", "tàu thủy", "boat sea", "cruise ship", "ferry boat", "container ship"
  - `tau_hoa`: "train railway", "tàu hỏa", "passenger train", "locomotive", "tàu hỏa việt nam", "high speed train"

  Các hàm:
  - `crawl_class(class_name: str, keywords: list[str]) -> None`:
    - Mỗi từ khóa dùng một `BingImageCrawler(storage={'root_dir': f'{RAW_DIR}/{class_name}/{i}'}, downloader_threads=8)`.
    - Gọi `.crawl(keyword=kw, filters={'type': 'photo'}, max_num=MAX_PER_KEYWORD)`.
    - Mỗi từ khóa một thư mục con để tên file `000001.jpg` của icrawler không đè nhau.
  - `average_hash(img: Image.Image) -> int`:
    - Resize ảnh về 8x8 grayscale, mỗi pixel > trung bình thì là bit 1.
    - Ghép thành số nguyên 64 bit.
  - `clean_class(class_name: str, seen_hashes: set) -> int`:
    - Duyệt mọi file trong `RAW_DIR/class_name/**`.
    - Mở bằng PIL và `convert('RGB')`. Gặp exception thì bỏ qua.
    - Bỏ ảnh có `min(w,h) < MIN_SIDE` hoặc tỉ lệ cạnh `> 3`.
    - Bỏ ảnh có hash đã nằm trong `seen_hashes`. Đây là set **dùng chung cho cả 8 lớp**.
    - `img.thumbnail((MAX_SIDE, MAX_SIDE))`, rồi lưu `OUT_DIR/class_name/{class_name}_{k:05d}.jpg` với quality 90.
    - Dừng khi đủ `MAX_PER_CLASS`. Trả về số ảnh đã giữ.

  Phần chạy chính:
  - Lặp 8 lớp, gọi `crawl_class`.
  - Sau đó lặp lại 8 lớp, gọi `clean_class`.
  - In bảng `lớp: số ảnh`.

- [ ] **Step 3: Chạy crawl + làm sạch (chạy nền, ước tính 20–40 phút)**

  Run: `cd /home/ncd/learnspaces/ANN_15 && python3 tai_du_lieu.py`
  Expected: bảng cuối in đủ 8 lớp, mỗi lớp ≥ 800 ảnh, max/min ≤ 1.5. Nếu lớp nào < 800 ảnh thì thêm 2–3 từ khóa cho lớp đó, chỉ crawl lại lớp đó, rồi làm sạch lại toàn bộ.

- [ ] **Step 4: Kiểm tra trùng lặp và chất lượng bằng mắt (Review Focus 1, 4)**

  - Dùng script tạm trong scratchpad, không phải file nộp: tính lại `average_hash` cho mọi ảnh trong `dataset/`.
    Expected: `duplicates = 0`.
  - Ghép 48 ảnh ngẫu nhiên mỗi lớp thành `contact_<lớp>.png` (lưới 6x8) trong scratchpad, rồi xem bằng tool Read.
    Expected: hơn 80% ảnh đúng lớp và là ảnh chụp thật. Lớp nào nhiều rác thì truy ra từ khóa gây rác (qua thư mục `data_raw/<lớp>/<i>`), thay từ khóa đó, rồi chạy lại Step 3 cho lớp đó.

- [ ] **Step 5: Dọn ảnh thô để giải phóng ổ đĩa**

  Run: `du -sh data_raw dataset && rm -rf data_raw`
  Expected: `dataset/` khoảng 150–300MB, và `data_raw/` không còn tồn tại.

---

### Task 2: File nộp bài, phần xử lý dữ liệu (`phan_loai_phuong_tien.py` khối 1–6)

**Files:**
- Create: `/home/ncd/learnspaces/ANN_15/phan_loai_phuong_tien.py`

**Interfaces:**
- Consumes: `dataset/<lớp>/*.jpg` từ Task 1.
- Produces (biến top-level Task 3 dùng):
  - `CLASSES: list[str]` (8 phần tử)
  - `X_train_tensor (N_train, 1812) float32`, `y_train_tensor (N_train,) long`
  - `X_test_tensor`, `y_test_tensor`
  - `train_loader: DataLoader`
  - `scaler: ManualZScaler`
  - `extract_features(img: Image.Image) -> np.ndarray` (shape `(1812,)`)
  - `test_paths: list[str]`, `device`

Các khối cần viết, theo đúng thứ tự (tiêu đề `# ====` như `btap1.py`):

- [ ] **Step 1: Khối `# 1. CẤU HÌNH THAM SỐ`**

  ```
  DATA_DIR='dataset'
  CLASSES=[...8 lớp...]
  IMG_SIZE=64
  TRAIN_RATIO=0.8
  EPOCHS=60
  BATCH_SIZE=64
  LEARNING_RATE=0.001
  WEIGHTS_FILE='weight_ann_phuong_tien.pth'
  USE_SAVED_WEIGHTS=False
  ```

  Thêm 2 seed và `device` như trong `btap1.py`.

- [ ] **Step 2: Khối `# 2. ĐỌC DANH SÁCH ẢNH & CHIA TRAIN/TEST THEO TỪNG LỚP`**

  - Tạo DataFrame `path, label`, với label là chỉ số trong `CLASSES`.
  - Chia **theo từng lớp** đúng mẫu `trichdactrung.py`: `groupby("label")`, `sample(frac=1, random_state=42)`, `split_idx=int(TRAIN_RATIO*len(group))`.
  - Chia ở mức **đường dẫn file**, trước khi trích đặc trưng, để ảnh lật chỉ thêm vào train.
  - Vẽ `phan_bo_du_lieu.png`: biểu đồ cột số ảnh train/test mỗi lớp. Bài tập cũ yêu cầu "trực quan hóa dữ liệu".

- [ ] **Step 3: Khối `# 3. HÀM TRÍCH ĐẶC TRƯNG (HOG + HISTOGRAM MÀU)`, hàm `extract_features(img)`**

  1. `img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))`.
  2. HOG trên ảnh xám: `hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))`, được **1764** số.
  3. Histogram màu: `img.convert('HSV')`, mỗi kênh `np.histogram(..., bins=16, range=(0,256))`, chia cho tổng số pixel, được **48** số.
  4. `np.concatenate`, được **1812** số.

  Comment giải thích: HOG mô tả hình dạng cạnh (bánh xe, thân xe, cánh máy bay), còn màu mô tả bối cảnh (trời, nước, đường).

  Vẽ `anh_mau_va_hog.png`: mỗi lớp 1 ảnh gốc cạnh ảnh HOG (`hog(..., visualize=True)`), dùng để minh họa đầu vào mạng trong báo cáo.

- [ ] **Step 4: Khối `# 4. TRÍCH ĐẶC TRƯNG CHO TOÀN BỘ ẢNH`**

  - Lặp danh sách train: mỗi ảnh thêm 2 vector, ảnh gốc và ảnh lật ngang `img.transpose(Image.FLIP_LEFT_RIGHT)`. Đây là tăng cường dữ liệu đơn giản.
  - Lặp danh sách test: chỉ ảnh gốc.
  - In tiến độ mỗi 500 ảnh.
  - In shape `X_train`, `X_test`.

- [ ] **Step 5: Khối `# 5. CHUẨN HÓA Z-SCORE`**

  - Chép class `ManualZScaler` từ `btap1.py`, **sửa** `self.std_ = np.where(self.std_ == 0, 1.0, self.std_)` (Review Focus 3).
  - `fit_transform(X_train)`, `transform(X_test)`.

- [ ] **Step 6: Khối `# 6. CHUYỂN SANG TENSOR & DATALOADER`**

  - `torch.tensor(..., dtype=torch.float32/long)` như `classification.ipynb`.
  - `train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)`.
  - Comment lý do dùng mini-batch thay vì "siêu batch": cập nhật trọng số nhiều lần mỗi epoch, nên tổng quát hóa tốt hơn.

- [ ] **Step 7: Chạy thử phần dữ liệu**

  - Tạm thêm `import sys; sys.exit()` ở cuối file. Ngay sau khi chạy xong thì xóa dòng này đi.
  - Run: `cd /home/ncd/learnspaces/ANN_15 && MPLBACKEND=Agg python3 phan_loai_phuong_tien.py`
  - Expected:
    - Không exception (Review Focus 2).
    - `X_train` shape `(2*N_train, 1812)`, `X_test` shape `(N_test, 1812)`, với `N_test ≈ 0.2 × tổng ảnh`.
    - Thêm một dòng in tạm `np.isfinite(X_test_scaled).all(), np.abs(X_test_scaled).max()` cho ra `True` và `< 1e3` (Review Focus 3).
    - Có 2 file `phan_bo_du_lieu.png`, `anh_mau_va_hog.png`. Mở xem bằng Read, đồ thị phải có nhãn tiếng Việt đọc được.

---

### Task 3: File nộp bài, mạng ANN + huấn luyện + đánh giá (`phan_loai_phuong_tien.py` khối 7–10)

**Files:**
- Modify: `/home/ncd/learnspaces/ANN_15/phan_loai_phuong_tien.py` (nối tiếp sau khối 6)

**Interfaces:**
- Consumes: các biến ở mục Produces của Task 2.
- Produces:
  - `weight_ann_phuong_tien.pth`
  - `training_curves.png`, `confusion_matrix.png`, `du_doan_mau.png`
  - hàm `predict_image(path: str) -> str`, trả về tên lớp

- [ ] **Step 1: Khối `# 7. XÂY DỰNG MẠNG NƠ-RON ANN (MLP)`**

  ```
  nn.Sequential(
      nn.Linear(input_nodes, 512), nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(128, len(CLASSES))
  ).to(device)
  ```

  Trong đó `input_nodes = X_train_tensor.shape[1]` (= 1812), giống cách `btap1.py` tính `input_nodes`. Đổi `IMG_SIZE` thì mạng tự khớp.

  - Comment lý do chọn từng lớp: đầu vào 1812 đặc trưng, hai lớp ẩn giảm dần để nén thông tin, ReLU tránh bão hòa gradient như Tanh, Dropout chống học thuộc, đầu ra 8 nơ-ron ứng với 8 lớp.
  - In tổng số tham số.
  - `criterion = nn.CrossEntropyLoss()`, `optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)`.

- [ ] **Step 2: Khối `# 8. HUẤN LUYỆN (hoặc load weights)`**

  - Cấu trúc `if USE_SAVED_WEIGHTS and os.path.exists(WEIGHTS_FILE)` giống `btap1.py`. File trọng số không tồn tại thì train.
  - Mỗi epoch:
    - `model.train()`, lặp `train_loader` với forward / `zero_grad` / `backward` / `step`, cộng dồn loss.
    - Sau đó `model.eval()` + `no_grad`, tính train acc và test acc.
    - Lưu vào 3 list: `losses`, `train_accs`, `test_accs`.
  - In mỗi 5 epoch: `Epoch [x/60], Loss: ..., Train Acc: ...%, Test Acc: ...%`.
  - Train xong thì `torch.save(model.state_dict(), WEIGHTS_FILE)`.
  - Vẽ `training_curves.png` gồm 2 subplot: loss, và accuracy train/test.

- [ ] **Step 3: Khối `# 9. ĐÁNH GIÁ TRÊN TẬP TEST`**

  - `predicted = torch.argmax(model(X_test_tensor), dim=1)`, tính accuracy theo công thức của repo.
  - Ma trận nhầm lẫn **tự tính bằng vòng lặp numpy** (không dùng sklearn): `cm[true][pred] += 1`.
  - In accuracy từng lớp `cm[i,i]/cm[i].sum()` kèm tên lớp (Review Focus 5).
  - Vẽ `confusion_matrix.png` bằng `plt.imshow` + ghi số vào từng ô, trục là tên lớp.

- [ ] **Step 4: Khối `# 10. DỰ ĐOÁN ẢNH MỚI`**

  - `predict_image(path)`: mở ảnh, `extract_features`, `scaler.transform`, đưa vào model, `argmax`, trả về `CLASSES[idx]`.
  - Vẽ `du_doan_mau.png`: 16 ảnh test ngẫu nhiên, tiêu đề "Thật: … / Đoán: …", đúng thì chữ xanh, sai thì chữ đỏ.

- [ ] **Step 5: Chạy toàn bộ file**

  - Run: `cd /home/ncd/learnspaces/ANN_15 && time MPLBACKEND=Agg python3 phan_loai_phuong_tien.py`
  - Expected:
    - Chạy hết không lỗi trong khoảng dưới 10 phút trên CPU.
    - Loss giảm dần.
    - In `Test Accuracy: xx.xx%` với **mục tiêu ≥ 65%**.
    - Có đủ 5 file PNG + file `.pth`.
    - Mở xem `training_curves.png` và `confusion_matrix.png` bằng Read.

- [ ] **Step 6: Tinh chỉnh nếu accuracy < 65% (tối đa 3 vòng, ghi lại kết quả mỗi vòng)**

  Làm theo thứ tự:
  1. Nhìn ma trận nhầm lẫn. Nếu nhầm tập trung ở một lớp, quay lại Task 1 Step 4 để làm sạch lớp đó.
  2. Nếu test acc thấp hơn hẳn train acc (overfit), tăng Dropout lên 0.5 hoặc `weight_decay` lên 1e-3.
  3. Thử `IMG_SIZE=96` (HOG thành 4356 chiều; `input_nodes` tự cập nhật).

  Nếu 3 vòng vẫn chưa đạt thì dừng và báo kết quả thật cho bạn, không ép số.

- [ ] **Step 7: Kiểm cờ load trọng số**

  - Đặt `USE_SAVED_WEIGHTS=True` rồi chạy lại.
  - Expected: in "Đã load weights thành công", bỏ qua train, Test Accuracy giống hệt lần trước.
  - Sau đó trả cờ về `False`.

---

## Kiểm tra cuối (end-to-end)

1. `ls dataset/*/ | head` và đếm ảnh mỗi lớp: đủ 8 lớp, mỗi lớp ≥ 800 ảnh.
2. `MPLBACKEND=Agg python3 phan_loai_phuong_tien.py` chạy sạch, in Test Accuracy + accuracy từng lớp.
3. Đọc lại file `phan_loai_phuong_tien.py`:
   - Mọi khối có tiêu đề `# N.` và comment tiếng Việt.
   - Không có class nào ngoài `ManualZScaler`.
   - Không dùng sklearn.
   - Đọc từ trên xuống dễ như `btap1.py`.
4. Báo lại cho bạn: số ảnh mỗi lớp, accuracy tổng và từng lớp, danh sách file output, và các từ khóa đã phải thay (nếu có).

## Thực thi

Đề xuất **Native** (mình làm tuần tự trong phiên này, cuối cùng có một reviewer kiểm toàn bộ). Lý do: chỉ 3 task nối tiếp nhau, task sau dùng trực tiếp biến của task trước trong cùng một file, và cần xem ảnh/biểu đồ bằng mắt giữa chừng, nên làm liền mạch sẽ nhanh và ít sai hơn.
