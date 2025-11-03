# Checklist Tiến độ Đồ án

Đây là danh sách các công việc cần hoàn thành cho đồ án, được sắp xếp theo 4
giai đoạn chính.

Đánh dấu `[x]` vào các mục đã hoàn thành.

## Giai đoạn 1: Phân tích & Tiền xử lý Dữ liệu

Nền tảng của toàn bộ dự án.

- [x] **1.1. Phân tích Biến mục tiêu (`SalePrice`)**
    - [x] Vẽ biểu đồ phân phối `SalePrice` (phát hiện lệch phải).
    - [x] Áp dụng phép biến đổi logarit (`np.log1p`) và sử dụng `SalePrice_Log` làm mục tiêu.
    - [x] Vẽ lại biểu đồ để xác nhận phân phối đã chuẩn hóa.

- [ ] **1.2. Xử lý Giá trị bị thiếu (Missing Values)**
    - [ ] Đọc và hiểu `data/data_description.txt`.
    - [ ] Điền các giá trị `NaN` (Loại 1: có ý nghĩa, ví dụ `PoolQC` -> "None").
    - [ ] Điền các giá trị `NaN` (Loại 2: thiếu thật, ví dụ `LotFrontage` -> median).

- [ ] **1.3. Mã hóa Biến phân loại (Categorical Encoding)**
    - [ ] Mã hóa Ordinal (có thứ tự) cho các biến như `ExterQual`.
    - [ ] Mã hóa One-Hot (không thứ tự) cho các biến như `Neighborhood`.

- [ ] **1.4. Kỹ thuật Đặc trưng (Feature Engineering)**
    - [ ] Tạo đặc trưng mới (ví dụ: `TotalSF`, `HouseAge`).

- [ ] **1.5. Co giãn Đặc trưng (Feature Scaling)**
    - [ ] Áp dụng `StandardScaler` cho tất cả các đặc trưng số.

- [ ] **1.6. Phân chia Dữ liệu**
    - [ ] Tách `train.csv` thành 80% tập huấn luyện (train) và 20% tập kiểm tra (validation).

## Giai đoạn 2: Xây dựng & Đánh giá Mô hình

Tìm ra mô hình dự đoán tốt nhất.

- [ ] **2.1. Xây dựng `Pipeline`**
    - [ ] Xây dựng `ColumnTransformer` để đóng gói các bước 1.2, 1.3, 1.5.
    - [ ] Xây dựng `Pipeline` hoàn chỉnh kết hợp `ColumnTransformer` và mô hình.

- [ ] **2.2. Huấn luyện & So sánh 6 Mô hình**
    - [ ] (Linear, Ridge, Lasso, SVM, Decision Tree, Random Forest, XGBoost)
    - [ ] Chạy vòng lặp, huấn luyện và dự đoán trên tập validation.
    - [ ] Chuyển đổi ngược (`np.expm1`) kết quả dự đoán trước khi đánh giá.

- [ ] **2.3. Đánh giá Mô hình**
    - [ ] Tính toán **RMSE** và **R²** cho 6 mô hình.
    - [ ] Lập bảng so sánh kết quả.
    - [ ] Xác định 1-2 mô hình tốt nhất.

- [ ] **2.4. Tinh chỉnh & Phân tích**
    - [ ] Tinh chỉnh siêu tham số (ví dụ: `GridSearchCV`) cho mô hình tốt nhất.
    - [ ] Trích xuất Tầm quan trọng Đặc trưng (Feature Importance).

- [ ] **2.5. Lưu Mô hình**
    - [ ] Lưu `pipeline` tốt nhất vào `models/house_price_model.joblib`.

## Giai đoạn 3: Viết Báo cáo & Phân tích Bổ sung

Hoàn thiện sản phẩm nộp cho giảng viên.

- [ ] **3.1. Viết Báo cáo (LaTeX)**
    - [ ] Viết phần Giới thiệu và Phương pháp luận.
    - [ ] Viết phần Kết quả & Thảo luận.
    - [ ] Viết phần Kết luận.

- [ ] **3.2. Chèn Biểu đồ/Bảng**
    - [ ] Chèn Bảng so sánh RMSE/R² (từ 2.3).
    - [ ] Chèn Biểu đồ Feature Importance (từ 2.4).

- [ ] **3.3. Phân tích Chuỗi thời gian (Bonus)**
    - [ ] Gộp (aggregate) `Median_SalePrice` theo `YrSold` và `MoSold`.
    - [ ] Vẽ biểu đồ đường và phân tích xu hướng thị trường.
    - [ ] Chèn biểu đồ này vào báo cáo.

## Giai đoạn 4: Triển khai Web App (Điểm cộng)

Tạo một sản phẩm demo tương tác.

- [ ] **4.1. Backend (Flask)**
    - [ ] Xây dựng `app/app.py`
    - [ ] Tải file `models/house_price_model.joblib`.
    - [ ] Tạo endpoint `/predict` để nhận dữ liệu và trả về dự đoán.

- [ ] **4.2. Frontend (HTML)**
    - [ ] Xây dựng `app/templates/index.html` với 3 tab.
    - [ ] Tạo file `app/static/css/style.css` cơ bản.

- [ ] **4.3. Hoàn thiện Tab**
    - [ ] Tab 1 (Dự đoán): Hoàn thiện form nhập liệu (lấy giá trị dropdown từ tập train).
    - [ ] Tab 2 (Phân tích): Nhúng ảnh biểu đồ Feature Importance.
    - [ ] Tab 3 (Thị trường): Nhúng ảnh biểu đồ Chuỗi thời gian.

- [ ] **4.4. Kiểm thử**
    - [ ] Chạy `flask run` và kiểm tra toàn bộ chức năng.
