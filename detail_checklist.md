# Checklist Hoàn thành Đồ án (Markdown)

##  Giai đoạn 1: Phân tích & Tiền xử lý Dữ liệu

Bước này là nền tảng. Bạn cần giải thích *tại sao* bạn lại làm sạch dữ liệu
theo cách này.

- [x] **1.1. Phân tích Biến mục tiêu (`SalePrice`)**
    - [x] Vẽ biểu đồ histogram và Q-Q plot của `SalePrice`.
    - [x] Áp dụng phép biến đổi `log1p` (Logarit) và lưu vào `SalePrice_Log`.
    - [x] Vẽ lại biểu đồ histogram/Q-Q plot cho `SalePrice_Log` để so sánh.
    -  **Kiến thức cần nắm:**
        * **Skewness (Độ lệch):** Giải thích "lệch phải" (right-skewed) là gì
          và tại sao nó không tốt cho các mô hình hồi quy tuyến tính (vi phạm
          giả định về phân phối chuẩn của phần dư).
        * **Tại sao dùng Logarit:** Trình bày được `log(SalePrice)` giúp chuẩn
          hóa phân phối, làm giảm tác động của các giá trị ngoại lệ (outliers)
          rất lớn và giúp mô hình hội tụ tốt hơn.

- [ ] **1.2. Xử lý Giá trị bị thiếu (Missing Values)**
    - [ ] Đọc `data_description.txt` để hiểu ý nghĩa của các giá trị `NaN`.
    - [ ] Điền giá trị cho các cột phân loại (Categorical). Ví dụ: `PoolQC`
      `NaN` -> "None" (Không có hồ bơi).
    - [ ] Điền giá trị cho các cột số (Numerical). Ví dụ: `LotFrontage` `NaN`
      -> `median` (Trung vị).
    -  **Kiến thức cần nắm:**
        * **Ý nghĩa của `NaN`:** Phân biệt được 2 loại `NaN`:
            1.  `NaN` có ý nghĩa (ví dụ: `PoolQC` = "Không có hồ bơi").
            2.  `NaN` là dữ liệu bị thiếu thật (ví dụ: `LotFrontage` = "Không
                rõ số đo").
        * **Tại sao dùng Median:** Giải thích tại sao bạn dùng `median` (trung
          vị) thay vì `mean` (trung bình) để điền vào `LotFrontage`. (Gợi ý:
          `Median` ít bị ảnh hưởng bởi các giá trị ngoại lệ).

- [ ] **1.3. Mã hóa Biến phân loại (Categorical Encoding)**
    - [ ] Mã hóa **Ordinal (Thứ tự):** Chuyển các cột như `ExterQual` ('Ex',
      'Gd', 'TA') thành số (ví dụ: 5, 4, 3).
    - [ ] Mã hóa **Nominal (Không thứ tự):** Dùng `One-Hot Encoding` cho các
      cột như `Neighborhood`.
    -  **Kiến thức cần nắm:**
        * **Ordinal vs. Nominal:** Phân biệt rõ ràng hai loại biến này. Tại sao
          `ExterQual` là Ordinal mà `Neighborhood` là Nominal?
        * **Tại sao dùng One-Hot:** Giải thích điều gì sẽ xảy ra nếu bạn mã hóa
          `Neighborhood` thành các số 1, 2, 3... (Gợi ý: Bạn vô tình tạo ra một
          mối quan hệ thứ tự sai lệch, ví dụ: "Khu vực 3" > "Khu vực 2").

- [ ] **1.4. Kỹ thuật Đặc trưng (Feature Engineering)**
    - [ ] Tạo ít nhất 2 đặc trưng mới, ví dụ:
        * `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
        * `HouseAge` = `YrSold` - `YearBuilt`
    -  **Kiến thức cần nắm:**
        * **Domain Knowledge (Kiến thức nghiệp vụ):** Giải thích tại sao
          `TotalSF` có thể là một tín hiệu dự đoán tốt hơn là 3 cột diện tích
          riêng lẻ.

- [ ] **1.5. Co giãn Đặc trưng (Feature Scaling)**
    - [ ] Áp dụng `StandardScaler` cho tất cả các đặc trưng số.
    -  **Kiến thức cần nắm:**
        * **Tại sao phải Scaling:** Giải thích rằng các mô hình nhạy cảm với
          khoảng cách (như Linear Regression, SVM, Ridge/Lasso) sẽ hoạt động
          kém nếu một đặc trưng có thang đo lớn (ví dụ: `GrLivArea` từ
          1000-5000) và đặc trưng khác có thang đo nhỏ (ví dụ: `FullBath` từ
          1-3).
        * **Mô hình nào KHÔNG cần:** Nêu được rằng các mô hình dựa trên cây
          (Decision Tree, Random Forest) không quan tâm đến scaling.

- [ ] **1.6. Phân chia Dữ liệu (Train/Test Split)**
    - [ ] Tách `train.csv` thành `X_train`, `X_val`, `y_train`, `y_val` (ví dụ:
      tỷ lệ 80/20).
    -  **Kiến thức cần nắm:**
        * **Mục đích:** Giải thích rằng bạn cần một tập validation (Validation
          Set) để kiểm tra xem mô hình có bị **overfitting** (học vẹt) trên tập
          train hay không. Đây là dữ liệu "lạ" mà mô hình chưa từng thấy.

---

##  Giai đoạn 2: Xây dựng, Đánh giá & Tinh chỉnh Mô hình

Trọng tâm là so sánh và lựa chọn mô hình một cách có cơ sở.

- [ ] **2.1. Xây dựng `Pipeline`**
    - [ ] Đóng gói tất cả các bước Tiền xử lý (1.2 đến 1.5) vào một
      `ColumnTransformer` và `Pipeline`.
    -  **Kiến thức cần nắm:**
        * **Tại sao dùng Pipeline:** Giải thích 2 lý do chính:
            1.  **Tiện lợi:** Tự động hóa toàn bộ quy trình.
            2.  **Ngăn chặn Rò rỉ Dữ liệu (Data Leakage):** Đảm bảo rằng bạn
                chỉ `fit` (học) `StandardScaler` hoặc `Imputer` trên tập train,
                và `transform` (áp dụng) trên tập validation.

- [ ] **2.2. Huấn luyện 6 Mô hình**
    - [ ] Tạo và `fit` (huấn luyện) 6 pipeline mô hình: Linear, Ridge, Lasso,
      SVM, Decision Tree, Random Forest, XGBoost.
    -  **Kiến thức cần nắm:**
        * **Giải thích 1 câu về từng mô hình:**
            * **Linear:** Tìm đường thẳng tuyến tính tốt nhất.
            * **Ridge/Lasso:** Giống Linear, nhưng có **Regularization** (điều
              chuẩn) để giảm overfitting. Phải biết sự khác biệt L1 (Lasso, có
              thể loại bỏ đặc trưng) và L2 (Ridge).
            * **SVM:** Tìm "lề" (margin) tốt nhất để phân chia dữ liệu.
            * **Decision Tree:** Một loạt các quy tắc "if-then-else". Rất dễ bị
              overfitting.
            * **Random Forest:** (Bagging) Kết hợp nhiều Decision Tree độc lập
              để giảm overfitting.
            * **XGBoost:** (Boosting) Xây dựng các cây một cách tuần tự, cây
              sau sửa lỗi cho cây trước. Thường cho hiệu suất cao nhất.

- [ ] **2.3. Đánh giá Mô hình**
    - [ ] Dự đoán (`predict`) trên tập validation.
    - [ ] **Quan trọng:** Chuyển đổi ngược giá trị dự đoán về $ (dùng
      `np.expm1`).
    - [ ] Tính toán **RMSE** và **R²** cho cả 6 mô hình.
    - [ ] Tạo một bảng so sánh kết quả.
    -  **Kiến thức cần nắm:**
        * **RMSE (Root Mean Squared Error):** Giải thích ý nghĩa của nó (ví dụ:
          "RMSE của $25,000 nghĩa là trung bình, dự đoán của mô hình sai lệch
          $25,000 so với giá thực"). **Càng thấp càng tốt**.
        * **R-squared (R²):** Giải thích ý nghĩa (ví dụ: "R² = 0.90 nghĩa là mô
          hình của tôi giải thích được 90% sự biến động của giá nhà"). **Càng
          cao càng tốt**.
        * **Lý do chuyển đổi ngược:** Bạn phải báo cáo sai số (RMSE) theo đơn
          vị tiền tệ ($), không phải theo đơn vị logarit, để giảng viên/người
          dùng hiểu được.

- [ ] **2.4. Tinh chỉnh (Tuning) & Phân tích**
    - [ ] Chọn 2-3 mô hình tốt nhất (ví dụ: Ridge, RF, XGBoost).
    - [ ] Dùng `GridSearchCV` hoặc `RandomizedSearchCV` để tìm siêu tham số tốt
      nhất.
    - [ ] Trích xuất **Feature Importance** (Tầm quan quan trọng của Đặc trưng)
      từ mô hình tốt nhất (RF hoặc XGBoost).
    -  **Kiến thức cần nắm:**
        * **Hyperparameter (Siêu tham số):** Phân biệt nó với "parameter" (tham
          số). (Hyperparameter là cái bạn chọn *trước khi* huấn luyện, ví dụ:
          `alpha` trong Ridge; Parameter là cái mô hình *học được*, ví dụ: hệ
          số `coefficient`).
        * **Feature Importance:** Chỉ ra **Top 5 đặc trưng quan trọng nhất**
          (ví dụ: `OverallQual`, `GrLivArea`...) và giải thích ý nghĩa kinh
          doanh của chúng ("Điều này cho thấy chất lượng tổng thể là yếu tố ảnh
          hưởng giá mạnh nhất...").

- [ ] **2.5. Lưu Mô hình**
    - [ ] Lưu **toàn bộ pipeline** tốt nhất ra tệp `.joblib` (ví dụ:
      `final_model.joblib`).
    -  **Kiến thức cần nắm:**
        * **Tại sao lưu Pipeline:** Giải thích rằng bạn phải lưu cả các bước
          tiền xử lý. Nếu chỉ lưu mô hình XGBoost, nó sẽ không biết cách xử lý
          dữ liệu thô (văn bản) từ người dùng trên web.

---

##  Giai đoạn 3: Viết Báo cáo (LaTeX)

Bây giờ bạn đã có tất cả kết quả để viết.

- [ ] **3.1. Viết Nội dung Thô**
    - [ ] **Introduction:** Đặt vấn đề, mục tiêu (so sánh 6 mô hình...).
    - [ ] **Methodology:** Mô tả Giai đoạn 1 (tại sao `log`, tại sao OHE, tại
      sao `Pipeline`...). Liệt kê 6 mô hình và 2 thước đo (RMSE, R²).
    - [ ] **Results:** Trình bày và phân tích các kết quả.
    - [ ] **Conclusion:** Tóm tắt và đề xuất kinh doanh.
    -  **Kiến thức cần nắm:**
        * **Khả năng kể chuyện:** Bài báo cáo phải là một câu chuyện có logic:
          "Vấn đề là X (lệch). Chúng tôi giải quyết bằng Y (log). Chúng tôi thử
          6 mô hình. Kết quả Z (XGBoost) là tốt nhất. Nó cho thấy A và B
          (`OverallQual`, `GrLivArea`) là quan trọng nhất."

- [ ] **3.2. Chèn Bảng & Biểu đồ (Assets)**
    - [ ] Chèn biểu đồ `SalePrice` (trước/sau log).
    - [ ] Chèn **Bảng so sánh RMSE/R²** của 6 mô hình.
    - [ ] Chèn **Biểu đồ Feature Importance** (Top 10-15 đặc trưng).
    -  **Kiến thức cần nắm:**
        * **Giải thích Hình ảnh:** Đảm bảo bạn có thể giải thích từng chi tiết
          trong mọi biểu đồ bạn chèn vào. Giảng viên sẽ hỏi về chúng.

---

## 💻 Giai đoạn 4: Phân tích Bổ sung & Triển khai Web (Bonus)

Đây là phần "ăn điểm" cộng.

- [ ] **4.1. Phân tích Chuỗi thời gian**
    - [ ] Gộp (aggregate) `Median_SalePrice` theo `YrSold` và `MoSold`.
    - [ ] Vẽ biểu đồ đường (line chart) để xem xu hướng.
    -  **Kiến thức cần nắm:**
        * **Góc nhìn Vĩ mô vs. Vi mô:** Giải thích rằng mô hình Hồi quy của bạn
          là "vi mô" (định giá 1 căn nhà). Phân tích này là "vĩ mô" (xem xu
          hướng toàn thị trường). Bạn có thể thấy được ảnh hưởng của khủng
          hoảng 2008 không?

- [ ] **4.2. Xây dựng Web App (Flask)**
    - [ ] Backend (Flask): Tải tệp `.joblib` và tạo 1 endpoint `/predict`.
    - [ ] Frontend (HTML): Tạo 3 tab như đã thiết kế.
    - [ ] **Tab 1:** Tạo form nhập liệu. **Cực kỳ quan trọng:** Đảm bảo các giá
      trị trong dropdown (ví dụ: `Neighborhood`) khớp 100% với các giá trị đã
      huấn luyện.
    - [ ] **Tab 2:** Nhúng (embed) biểu đồ Feature Importance (dưới dạng ảnh).
    - [ ] **Tab 3:** Nhúng (embed) biểu đồ Chuỗi thời gian (dưới dạng ảnh).
    -  **Kiến thức cần nắm:**
        * **Kiến trúc Web:** Giải thích cách Frontend (HTML/JS) gửi yêu cầu
          (request) đến Backend (Flask), Backend dùng pipeline để dự đoán và
          gửi kết quả (response) trở lại.
        * **Lý do dùng Dropdown:** Giải thích tại sao bạn dùng dropdown cho
          `Neighborhood` thay vì ô nhập text. (Gợi ý: Để **ràng buộc đầu vào**
          của người dùng, ngăn họ nhập một giá trị "lạ" mà pipeline không biết
          cách xử lý).
