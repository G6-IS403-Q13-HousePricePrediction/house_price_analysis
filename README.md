# Đồ án Phân tích & Dự đoán giá nhà 

Đây là đồ án phân tích dữ liệu và xây dựng mô hình Machine Learning để dự đoán
giá nhà từ bộ dữ liệu Kaggle (Ames Housing).

Dự án này sử dụng `uv` để quản lý môi trường, `Jupyterlab` cho phân tích,
`Flask` để triển khai một web app demo, `Jupytext` để đồng bộ notebook với file
Python, và `LaTeX` để viết báo cáo.

## 🚀 Công nghệ sử dụng

* **Quản lý Môi trường & Gói:** `uv`
* **Phân tích & Mô hình:** `Python`, `pandas`, `scikit-learn`, `xgboost`
* **Notebook:** `JupyterLab`, `Jupytext`
* **Báo cáo:** `LaTeX`
* **Triển khai Web App:** `Flask`
* **Hỗ trợ Editor:** `python-lsp-server` (cho Vim/Neovim), `.vscode/` (cho VSCode)

## 📁 Cấu trúc Thư mục

Đây là cấu trúc thư mục tiêu chuẩn cho dự án:

```bash
DoAnPhanTichDuLieu/
├── .venv/                   <-- Môi trường ảo (do uv tạo ra, nằm trong .gitignore)
├── .vscode/                 <-- Cấu hình riêng cho VSCode
│   ├── extensions.json
│   └── settings.json
├── app/                     <-- Thư mục Web App (Flask)
│   ├── static/
│   ├── templates/
│   └── app.py
├── data/                    <-- Dữ liệu thô
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── models/                  <-- Model đã huấn luyện
│   └── house_price_model.joblib
├── notebooks/               <-- Nơi làm việc chính (Phân tích)
│   ├── House_Price_Analysis.ipynb
│   └── House_Price_Analysis.py
├── report/                  <-- Báo cáo LaTeX
│   ├── images/              <-- Chứa các biểu đồ do notebook tạo ra
│   └── main.tex             <-- File LaTeX báo cáo chính
│
├── .gitignore               <-- File bỏ qua (.venv, .aux, .log...)
├── checklist.md             <-- File theo dõi tiến độ dự án
├── pyproject.toml           <-- File "trái tim" của dự án (quản lý dependencies)
└── README.md                <-- Chính là file này
```

## 🛠️ Hướng dẫn Cài đặt

### Yêu cầu Tiên quyết

Bạn chỉ cần cài đặt uv một lần duy nhất trên hệ thống của mình.

```bash
# Cài đặt uv (nếu chưa có)
pip install uv
```

### Các bước cài đặt dự án

1. Clone dự án

```
git clone https://github.com/G6-IS403-Q13-HousePricePrediction/house_price_analysis
cd house_price_analysis
```

2. **Đồng bộ môi trường (Sync Environment)** Lệnh này sẽ tự động đọc file
   `pyproject.toml`, tạo một môi trường ảo (`.venv`) và cài đặt tất cả các thư viện
   cần thiết.

```bash
uv sync --all
```

3. **Kích hoạt Môi trường Ảo** Luôn kích hoạt môi trường trước khi làm việc.
  - macOS/Linux (bash/zsh):

  ```bash
  source .venv/bin/activate
  ```

  - Windows (Command Prompt):

  ```bash
  .venv\Scripts\activate
  ```

  - Windows (PowerShell):

  ```bash
  .venv\Scripts\Activate.ps1
  ```

## 🏃 Quy trình Làm việc

1. Phân tích Dữ liệu (Jupyter)

Sau khi đã kích hoạt môi trường, khởi động máy chủ notebook:

```bash
jupyter lab
```

Trình duyệt của bạn sẽ tự động mở. Hãy vào thư mục `notebooks/` để bắt đầu làm
việc. Các biểu đồ bạn tạo ra nên được lưu vào thư mục `report/images/`.

2. Viết Báo cáo (LaTeX)

Bạn có thể chỉnh sửa file `report/main.tex` bằng trình soạn thảo LaTeX yêu thích
của mình. Các hình ảnh sẽ được lấy từ `report/images/`.

3. Chạy Web App (Flask)

Để chạy ứng dụng web demo (sau khi đã huấn luyện và lưu model vào thư mục `models/`):

```bash
# Đảm bảo môi trường đã được kích hoạt
cd app
flask run
```

Sau đó, truy cập `http://127.0.0.1:5000` (hoặc `http://localhost:5000`) trên trình duyệt của bạn.

---

## 💻 Hướng dẫn Cấu hình Editor

### Dành cho Người dùng VSCode

Dự án này đã được cấu hình sẵn cho VSCode.

1.  **Cài đặt Tiện ích (Extensions):**
      - Sau khi mở dự án, VSCode sẽ hiển thị một thông báo ở góc dưới bên phải,
        đề xuất "Install Recommended Extensions" (dựa trên file
        `.vscode/extensions.json`).
      - Hãy nhấn "Install" để cài đặt **Python (Microsoft)** và **Jupytext**.
        (Bạn cũng có thể cài thêm `LaTeX Workshop` để soạn thảo file `.tex`).

2.  **Chọn Interpreter:**
      - VSCode sẽ tự động chọn môi trường ảo `.venv` của dự án (dựa trên file
        `.vscode/settings.json`).
      - Bạn có thể xác nhận điều này bằng cách mở một file `.py` và nhìn vào
        góc dưới bên phải màn hình, bạn sẽ thấy `Python 3.x.x ('.venv')`.

3.  **Làm việc với Jupytext:**
      - Nhờ tiện ích Jupytext, bạn có thể mở file
        `notebooks/House_Price_Analysis.py` và VSCode sẽ tự động hiển thị nó
        dưới dạng Notebook (giống như file `.ipynb`).
       
### Dành cho Người dùng Vim / Neovim

Dự án này đã bao gồm `python-lsp-server` (Python Language Server) trong `dev-dependencies`.
Bạn chỉ cần cấu hình Vim/Neovim của mình để sử dụng `python-lsp-server` làm Language Server
cho Python.

Bạn có thể thoải mái chỉnh sửa file `notebooks/House_Price_Analysis.py` bằng
Vim. Khi bạn lưu lại, `Jupytext` (chạy ngầm bởi JupyterLab) sẽ tự động đồng bộ
các thay đổi đó vào file `.ipynb` tương ứng.
