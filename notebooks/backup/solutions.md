1. Xử lý "Dữ liệu ngoại lai" (Outliers) triệt để
Vấn đề: Trong tập dữ liệu, có những căn nhà diện tích cực lớn (>4000 sqft) nhưng giá bán lại rất rẻ. Đây là những điểm dị biệt (có thể do bán gấp, giao dịch nội bộ...). Nếu để lại, chúng sẽ làm "lệch" đường hồi quy, khiến mô hình học sai quy luật chung.

Cách làm: train_df.drop(...) loại bỏ các điểm này dựa trên biểu đồ tương quan giữa GrLivArea và SalePrice.

Hiệu quả: Giúp mô hình ổn định hơn, không bị nhiễu bởi các trường hợp cá biệt.

2. Chuẩn hóa biến mục tiêu (Target Transformation)
Vấn đề: Giá nhà (SalePrice) thường bị lệch phải (right-skewed) - tức là có nhiều nhà giá thấp/trung bình và rất ít nhà giá cực cao. Các mô hình hồi quy tuyến tính hoạt động kém hiệu quả trên dữ liệu lệch.

Cách làm: Áp dụng hàm logarit np.log1p(train_df["SalePrice"]).

Hiệu quả: Biến đổi phân phối giá nhà về dạng Phân phối chuẩn (Bell curve). Điều này giúp việc dự đoán trở nên tuyến tính và dễ dàng hơn cho thuật toán. Cuối cùng, dùng np.expm1 để đổi lại giá trị thực khi nộp bài.

3. Xử lý giá trị thiếu (Missing Values) theo ngữ cảnh
Thay vì điền tất cả bằng trung bình (mean), mình phân loại kỹ:

Biến phân loại (Category): Ví dụ PoolQC (Chất lượng hồ bơi) bị thiếu không phải là lỗi, mà nghĩa là "Không có hồ bơi". Mình điền là "None".

Biến số (Numeric): Ví dụ GarageArea bị thiếu nghĩa là diện tích bằng 0. Mình điền số 0.

Biến đặc biệt (LotFrontage): Mặt tiền nhà thường giống nhau trong cùng một khu phố. Mình điền giá trị thiếu bằng Trung vị (Median) của từng khu phố (Neighborhood) thay vì trung vị của toàn bộ tập dữ liệu. Cách này chính xác hơn nhiều.

4. Kỹ thuật Feature Engineering (Tạo đặc trưng mới)
Đây là bước quan trọng nhất để tăng điểm số:

Tạo biến TotalSF (Tổng diện tích): cộng tổng diện tích hầm + tầng 1 + tầng 2. Trong bất động sản, tổng diện tích sử dụng là yếu tố quan trọng nhất quyết định giá nhà. Biến này thường có tương quan mạnh nhất với giá.

Chuyển đổi kiểu dữ liệu: MSSubClass (loại nhà) hay MoSold (tháng bán) tuy là số nhưng thực chất là phân loại (tháng 1 không nhỏ hơn tháng 12 về mặt giá trị). Mình chuyển chúng sang dạng chuỗi (String) để mô hình hiểu đúng bản chất.

Label Encoding: Với các biến có thứ bậc (như ExterQual: Tốt > Khá > Trung bình > Kém), mình mã hóa thành số thứ tự để giữ lại thông tin về cấp độ chất lượng.

5. Xử lý độ lệch của Features (Box-Cox Transformation)
Vấn đề: Không chỉ giá nhà, mà các đặc trưng đầu vào (diện tích, kích thước...) cũng bị lệch.

Cách làm: Sử dụng scipy.special.boxcox1p cho tất cả các biến số có độ lệch (skew) > 0.75.

Hiệu quả: Giúp dữ liệu "mượt" hơn, gần với phân phối chuẩn hơn, giúp các mô hình như Lasso hay Ridge hoạt động cực tốt.

6. Mô hình Stacking (Stacking Ensemble)
Thay vì chỉ dùng một mô hình, mình sử dụng kiến trúc Stacking:

Tầng 1 (Base Models):

Lasso & ElasticNet: Rất giỏi trong việc lọc bỏ các biến không quan trọng (Feature Selection) và nắm bắt các quan hệ tuyến tính. Dùng RobustScaler để chống lại các outliers còn sót lại.

Gradient Boosting & LightGBM: Các mô hình cây (Tree-based) rất mạnh trong việc học các mối quan hệ phi tuyến tính phức tạp.

Tầng 2 (Meta Model - XGBoost): Học từ dự đoán của các mô hình tầng 1 để đưa ra kết quả cuối cùng. Nó đóng vai trò như một "trọng tài", biết khi nào nên tin Lasso, khi nào nên tin LightGBM.

Tóm lại: Cách tối ưu này là sự kết hợp giữa việc "làm sạch dữ liệu thủ công" (dựa trên hiểu biết về bất động sản) và "sức mạnh toán học" (Box-Cox, Stacking) để đạt kết quả tốt nhất.