# Giải thích: Thực nghiệm trên tập SynTReN (test_syntren.py)

SynTReN là trình mô phỏng mạng điều hòa gen. Trong thực nghiệm này, chúng ta sử dụng cấu hình 20 nút (n_vars=20), đòi hỏi các tham số mạnh mẽ hơn so với các tập dữ liệu nhỏ.

## Các tham số quan trọng và Lý do chọn (Rationale)

### 1. `n_vars`: 20 & `N`: 2000
- **Lý do**: Tăng số lượng nút lên 20 để kiểm tra khả năng mở rộng (scalability) của mô hình. Với 2000 mẫu dữ liệu, mô hình có đủ thông tin để học các tương tác đa biến mà không bị quá khớp (overfitting).

### 2. `n_clusters`: 4
- **Lý do**: Mặc dù SynTReN là dữ liệu giả lập, nhưng cấu trúc mạng gen thường có các nhóm gen hoạt động đồng bộ. Việc chọn 4 cụm giúp mô hình hóa các trạng thái biểu hiện gen ngầm định bên trong mạng lưới.

### 3. `flow_bins`: 10
- **Lý do**: Một giá trị trung bình (10) đủ để NSF mô hình hóa các phân phối nhiễu phi tuyến sinh ra từ các hàm Hill và Michaelis-Menten trong trình mô phỏng SynTReN.

### 4. `lda_hsic`: 0.05
- **Lý do**: Đối với mạng lưới 20 nút, nguy cơ nhầm lẫn hướng cạnh tăng cao. Vì vậy, trọng số HSIC được tăng lên 0.05 (so với 0.03 ở Sachs) để áp đặt điều kiện độc lập khắt khe hơn, giúp xác định hướng nhân quả chính xác hơn trong mạng gen phức tạp.

### 5. `stage1_epochs`: 40 & `stage2_epochs`: 25
- **Lý do**: Do số lượng nút tăng (20 nodes), không gian tìm kiếm đồ thị rộng hơn nhiều. Chúng ta tăng số lượng vòng lặp (epochs) ở cả hai giai đoạn để đảm bảo thuật toán Augmented Lagrangian có đủ thời gian để hội tụ về một DAG hợp lệ.

### 6. `l1_stage1`: 0.001 & `l1_stage2`: 0.012
- **Lý do**: Giữ nguyên chiến lược "Khám phá - Tinh chỉnh". Mức L1 mạnh ở giai đoạn 2 là cần thiết để loại bỏ các cạnh giả sinh ra do độ tương quan cao giữa các dòng gen trong mạng lưới lớn.

---
**Kết luận**: Các tham số được lựa chọn dựa trên nguyên tắc cân bằng giữa khả năng biểu thị (Expressivity) và tính thưa của đồ thị (Sparsity), đồng thời tận dụng lợi thế của NSF và HSIC để xử lý các đặc trưng riêng biệt của từng loại dữ liệu.
