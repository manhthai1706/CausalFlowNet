# Giải thích: Thực nghiệm trên tập Sachs (test_sachs.py)

Tập dữ liệu Sachs là một bộ dữ liệu sinh học thực tế đo lường nồng độ protein trong các tế bào đơn lẻ. Việc lựa chọn tham số trong `test_sachs.py` được tinh chỉnh để phù hợp với tính chất nhiễu và phức tạp của dữ liệu sinh học.

## Các tham số quan trọng và Lý do chọn (Rationale)

### 1. `n_clusters`: 5
- **Lý do**: Tập dữ liệu Sachs bao gồm dữ liệu từ nhiều điều kiện can thiệp khác nhau (interventions). Việc chọn 5 cụm (clusters) giúp mô hình GMM Prior bắt được các "phân tầng" hoặc trạng thái tế bào khác nhau, từ đó mô hình hóa nhiễu chính xác hơn cho từng nhóm.

### 2. `flow_bins`: 12
- **Lý do**: Dữ liệu sinh học thực tế thường không tuân theo phân phối Gauss và có các đặc trưng phi tuyến mạnh. Việc tăng số lượng thùng (bins) lên 12 giúp hàm Spline có độ phân giải cao hơn để khớp chính xác với các phân phối nhiễu phức tạp.

### 3. `lda_hsic`: 0.03
- **Lý do**: Một giá trị vừa đủ (0.03) để ép buộc tính độc lập giữa nhiễu và biến cha mà không làm lấn át hàm mục tiêu Likelihood. Điều này cực kỳ quan trọng để xác định đúng hướng của dòng tín hiệu protein.

### 4. `stage1_epochs`: 30 & `stage2_epochs`: 20
- **Lý do**: Giai đoạn 1 (Discovery) cần đủ thời gian để các cạnh tiềm năng lộ diện dưới ràng buộc L1 thấp. Giai đoạn 2 (Refinement) tập trung vào việc tinh chỉnh trọng số và loại bỏ cạnh giả dựa trên ràng buộc L1 mạnh hơn và điều kiện DAG.

### 5. `l1_stage1`: 0.001 & `l1_stage2`: 0.012
- **Lý do**: 
  - Giai đoạn 1 dùng `0.001` (rất thấp) để cho phép mô hình tự do khám phá tất cả các kết nối có thể có.
  - Giai đoạn 2 tăng lên `0.012` (mạnh hơn 12 lần) để thực hiện "cắt tỉa" (pruning) các cạnh yếu, giữ lại cấu trúc đồ thị tối giản và chính xác nhất.

### 6. `threshold`: 0.05
- **Lý do**: Sau khi huấn luyện, các cạnh có trọng số tuyệt đối dưới 0.05 sẽ bị loại bỏ hoàn toàn. Ngưỡng này giúp cân bằng giữa việc giữ lại các tương tác sinh học thực sự và loại bỏ nhiễu đồ thị.
