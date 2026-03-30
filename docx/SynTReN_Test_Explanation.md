# Giải thích Thực nghiệm: Tập dữ liệu SynTReN (test_syntren.py)

Tệp này thực hiện việc đánh giá trên dữ liệu mô phỏng mạng điều hòa gene (Gene Regulatory Network) với 20 nút.

## 1. Tại sao chọn các tham số này?

Trong `test_syntren.py`, các tham số được tối ưu hóa cho cấu trúc đồ thị lớn hơn:

- **`n_vars: 20`**: 
    - *Lý do*: Đây là ngưỡng thử nghiệm độ phức tạp trung bình của mô hình. 20 nút tạo ra không gian tìm kiếm đồ thị rộng hơn nhiều so với 11 nút của tập Sachs.
- **`n_clusters: 4`**: 
    - *Lý do*: Dữ liệu mô phỏng từ SynTReN thường có các ngữ cảnh ít đa dạng hơn dữ liệu thực tế từ tế bào sống, nên 4 cụm là đủ để mô hình hóa cấu trúc nhiễu giả lập.
- **`lda_hsic: 0.05`**: 
    - *Lý do*: Chúng ta tăng hình phạt độc lập (lda_hsic) lên cao hơn so với tập Sachs. Điều này rất quan trọng khi số lượng nút tăng lên, giúp mô hình khắt khe hơn trong việc loại bỏ các cạnh giả (False Positives) do tương quan ngẫu nhiên.
- **`stage1_epochs: 40` / `stage2_epochs: 25`**: 
    - *Lý do*: Do số nút tăng lên 20, mô hình cần nhiều thời gian hơn để các tham số ma trận kề $W$ hội tụ và để ràng buộc Acyclicity (đồ thị không chu trình) được thực hiện triệt để.
- **`flow_bins: 10`**: 
    - *Lý do*: Nhiễu trong SynTReN thường tuân theo các quy luật toán học (log-normal hoặc gauss), nên 10 thùng là đủ để mô hình hóa mà vẫn đảm bảo tốc độ tính toán.

## 2. Điểm khác biệt so với tập Sachs
Điểm mấu chốt của cấu hình này là sự **"Khắt khe"**. Khi mạng lưới càng lớn, nguy cơ xuất hiện vòng lặp (Cycles) và cạnh giả càng cao. Do đó, các tham số về Epoch và lda_hsic đều được đẩy cao hơn để đảm bảo tính ổn định của cấu trúc DAG cuối cùng.
